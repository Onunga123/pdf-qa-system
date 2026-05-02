import streamlit as st
import faiss
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

st.set_page_config(page_title='PDF Q&A System', page_icon='📄', layout='wide')

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    model_name = 'deepset/roberta-base-squad2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    return embedding_model, tokenizer, model

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return ' '.join(pages)

def build_index(text, embedding_model):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = np.array(embedding_model.encode(chunks)).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

def ask_question(question, index, chunks, embedding_model, tokenizer, model, top_k=3):
    q_emb = np.array(embedding_model.encode([question])).astype('float32')
    distances, indices = index.search(q_emb, top_k)
    context = ' '.join([chunks[i] for i in indices[0]])
    encoding = tokenizer(
        question,
        context,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )
    offset_mapping = encoding.pop('offset_mapping')
    with torch.no_grad():
        outputs = model(**encoding)
    start = int(torch.argmax(outputs.start_logits))
    end = int(torch.argmax(outputs.end_logits))
    if end < start:
        end = start
    offsets = offset_mapping[0].tolist()
    start_char = offsets[start][0]
    end_char = offsets[end][1]
    question_length = len(question) + 4
    if start_char >= question_length:
        answer = context[start_char - question_length: end_char - question_length].strip()
    else:
        answer = context[start_char:end_char].strip()
    if not answer or len(answer) < 2:
        answer = 'Could not find a specific answer. Try rephrasing your question.'
    confidence = round(float(torch.max(torch.softmax(outputs.start_logits, dim=1))) * 100, 1)
    return answer, confidence, context

st.title('PDF Q&A System')
st.markdown('Upload any PDF and ask questions about its contents using AI')
st.markdown('---')

with st.spinner('Loading AI models... may take 1-2 minutes on first load'):
    embedding_model, tokenizer, model = load_models()
st.success('Models ready!')

left, right = st.columns([1, 2])

with left:
    st.subheader('Upload PDF')
    uploaded_file = st.file_uploader('Choose a PDF file', type='pdf')
    if uploaded_file:
        with st.spinner('Processing PDF...'):
            text = extract_text(uploaded_file)
            index, chunks = build_index(text, embedding_model)
        st.success(f'Done! {len(chunks)} chunks indexed.')
        st.info(f'Characters extracted: {len(text)}')

with right:
    st.subheader('Ask a Question')
    if uploaded_file is None:
        st.warning('Please upload a PDF first.')
    else:
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        question = st.text_input('Type your question here...', key='q')
        if question:
            with st.spinner('Thinking...'):
                answer, confidence, context = ask_question(
                    question, index, chunks, embedding_model, tokenizer, model
                )
            st.session_state.chat_history.append({
                'question': question,
                'answer': answer,
                'confidence': confidence,
                'context': context
            })
        for item in reversed(st.session_state.chat_history):
            st.markdown(f'**Q:** {item["question"]}')
            st.markdown(f'**A:** {item["answer"]}')
            color = 'green' if item['confidence'] > 50 else 'orange' if item['confidence'] > 20 else 'red'
            st.markdown(f'Confidence: :{color}[{item["confidence"]}%]')
            with st.expander('View retrieved context'):
                st.caption(item['context'])
            st.markdown('---')

st.caption('Built with SentenceTransformers, FAISS, RoBERTa, LangChain & Streamlit')
