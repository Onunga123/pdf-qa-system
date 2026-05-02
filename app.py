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
    inputs = tokenizer(
        question,
        context,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )
    offsets = inputs.pop('offset_mapping')[0].tolist()
    seq_ids = inputs.sequence_ids(0)
    with torch.no_grad():
        outputs = model(**inputs)
    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]
    context_start = seq_ids.index(1)
    context_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)
    start_logits[:context_start] = float('-inf')
    start_logits[context_end + 1:] = float('-inf')
    end_logits[:context_start] = float('-inf')
    end_logits[context_end + 1:] = float('-inf')
    best_start = int(torch.argmax(start_logits))
    best_end = int(torch.argmax(end_logits))
    if best_end < best_start:
        best_end = best_start
    start_char = offsets[best_start][0]
    end_char = offsets[best_end][1]
    context_offset = len(question) + 4
    answer_start = max(0, start_char - context_offset)
    answer_end = max(0, end_char - context_offset)
    answer = context[answer_start:answer_end].strip()
    if not answer or len(answer) < 2:
        answer = 'Could not find a specific answer. Try rephrasing.'
    confidence = round(float(torch.softmax(outputs.start_logits, dim=1)[0][best_start]) * 100, 1)
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
