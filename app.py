import streamlit as st
import faiss
import numpy as np
import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title='PDF Q&A System', page_icon='📄', layout='wide')

HF_API_URL = 'https://router.huggingface.co/hf-inference/models/deepset/roberta-base-squad2'

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

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

def ask_question(question, index, chunks, embedding_model, hf_token, top_k=3):
    q_emb = np.array(embedding_model.encode([question])).astype('float32')
    distances, indices = index.search(q_emb, top_k)
    context = ' '.join([chunks[i] for i in indices[0]])
    headers = {'Authorization': f'Bearer {hf_token}'}
    payload = {'inputs': {'question': question, 'context': context}}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        answer = result.get('answer', '').strip()
        confidence = round(result.get('score', 0) * 100, 1)
        if not answer or len(answer) < 2:
            answer = 'Could not find a specific answer. Try rephrasing.'
        return answer, confidence, context
    else:
        return f'API error {response.status_code}: {response.text}', 0, context

st.title('PDF Q&A System')
st.markdown('Upload any PDF and ask questions about its contents using AI')
st.markdown('---')

with st.spinner('Loading embedding model...'):
    embedding_model = load_embedding_model()
st.success('Ready!')

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
    st.markdown('---')
    st.subheader('Hugging Face Token')
    hf_token = st.text_input('Enter your HF token', type='password')
    st.caption('Free token at huggingface.co/settings/tokens')

with right:
    st.subheader('Ask a Question')
    if uploaded_file is None:
        st.warning('Please upload a PDF first.')
    elif not hf_token:
        st.warning('Please enter your Hugging Face token in the sidebar.')
    else:
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        question = st.text_input('Type your question here...', key='q')
        if question:
            with st.spinner('Thinking...'):
                answer, confidence, context = ask_question(
                    question, index, chunks, embedding_model, hf_token
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

st.caption('Built with SentenceTransformers, FAISS, RoBERTa, HF Inference API & Streamlit')
