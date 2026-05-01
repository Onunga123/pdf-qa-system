import streamlit as st
import faiss
import pickle
import numpy as np
import tempfile
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.set_page_config(page_title='PDF Q&A System', page_icon='📄', layout='wide')

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    llm = pipeline('question-answering', model='deepset/roberta-base-squad2')
    return embedding_model, llm

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + '
'
    return text

def build_index(text, embedding_model):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = embedding_model.encode(chunks)
    embeddings_f32 = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings_f32.shape[1])
    index.add(embeddings_f32)
    return index, chunks

def ask_question(question, index, chunks, embedding_model, llm, top_k=2):
    q_embedding = embedding_model.encode([question]).astype('float32')
    distances, indices = index.search(q_embedding, top_k)
    context_chunks = [chunks[i] for i in indices[0]]
    context = ' '.join(context_chunks)
    result = llm(question=question, context=context)
    return result['answer'], round(result['score'] * 100, 1), context_chunks

st.title('PDF Q&A System')
st.markdown('Upload any PDF and ask questions about its contents using AI')
st.markdown('---')

with st.spinner('Loading AI models... this may take 1-2 minutes on first load'):
    embedding_model, llm = load_models()
st.success('Models ready!')

left, right = st.columns([1, 2])

with left:
    st.subheader('Upload PDF')
    uploaded_file = st.file_uploader('Choose a PDF file', type='pdf')

    if uploaded_file:
        with st.spinner('Processing PDF...'):
            text = extract_text(uploaded_file)
            index, chunks = build_index(text, embedding_model)
        st.success(f'PDF processed! {len(chunks)} chunks indexed.')
        st.info(f'Characters: {len(text)}')

with right:
    st.subheader('Ask a Question')
    if uploaded_file is None:
        st.warning('Please upload a PDF first using the panel on the left.')
    else:
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        question = st.text_input('Type your question here...', key='question_input')
        if question:
            with st.spinner('Searching and generating answer...'):
                answer, confidence, context_chunks = ask_question(
                    question, index, chunks, embedding_model, llm
                )
            st.session_state.chat_history.append({
                'question': question,
                'answer': answer,
                'confidence': confidence
            })
        if st.session_state.chat_history:
            for item in reversed(st.session_state.chat_history):
                st.markdown(f'**Q:** {item["question"]}')
                conf_color = 'green' if item['confidence'] > 50 else 'orange' if item['confidence'] > 20 else 'red'
                st.markdown(f'**A:** {item["answer"]}')
                st.markdown(f'Confidence: :{conf_color}[{item["confidence"]}%]')
                st.markdown('---')

st.caption('Built with SentenceTransformers, FAISS, RoBERTa, LangChain & Streamlit')
