# PDF Q&A System

An AI-powered document question-answering system built with RAG architecture.

## Live Demo
*(Streamlit URL will go here after deployment)*

## What it does
- Upload any PDF document
- Ask questions in plain English
- AI retrieves relevant sections and generates accurate answers
- Shows confidence score for every answer
- Maintains chat history within the session

## Architecture — RAG Pipeline
1. PDF text extraction using PyPDF
2. Text chunking with LangChain RecursiveCharacterTextSplitter
3. Semantic embeddings using SentenceTransformers (all-MiniLM-L6-v2)
4. Vector storage and similarity search using FAISS
5. Answer generation using RoBERTa (deepset/roberta-base-squad2)

## Tools & Technologies
- Python, PyPDF, LangChain
- SentenceTransformers (Hugging Face)
- FAISS (Facebook AI Similarity Search)
- RoBERTa QA model (Hugging Face)
- Streamlit (deployed on Streamlit Cloud)
- Google Colab

## How to Run Locally
1. Clone this repo
2. pip install -r requirements.txt
3. streamlit run app.py
