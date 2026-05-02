# PDF Q&A System

An AI-powered document question-answering system built with RAG architecture.

## Live Demo
Add your Streamlit URL here after deployment

## What it does
- Upload any PDF document
- Ask questions in plain English
- AI retrieves relevant sections using semantic search
- RoBERTa extracts precise answers from retrieved context
- Shows confidence score and source context for every answer
- Maintains full chat history within session

## RAG Pipeline Architecture
```
Upload PDF
    -> Extract text (PyPDF)
    -> Split into chunks (LangChain)
    -> Generate embeddings (SentenceTransformers)
    -> Store in vector DB (FAISS)
    -> User asks question
    -> Embed question -> Search FAISS
    -> Retrieve top chunks
    -> Send to RoBERTa via HF Inference API
    -> Return answer + confidence
```

## Tools & Technologies
- Python, PyPDF, LangChain
- SentenceTransformers — all-MiniLM-L6-v2 (embeddings)
- FAISS — Facebook AI Similarity Search (vector database)
- RoBERTa — deepset/roberta-base-squad2 (answer extraction)
- Hugging Face Inference API
- Streamlit (deployed on Streamlit Cloud)

## How to Run Locally
1. Clone this repo
2. pip install -r requirements.txt
3. streamlit run app.py
4. Get a free HF token at huggingface.co/settings/tokens
