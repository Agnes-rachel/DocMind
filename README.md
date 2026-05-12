#  DocMind — AI Document Intelligence

DocMind is a full-stack Retrieval-Augmented Generation (RAG) system 
that lets you chat with your PDF documents using AI. Upload any PDF, 
ask questions in natural language, and get precise answers grounded 
in your document's actual content — no hallucinations.

## What it does

- Upload one or multiple PDF documents
- Ask any question about your documents
- AI retrieves the most relevant passages using semantic search
- Generates accurate, source-cited answers using LLaMA 3.3 (via Groq)
- Shows exactly which part of the document the answer came from

##  How it works (RAG Pipeline)

1. PDF Parsing    — extracts raw text from uploaded PDFs
2. Chunking       — splits text into overlapping segments
3. TF-IDF Search  — finds the most relevant chunks for your question
4. LLM Generation — LLaMA 3.3 answers strictly from retrieved context

##  Tech Stack

| Layer      | Technology                        |
|------------|-----------------------------------|
| Frontend   | HTML, CSS, Vanilla JavaScript     |
| Backend    | Python, FastAPI                   |
| AI Model   | LLaMA 3.3 70B via Groq API (free) |
| Search     | TF-IDF + Cosine Similarity        |
| PDF Parse  | pypdf                             |
| Server     | Uvicorn                           |

##  Quick Start

# Install dependencies
cd backend
pip install -r requirements.txt

# Add your free Groq API key to backend/.env
GROQ_API_KEY=your-key-here

# Start the backend
uvicorn main:app --reload --port 8000

# Open frontend/index.html with Live Server
