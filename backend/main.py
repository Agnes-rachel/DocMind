"""
RAG System Backend - FastAPI (Lightweight Version)
"""
from dotenv import load_dotenv
load_dotenv()
import os
import uuid
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()  # reads .env file automatically

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_engine import RAGEngine

app = FastAPI(title="RAG System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGEngine()
document_registry: dict = {}


class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    query: str
    processing_time: float

class DocumentInfo(BaseModel):
    id: str
    filename: str
    size: int
    chunks: int
    status: str
    uploaded_at: float


@app.get("/health")
def health():
    return {"status": "ok", "engine": "RAG v1.0 (lightweight)"}


@app.post("/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(400, "File too large. Max 50MB.")
    doc_id = str(uuid.uuid4())
    try:
        tmp_path = Path(f"/tmp/{doc_id}_{file.filename}")
        tmp_path.write_bytes(content)
        chunks = rag.ingest_document(str(tmp_path), doc_id, file.filename)
        doc_info = {
            "id": doc_id,
            "filename": file.filename,
            "size": len(content),
            "chunks": chunks,
            "status": "ready",
            "uploaded_at": time.time(),
        }
        document_registry[doc_id] = doc_info
        tmp_path.unlink(missing_ok=True)
        return DocumentInfo(**doc_info)
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    if not document_registry:
        raise HTTPException(400, "No documents uploaded yet.")
    start = time.time()
    try:
        result = rag.query(query=req.query, doc_ids=req.document_ids, top_k=req.top_k)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            query=req.query,
            processing_time=round(time.time() - start, 2),
        )
    except Exception as e:
        raise HTTPException(500, f"Query failed: {str(e)}")


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    return [DocumentInfo(**d) for d in document_registry.values()]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if doc_id not in document_registry:
        raise HTTPException(404, "Document not found.")
    rag.delete_document(doc_id)
    del document_registry[doc_id]
    return {"deleted": doc_id}


@app.get("/stats")
def get_stats():
    return {
        "total_documents": len(document_registry),
        "total_chunks": sum(d["chunks"] for d in document_registry.values()),
        "vector_store": rag.collection_count(),
    }
