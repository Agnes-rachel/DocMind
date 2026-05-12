"""
RAG Engine - Improved retrieval with better chunking and scoring
"""
 
import os
import re
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
 
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
 
load_dotenv()
 
STORE_PATH = Path("/tmp/rag_store.pkl")
 
 
class RAGEngine:
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 200
 
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        self.chunks: List[Dict] = []
        self._load()
 
    # ── INGESTION ──────────────────────────────────────────────────
 
    def ingest_document(self, filepath: str, doc_id: str, filename: str) -> int:
        text = self._extract_pdf_text(filepath)
        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("No text could be extracted from this PDF.")
        # Remove old chunks for this doc first (re-upload safe)
        self.chunks = [c for c in self.chunks if c["doc_id"] != doc_id]
        for i, chunk in enumerate(chunks):
            self.chunks.append({
                "id": f"{doc_id}_chunk_{i}",
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk,
            })
        self._save()
        return len(chunks)
 
    def _extract_pdf_text(self, filepath: str) -> str:
        reader = PdfReader(filepath)
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t.strip())
        return "\n\n".join(pages)
 
    def _chunk_text(self, text: str) -> List[str]:
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Split on sentence boundaries when possible
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) <= self.CHUNK_SIZE:
                current += " " + sentence
            else:
                if current.strip():
                    chunks.append(current.strip())
                # Start new chunk with overlap from previous
                overlap_start = max(0, len(current) - self.CHUNK_OVERLAP)
                current = current[overlap_start:] + " " + sentence
        if current.strip():
            chunks.append(current.strip())
        return chunks
 
    # ── RETRIEVAL + GENERATION ─────────────────────────────────────
 
    def query(self, query: str, doc_ids: Optional[List[str]] = None, top_k: int = 6) -> Dict:
        pool = self.chunks
        if doc_ids:
            pool = [c for c in self.chunks if c["doc_id"] in doc_ids]
        if not pool:
            return {"answer": "No documents found. Please upload a PDF first.", "sources": []}
 
        texts = [c["text"] for c in pool]
 
        # TF-IDF vectorizer with better settings
        vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            stop_words="english",
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, matrix)[0]
 
        top_indices = np.argsort(scores)[::-1][:top_k]
        retrieved = [(pool[i], float(scores[i])) for i in top_indices if scores[i] > 0]
 
        if not retrieved:
            # Fallback: return first few chunks as context even with 0 score
            retrieved = [(pool[i], 0.0) for i in range(min(3, len(pool)))]
 
        context_parts = []
        sources = []
        for i, (chunk, score) in enumerate(retrieved):
            context_parts.append(
                f"[Source {i+1} — File: {chunk['filename']}, Section {chunk['chunk_index']+1}/{chunk['total_chunks']}]\n{chunk['text']}"
            )
            sources.append({
                "filename": chunk["filename"],
                "doc_id": chunk["doc_id"],
                "chunk_index": chunk["chunk_index"],
                "relevance_score": round(score, 3),
                "excerpt": chunk["text"][:250] + ("..." if len(chunk["text"]) > 250 else ""),
            })
 
        answer = self._generate_answer(query, "\n\n---\n\n".join(context_parts))
        return {"answer": answer, "sources": sources}
 
    def _generate_answer(self, query: str, context: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=2048,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert document analyst. Your job is to answer questions "
                            "based ONLY on the provided document context.\n\n"
                            "Rules:\n"
                            "- Answer thoroughly and specifically using information from the context\n"
                            "- Quote or reference specific parts of the document when relevant\n"
                            "- Use markdown formatting (headers, bullet points, bold) to structure your answer\n"
                            "- If the context does not contain enough information, say exactly what is missing\n"
                            "- Never make up facts not present in the context"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Document context:\n\n{context}\n\n---\n\nQuestion: {query}\n\nProvide a detailed answer based on the document context above:"
                    }
                ],
            )
            return response.choices[0].message.content
 
        except Exception as e:
            err = str(e)
            if "invalid_api_key" in err or "401" in err:
                return "❌ Invalid Groq API key. Check your GROQ_API_KEY in the .env file."
            if "rate_limit" in err:
                return "⚠️ Rate limit hit. Wait a few seconds and try again."
            return f"❌ Error: {err}"
 
    # ── MANAGEMENT ─────────────────────────────────────────────────
 
    def delete_document(self, doc_id: str):
        self.chunks = [c for c in self.chunks if c["doc_id"] != doc_id]
        self._save()
 
    def collection_count(self) -> int:
        return len(self.chunks)
 
    def _save(self):
        try:
            with open(STORE_PATH, "wb") as f:
                pickle.dump(self.chunks, f)
        except Exception:
            pass
 
    def _load(self):
        try:
            if STORE_PATH.exists():
                with open(STORE_PATH, "rb") as f:
                    self.chunks = pickle.load(f)
                print(f"Loaded {len(self.chunks)} chunks from disk.")
        except Exception:
            self.chunks = []