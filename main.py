"""
main.py  —  RAG Chatbot FastAPI Server

Endpoints:
  GET  /                    → serve index.html
  POST /upload              → ingest a document (PDF / TXT / DOCX)
  GET  /doc-status          → name of currently loaded document
  POST /rag/ask             → ask a question against the loaded doc
"""

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Local imports ──────────────────────────────────────────────
from backend import chatbot as _bot
from backend.config import UPLOAD_DIR
from backend.memory import clear_memory

# ── App ────────────────────────────────────────────────────────
app = FastAPI(title="RAG Chatbot", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve CSS / JS from the "templates" folder
app.mount(
    "/templates",
    StaticFiles(directory="templates"),
    name="templates",
)

BASE_DIR = Path(__file__).resolve().parent


# ── Models ─────────────────────────────────────────────────────
class Query(BaseModel):
    question: str
    model: str = "qwen2.5:3b"
    session_id: str = "default"  # identifies the conversation memory buffer


# ── Routes ─────────────────────────────────────────────────────

@app.get("/")
async def read_index():
    """Serve the SPA frontend."""
    return FileResponse(BASE_DIR / "index.html")


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".doc",".xlsx",".xls",".csv",".json",".xml"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accept a document upload, save it, and build the FAISS index.
    Supports: PDF, TXT, MD, DOCX
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    dest = UPLOAD_DIR / file.filename
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    result = _bot.load_document(dest)
    return result


@app.get("/doc-status")
def doc_status():
    """Return the name of the currently loaded document (if any)."""
    name = _bot.get_doc_name()
    return {"loaded": bool(name), "filename": name}


@app.post("/rag/ask")
def rag_ask(q: Query):
    """Answer a question against the loaded document, with rolling memory."""
    return _bot.ask_rag(q.question, model=q.model, session_id=q.session_id)


@app.post("/rag/clear-history")
def rag_clear_history(q: Query):
    """
    Clear conversation history for the given session_id.
    Call this when the user uploads a new document or wants a fresh chat.
    """
    clear_memory(q.session_id)
    return {"status": "ok", "message": f"Memory cleared for session '{q.session_id}'"}
