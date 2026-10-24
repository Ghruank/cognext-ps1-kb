"""Minimal PS1 API: exactly POST /index and POST /query."""

import os
import re
import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from nexus.index import build_folder
from nexus.pipeline import query

DATA_DIR = Path(os.getenv("NEXUS_DATA_DIR", Path(__file__).resolve().parent / "collections"))
COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")

# No default docs/OpenAPI endpoints, matching the PS requirement for exactly two routes.
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class QueryRequest(BaseModel):
    collection_name: str
    query: str


def _collection_path(name: str) -> Path:
    if not COLLECTION_RE.fullmatch(name) or name in {".", ".."}:
        raise HTTPException(400, "Invalid collection_name")
    return DATA_DIR / name


@app.post("/index")
async def index_documents(collection_name: str = Form(...), file: UploadFile = File(...)):
    """Index one uploaded ZIP in a separate named collection, using no LLM calls."""
    root = _collection_path(collection_name)
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "Upload a .zip file")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nexus-index-") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "upload.zip"
        archive.write_bytes(await file.read())
        extracted = tmp_path / "docs"
        extracted.mkdir()
        try:
            with zipfile.ZipFile(archive) as zf:
                for member in zf.infolist():
                    target = (extracted / member.filename).resolve()
                    if not target.is_relative_to(extracted.resolve()):
                        raise HTTPException(400, "ZIP contains an unsafe path")
                zf.extractall(extracted)
        except zipfile.BadZipFile as e:
            raise HTTPException(400, "Invalid ZIP archive") from e
        pdfs = list(extracted.rglob("*.pdf")) + list(extracted.rglob("*.PDF"))
        if not pdfs:
            raise HTTPException(400, "ZIP contains no PDF documents")
        # Build at a temporary filename and replace only after successful ingestion.
        staged = root / "index.sqlite.staged"
        try:
            workers = max(1, int(os.getenv("NEXUS_INGEST_WORKERS", "8")))
            stats = build_folder(extracted, staged, workers=workers,
                                 use_dense=os.getenv("NEXUS_DISABLE_DENSE_INDEX") != "1")
            staged.replace(root / "index.sqlite")
        except Exception as e:
            staged.unlink(missing_ok=True)
            raise HTTPException(500, f"Index build failed: {e}") from e
    return {"status": "indexed", "collection_name": collection_name, "documents": stats["documents"],
            "pages": stats["pages"], "units": stats["units"],
            "version_families": stats.get("version_families", 0)}


@app.post("/query")
def query_collection(body: QueryRequest):
    """Run NEXUS against one named collection and return page-cited evidence + trace."""
    root = _collection_path(body.collection_name)
    index = root / "index.sqlite"
    if not index.exists():
        raise HTTPException(404, "Collection is not indexed")
    llm_enabled = os.getenv("NEXUS_DISABLE_LLM") != "1"
    result = query(body.query, index, use_llm=llm_enabled, verify_relevance=True,
                   generate_answer=True, observe=llm_enabled)
    # The required per-step trace entries each have exactly the PS's four fields.
    return {"answer": result["answer"], "chunks": [
        {"text": c["text"], "doc": c["doc"], "page": c["pages"][0], "citation": f"[{c['doc']} p{c['pages'][0]}]"}
        for c in result["chunks"]], "trace": result["trace"]}
