"""ALPHA API: exactly POST /index and POST /query.

    .venv/bin/uvicorn alpha.api:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import os
import re
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from alpha import pipeline
from alpha.index import index_folder
from alpha.observe import finish_observation, query_observation

COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
_loaded: dict[str, tuple] = {}
_lock = threading.Lock()
_jobs: dict[str, asyncio.Future] = {}   # collection -> running index job (a retry joins it)
KEEPALIVE_S = 15
WORKERS = os.cpu_count() or 8


class QueryRequest(BaseModel):
    collection_name: str
    query: str


def _check(name: str) -> str:
    if not COLLECTION_RE.fullmatch(name):
        raise HTTPException(400, "Invalid collection_name")
    return name


def _collection(name: str) -> tuple:
    with _lock:
        if name not in _loaded:
            if not (pipeline.STORE / f"{name}_graph.pkl").exists():
                raise HTTPException(404, "unknown collection")
            _loaded[name] = pipeline.load(name)
        return _loaded[name]


def _run_index(tmp: str, name: str) -> dict:
    try:
        stats = index_folder(str(Path(tmp) / "docs"), name, workers=WORKERS)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with _lock:
        _loaded.pop(name, None)
    return {"status": "success", "collection_name": name, "documents_indexed": stats["documents"], **stats}


@app.post("/index")
async def index_documents(collection_name: str = Form(...), file: UploadFile = File(...)):
    """Synchronous for the caller: the body completes only when the collection is queryable.

    Indexing runs in a worker thread so /query stays responsive, and the response streams a space
    every few seconds until the JSON result, so proxies (ngrok) never see an idle connection.
    Leading whitespace is valid JSON. A retry for a collection already being indexed joins that job.
    """
    name = _check(collection_name)
    job = _jobs.get(name)
    if job is None or job.done():
        tmp = tempfile.mkdtemp(prefix="alpha-index-")
        archive = Path(tmp) / "upload.zip"
        archive.write_bytes(await file.read())
        docs = Path(tmp) / "docs"
        docs.mkdir()
        try:
            with zipfile.ZipFile(archive) as zf:
                for m in zf.infolist():
                    if not (docs / m.filename).resolve().is_relative_to(docs.resolve()):
                        raise HTTPException(400, "ZIP contains an unsafe path")
                zf.extractall(docs)
        except zipfile.BadZipFile as e:
            shutil.rmtree(tmp, ignore_errors=True)
            raise HTTPException(400, "Invalid ZIP archive") from e
        except HTTPException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        archive.unlink()
        if not any(p.suffix.lower() == ".pdf" for p in docs.rglob("*")):
            shutil.rmtree(tmp, ignore_errors=True)
            raise HTTPException(400, "ZIP contains no PDF documents")
        job = asyncio.ensure_future(run_in_threadpool(_run_index, tmp, name))
        _jobs[name] = job

    async def body():
        while not job.done():
            try:
                await asyncio.wait_for(asyncio.shield(job), timeout=KEEPALIVE_S)
            except asyncio.TimeoutError:
                yield b" "
            except Exception:
                break
        try:
            yield json.dumps(job.result()).encode()
        except Exception as e:  # headers are already sent; report the failure in the body
            yield json.dumps({"status": "error", "collection_name": name, "detail": str(e)}).encode()

    return StreamingResponse(body(), media_type="application/json")


@app.post("/query")
def query_collection(body: QueryRequest):
    graph, tree = _collection(_check(body.collection_name))
    with query_observation(body.query) as gen:
        result = pipeline.query(body.query, graph, tree)
        finish_observation(gen, result)
    # Contract: chunks[].content + metadata.file_name (corpus file name) + metadata.page_numbers (1-based)
    return {"answer": result["answer"] or "The documents do not contain this information.",
            "chunks": [{"content": c["text"],
                        "metadata": {"file_name": c["doc"], "page_numbers": [int(p) for p in _pages(c["pages"])],
                                     "unit_id": c["id"]}} for c in result["chunks"]],
            "trace": result["trace"],
            "usage": {"tokens": result["tokens"], "latency_s": round(result["latency_s"], 2)}}


def _pages(pages) -> list:
    if isinstance(pages, (list, tuple)):
        return list(pages)
    return [int(x) for x in str(pages).strip("[]").split(",") if x.strip()]
