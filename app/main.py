import os
import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Revibed Text Embeddings Service")

embedder = None
loader_thread = None

process_pid = os.getpid()
app_started_at = time.time()

state = {
    "started": False,
    "loading": False,
    "model_loaded": False,
    "model_error": None,
    "stage": "idle",
    "load_started_at": None,
}


class EmbedTextRequest(BaseModel):
    text: str


class EmbedTextResponse(BaseModel):
    success: bool
    embedding: list[float]


def load_model():
    global embedder, state

    state["loading"] = True
    state["model_error"] = None
    state["stage"] = "import_embedder"
    state["load_started_at"] = time.time()

    try:
        from app.services.embedder import EmbedderService

        state["stage"] = "init_service"
        embedder = EmbedderService()

        state["stage"] = "ready"
        state["model_loaded"] = True
    except Exception as e:
        state["stage"] = "error"
        state["model_loaded"] = False
        state["model_error"] = str(e)
    finally:
        state["loading"] = False


@app.on_event("startup")
async def startup_event():
    global loader_thread
    state["started"] = True
    loader_thread = threading.Thread(target=load_model, daemon=True)
    loader_thread.start()


@app.get("/health")
def health():
    uptime = round(time.time() - app_started_at, 2)
    load_time = None
    if state["load_started_at"]:
        load_time = round(time.time() - state["load_started_at"], 2)

    if state["model_loaded"]:
        return {
            "success": True,
            "status": "ok",
            "pid": process_pid,
            "uptime": uptime,
            "stage": state["stage"],
            "load_time": load_time,
        }

    if state["loading"]:
        return {
            "success": False,
            "status": "loading",
            "pid": process_pid,
            "uptime": uptime,
            "stage": state["stage"],
            "load_time": load_time,
        }

    return {
        "success": False,
        "status": "error" if state["model_error"] else "starting",
        "error": state["model_error"],
        "pid": process_pid,
        "uptime": uptime,
        "stage": state["stage"],
        "load_time": load_time,
    }


@app.post("/embed-text", response_model=EmbedTextResponse)
def embed_text(payload: EmbedTextRequest):
    text = (payload.text or "").strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "Text is required",
            },
        )

    if not state["model_loaded"] or embedder is None:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "status": "loading" if state["loading"] else "not_ready",
                "error": state["model_error"],
                "stage": state["stage"],
            },
        )

    try:
        embedding = embedder.embed_text(text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
            },
        )

    return {
        "success": True,
        "embedding": embedding,
    }