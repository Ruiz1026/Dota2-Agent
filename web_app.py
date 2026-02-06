import asyncio
import json
import re
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Set
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dota2_agent import Dota2ReActAgent


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
LOG_DIR = BASE_DIR / "logs"
FIGURE_DIR = BASE_DIR / "figure"

app = FastAPI(title="Dota2 ReAct Chat")
agent = Dota2ReActAgent(enable_logging=True)
agent_lock = asyncio.Lock()
background_tasks: Set[asyncio.Task] = set()

app.mount(
    "/ward_analysis",
    StaticFiles(directory=BASE_DIR / "ward_analysis", html=True),
    name="ward_analysis",
)
app.mount(
    "/figure",
    StaticFiles(directory=FIGURE_DIR, html=False),
    name="figure",
)


class ChatRequest(BaseModel):
    message: str
    new_session: bool = False
    session_id: Optional[str] = None


@app.on_event("startup")
async def startup_event() -> None:
    await agent.connect_mcp()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await agent.disconnect_mcp()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _load_history() -> List[Dict[str, str]]:
    if not LOG_DIR.exists():
        return []

    items: List[Dict[str, str]] = []
    session_files = sorted(LOG_DIR.glob("session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in session_files:
        try:
            session_data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        session_id = session_data.get("session_id")
        conversations = session_data.get("conversations", [])
        for conv in conversations:
            conv_id = conv.get("id")
            user_input = conv.get("user_input") or ""
            timestamp = conv.get("timestamp")
            status = conv.get("status") or "unknown"
            items.append({
                "session_id": str(session_id),
                "conversation_id": str(conv_id),
                "user_input": str(user_input),
                "timestamp": str(timestamp or ""),
                "status": str(status),
            })

    items.sort(
        key=lambda x: _parse_time(x.get("timestamp")) or datetime.min,
        reverse=True,
    )
    return items


def _extract_ward_html_path(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    match = re.search(r'(ward_analysis[\\/](ward_(?:timeline|multi)_[^\\/]+\.html))', text)
    if match:
        return "/" + match.group(1).replace("\\", "/")
    return None


def _load_session(session_id: str) -> Dict:
    session_path = LOG_DIR / f"session_{session_id}.json"
    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        session_data = json.loads(session_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read session: {exc}") from exc

    return session_data


@app.get("/api/history")
async def history() -> Dict[str, List[Dict[str, str]]]:
    return {"items": _load_history()}


@app.get("/api/conversations/{session_id}/{conversation_id}")
async def conversation(session_id: str, conversation_id: int) -> Dict[str, str]:
    session_data = _load_session(session_id)
    conversations = session_data.get("conversations", [])
    data = next((c for c in conversations if c.get("id") == conversation_id), None)
    if not data:
        raise HTTPException(status_code=404, detail="Conversation not found")

    final_answer = data.get("final_answer") or ""
    return {
        "session_id": session_id,
        "conversation_id": str(conversation_id),
        "timestamp": str(data.get("timestamp") or ""),
        "user_input": str(data.get("user_input") or ""),
        "final_answer": str(final_answer),
        "status": str(data.get("status") or "unknown"),
        "ward_html": _extract_ward_html_path(final_answer) or "",
    }


@app.get("/api/sessions/{session_id}")
async def session(session_id: str) -> Dict[str, object]:
    session_data = _load_session(session_id)
    conversations = session_data.get("conversations", [])
    view_items = []
    for conv in conversations:
        final_answer = conv.get("final_answer") or ""
        view_items.append({
            "conversation_id": str(conv.get("id")),
            "timestamp": str(conv.get("timestamp") or ""),
            "user_input": str(conv.get("user_input") or ""),
            "final_answer": str(final_answer),
            "status": str(conv.get("status") or "unknown"),
            "ward_html": _extract_ward_html_path(final_answer) or "",
        })

    return {
        "session_id": str(session_data.get("session_id") or session_id),
        "start_time": str(session_data.get("start_time") or ""),
        "end_time": str(session_data.get("end_time") or ""),
        "total_conversations": int(session_data.get("total_conversations") or len(view_items)),
        "conversations": view_items,
    }


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    async def _run_agent_stream(
        queue: asyncio.Queue,
        connected: asyncio.Event,
    ) -> None:
        async with agent_lock:
            if request.new_session:
                await agent.start_new_session()
            elif request.session_id:
                session_data = _load_session(request.session_id)
                agent.load_recent_context_from_session(session_data.get("conversations", []))
            async for event in agent.run_stream(message):
                if not connected.is_set():
                    continue
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        pass

    async def _response_stream() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        connected = asyncio.Event()
        connected.set()
        task = asyncio.create_task(_run_agent_stream(queue, connected))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        try:
            while True:
                if task.done() and queue.empty():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                yield json.dumps(event, ensure_ascii=False) + "\n"
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            connected.clear()
            return
        finally:
            connected.clear()

    return StreamingResponse(_response_stream(), media_type="application/x-ndjson; charset=utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False)
