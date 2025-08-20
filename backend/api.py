from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

from core import build_graph, initial_message

app = FastAPI(title="TripPlanner API (LangGraph)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compile LangGraph once
app_graph = build_graph()

class StartOut(BaseModel):
    session_id: str
    message: str

class ChatIn(BaseModel):
    session_id: str
    message: str

class ChatOut(BaseModel):
    kind: str
    message: str | None = None
    recommendations: list[dict] | None = None
    prefs: dict | None = None
    error: str | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/start", response_model=StartOut)
def start():
    # Use a new thread_id; LangGraph checkpoint will preserve state by this id
    thread_id = uuid.uuid4().hex
    return {"session_id": thread_id, "message": initial_message}

@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn):
    # Send only the per-turn input; checkpoint fills the rest of the state for this thread
    state_in = {"last_user_text": body.message}

    state_out = app_graph.invoke(state_in, config={"configurable": {"thread_id": body.session_id}})

    return {
        "kind": state_out.get("out_kind", "question"),
        "message": state_out.get("out_message"),
        "recommendations": state_out.get("out_recs"),
        "prefs": state_out.get("prefs"),
        "error": state_out.get("error"),
    }

