# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

from core import build_graph, initial_message
from qdrant_service import resolve_city 

app = FastAPI(title="TripPlanner API (LangGraph)", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class SelectCityIn(BaseModel):
    session_id: str
    city: str
    country: str | None = None
    lang: str | None = None

class SelectCityOut(BaseModel):
    title: str
    summary: str | None = None
    image_url: str | None = None
    page_url: str | None = None
    lang: str | None = None
    was_existing: bool

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/start", response_model=StartOut)
def start():
    thread_id = uuid.uuid4().hex
    return {"session_id": thread_id, "message": initial_message}

@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn):
    state_in = {"last_user_text": body.message}
    state_out = app_graph.invoke(state_in, config={"configurable": {"thread_id": body.session_id}})

    return {
        "kind": state_out.get("out_kind", "question"),
        "message": state_out.get("out_message"),
        "recommendations": state_out.get("out_recs"),
        "prefs": state_out.get("prefs"),
        "error": state_out.get("error"),
    }

@app.post("/select-city", response_model=SelectCityOut)
def select_city(body: SelectCityIn):
    r = resolve_city(city=body.city, country=body.country, lang=body.lang)
    p = r["payload"]
    return {
        "title": p.get("title") or body.city,
        "summary": p.get("summary"),
        "image_url": p.get("image_url") or p.get("thumb_url"),
        "page_url": p.get("page_url"),
        "lang": p.get("lang"),
        "was_existing": bool(r.get("was_existing")),
    }
