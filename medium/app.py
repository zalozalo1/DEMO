# app.py
# Travel Chatbot with LangGraph + Gemini 2.0 Flash
# -------------------------------------------------
# pip install -U langgraph langchain langchain-google-genai google-genai langchain-community duckduckgo-search python-dotenv

from __future__ import annotations

from typing import TypedDict, List, Optional
import os
import json
from dataclasses import dataclass

from dotenv import load_dotenv

# LangChain + Gemini (AI Studio key)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

# Tools
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# Messages / Graph
from langchain_core.messages import HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent


# =========================
# Env & API key
# =========================
load_dotenv()
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_KEY or not GEMINI_KEY.strip():
    raise RuntimeError("GOOGLE_API_KEY not found or empty. Put it in .env (GOOGLE_API_KEY=...)")

# Single shared LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    google_api_key=GEMINI_KEY,
)


# =========================
# Simple in-memory store
# =========================
class PrefStore:
    def __init__(self):
        self._by_user: dict[str, dict] = {}

    def get(self, user_id: str) -> dict:
        return self._by_user.get(user_id, {}).copy()

    def save(self, user_id: str, prefs: dict) -> None:
        self._by_user[user_id] = prefs

    def merge(self, user_id: str, update: dict) -> dict:
        existing = self.get(user_id)
        for k, v in (update or {}).items():
            if v is None:
                continue
            existing[k] = v
        self.save(user_id, existing)
        return existing


STORE = PrefStore()
# A tiny runtime context for tools (single-user demo).
CURRENT_USER_ID = "default"


# =========================
# Tools (called by the LLM)
# =========================
@tool
def save_user_prefs(
    interests: Optional[List[str]] = None,
    budget_level: Optional[str] = None,          # "low" | "mid" | "high"
    trip_length_days: Optional[int] = None,
    preferred_climate: Optional[str] = None,     # "warm" | "mild" | "cool" | "dry" | etc.
    region: Optional[str] = None,                # "Europe" | "Asia" | "Americas" | ...
    travel_pace: Optional[str] = None,           # "relaxed" | "balanced" | "packed"
    with_kids: Optional[bool] = None,
    max_flight_hours: Optional[int] = None,
) -> str:
    """
    Persist known preferences. The agent should call this whenever the user states something explicit.
    """
    updated = STORE.merge(
        CURRENT_USER_ID,
        {
            "interests": interests,
            "budget_level": budget_level,
            "trip_length_days": trip_length_days,
            "preferred_climate": preferred_climate,
            "region": region,
            "travel_pace": travel_pace,
            "with_kids": with_kids,
            "max_flight_hours": max_flight_hours,
        },
    )
    return json.dumps({"ok": True, "saved": updated}, ensure_ascii=False)


@tool
def get_user_prefs() -> str:
    """
    Return the current user's saved preferences as JSON.
    """
    prefs = STORE.get(CURRENT_USER_ID)
    return json.dumps(prefs, ensure_ascii=False)


# Optional search tool (for the recommender to sanity-check facts if needed)
search_tool = DuckDuckGoSearchRun()


# =========================
# Agents
# =========================
PREF_PROMPT = (
    "You are a travel preference elicitation assistant.\n"
    "Objectives:\n"
    "1) Ask ONE short, specific question at a time to discover travel preferences.\n"
    "2) When the user states a preference (explicit or strongly implied), CALL the tool `save_user_prefs` "
    "   with structured values (lists for interests, ints for days, bool for with_kids, etc.).\n"
    "3) Cover these fields (order flexibly): interests, budget_level, trip_length_days, preferred_climate, "
    "   region, travel_pace, with_kids, max_flight_hours.\n"
    "4) Do NOT recommend cities yet. Stay in Q&A mode until instructed by the graph.\n"
    "Keep questions crisp. Avoid compound questions."
)

pref_agent = create_react_agent(
    model=llm,
    tools=[save_user_prefs],
    prompt=PREF_PROMPT,
)

RECS_PROMPT = (
    "You are a travel recommender.\n"
    "First, call `get_user_prefs` to load the user's stored preferences. Then produce 5–7 city recommendations "
    "that match those preferences.\n"
    "For EACH city, include:\n"
    "- **Name, Country**\n"
    "- **Why it matches**: 1–2 sentences tailored to prefs\n"
    "- **Best months** (rough guidance)\n"
    "- **Highlights**: 2–3 bullets\n"
    "Output well-formatted Markdown with numbered items. Be specific, not generic. "
    "If necessary, you MAY use the search tool to sanity-check factual bits."
)

recs_agent = create_react_agent(
    model=llm,
    tools=[get_user_prefs, search_tool],
    prompt=RECS_PROMPT,
)


# =========================
# Graph state & nodes
# =========================
class ChatState(TypedDict, total=False):
    messages: List[AnyMessage]
    user_id: str
    rounds: int  # how many preference-question rounds so far


CORE_FIELDS = ["interests", "budget_level", "trip_length_days", "preferred_climate", "region"]


def _route(state: ChatState) -> str:
    prefs = STORE.get(state["user_id"])
    have = sum(1 for k in CORE_FIELDS if prefs.get(k))
    if have >= 3 or state.get("rounds", 0) >= 4:
        return "recs"
    return "pref"


def run_pref_agent(state: ChatState) -> ChatState:
    global CURRENT_USER_ID
    CURRENT_USER_ID = state["user_id"]
    result = pref_agent.invoke({"messages": state["messages"]})
    return {
        "messages": result["messages"],
        "user_id": state["user_id"],
        "rounds": state.get("rounds", 0) + 1,
    }


def run_recs_agent(state: ChatState) -> ChatState:
    global CURRENT_USER_ID
    CURRENT_USER_ID = state["user_id"]
    result = recs_agent.invoke({"messages": state["messages"]})
    return {
        "messages": result["messages"],
        "user_id": state["user_id"],
        "rounds": state.get("rounds", 0),
    }


# Build the graph
graph = StateGraph(ChatState)
graph.add_node("pref", run_pref_agent)
graph.add_node("recs", run_recs_agent)

graph.add_edge(START, "pref")
graph.add_conditional_edges("pref", _route, {"pref": "pref", "recs": "recs"})
graph.add_edge("recs", END)

app = graph.compile()


# =========================
# CLI demo
# =========================
if __name__ == "__main__":
    print("Travel Chatbot (Gemini 2.0 Flash) — type 'quit' to exit.\n")

    user_id = "demo-user-001"
    state: ChatState = {
        "messages": [HumanMessage(content="Hi! Help me plan a trip that fits my tastes.")],
        "user_id": user_id,
        "rounds": 0,
    }

    while True:
        try:
            out = app.invoke(state)
        except ChatGoogleGenerativeAIError as e:
            raise SystemExit(
                f"\nGemini call failed: {e}\n"
                "Check that your GOOGLE_API_KEY is a valid AI Studio key and your network allows access to generativelanguage.googleapis.com.\n"
            )

        last = out["messages"][-1]
        print(f"\nAI: {getattr(last, 'content', str(last))}\n")

        # If graph routed to recommendations, we're done.
        if _route(out) == "recs":
            break

        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            break

        out["messages"].append(HumanMessage(content=user))
        state = out
