from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
import google.generativeai as genai

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

@dataclass
class Settings:
    model_name: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

    def ensure(self) -> None:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (add it to environment or a .env file).")

SETTINGS = Settings()
SETTINGS.ensure()

_gen_config_json = {"response_mime_type": "application/json"}

genai.configure(api_key=SETTINGS.api_key)
MODEL = genai.GenerativeModel(
    model_name=SETTINGS.model_name,
    system_instruction=(
        "You are TripPlanner — a warm, practical travel copilot.\n"
        "Goal: collect just enough preferences to suggest 5–7 cities.\n"
        "Be concise; no emojis or fluff.\n"
    ),
)

class Prefs(TypedDict, total=False):
    interests: List[str]
    climate: str
    budget: str
    season: str
    duration: str
    companions: str
    pace: str
    region: str
    health: str

class ChatState(TypedDict, total=False):
    prefs: Prefs
    rounds: int
    enough: bool
    force_recommend: bool
    completed: bool
    asked_log: List[dict]
    skips: List[str]
    last_progress_round: int
    last_question_key: str | None

    last_user_text: str | None

    out_kind: str | None
    out_message: str | None
    out_recs: list[dict] | None

FIELD_ORDER = [
    "interests",
    "region",
    "season",
    "climate",
    "budget",
    "duration",
    "companions",
    "pace",
    "health",
]

QUESTION_BANK = {
    "interests": "What are your main interests for this trip (e.g., museums, beaches, nightlife)?",
    "region": "Do you prefer a region (e.g., Europe, Asia) or are you flexible?",
    "season": "Roughly when are you traveling (month or season)?",
    "climate": "What climate do you want (warm, mild, cold, dry)?",
    "budget": "What budget level fits best (low, mid, high)?",
    "duration": "How long is the trip (e.g., 1 week, 10 days)?",
    "companions": "Who are you traveling with (solo, family, friends)?",
    "pace": "Do you prefer a relaxed or packed schedule?",
    "health": "Any health or mobility considerations I should plan around?",
}

COOLDOWN_ROUNDS = 2
NO_PROGRESS_NUDGE = 3


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```") and t.endswith("```"):
        body = "\n".join(t.splitlines()[1:-1])
        return body.strip()
    return t


def _extract_json(text: str):
    t = _strip_code_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"(\{[^]*\}|\[[^]*\])", t)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    raise ValueError("Model did not return parseable JSON.")


def llm_json(prompt: str):
    """Call Gemini and return parsed JSON with a robust fallback."""
    try:
        resp = MODEL.generate_content(prompt, generation_config=_gen_config_json)
        return _extract_json(getattr(resp, "text", "") or "")
    except Exception:
        resp = MODEL.generate_content(prompt)
        return _extract_json(getattr(resp, "text", "") or "")


def assess_enough(prefs: Prefs) -> bool:
    interests = prefs.get("interests") or []
    if not isinstance(interests, list) or len(interests) == 0:
        return False
    other_truthy = sum(
        1 for k, v in prefs.items()
        if k != "interests" and isinstance(v, str) and v.strip()
    )
    return other_truthy >= 2


def merge_prefs(existing: Prefs, newbits: dict) -> Prefs:
    out: Prefs = dict(existing) if existing else {}
    changed = False
    for k, v in (newbits or {}).items():
        if k == "interests" and isinstance(v, list):
            existing_set = set(map(str.lower, out.get("interests", []) or []))
            new_set = {s.strip().lower() for s in v if isinstance(s, str) and s.strip()}
            merged = sorted(existing_set.union(new_set))
            if merged != out.get("interests"):
                out["interests"] = merged
                changed = True
        elif isinstance(v, str) and v.strip():
            val = v.strip()
            if out.get(k) != val:
                out[k] = val
                changed = True
    out["__changed__"] = changed 
    return out


def _is_skip_answer(text: str) -> bool:
    t = (text or "").strip().lower()
    SKIPS = {"skip", "idk", "i don't know", "dont know", "not sure", "no idea", "pass"}
    return any(s in t for s in SKIPS)

EXTRACT_PROMPT = (
    "Extract and merge user preferences from USER_TEXT into CURRENT_PREFS.\n"
    "Return ONLY a compact JSON object with ANY of these keys:\n"
    "interests (array of strings), climate, budget, season, duration, companions, pace, region, health.\n"
    "- interests: deduplicate, lowercase nouns (e.g., 'museums', 'beaches').\n"
    "- Do not invent. Preserve existing fields unless contradicted; update if user clarifies.\n"
    "- Output JSON only.\n"
)

RECOMMEND_PROMPT = (
    "Using FINAL_PREFS (JSON), propose 5–7 distinct city recommendations as a JSON array.\n"
    "Each item MUST be: {\"city\": string, \"country\": string, \"pitch\": string}.\n"
    "Rules:\n"
    "- Align closely with interests and constraints (health, climate, budget, pace, companions, season, region).\n"
    "- Keep 'pitch' to one tight sentence focusing on the key fit.\n"
    "- Avoid repeats and clichés; do not invent attributes a city doesn't have.\n"
    "- Prefer globally known, travel-feasible cities.\n"
    "- JSON ONLY.\n"
)


def initial_greeting() -> str:
    return "Hi! Tell me a bit about your trip goal and interests (e.g., beaches, museums, nightlife)."


def plan_next_question(state: ChatState) -> str:
    prefs = state.get("prefs", {})
    asked_log = state.get("asked_log", [])
    skips = set(state.get("skips", []))
    rounds = state.get("rounds", 0)

    if assess_enough(prefs):
        state["last_question_key"] = None
        return "Ready for recommendations?"

    for key in FIELD_ORDER:
        if key in skips:
            continue
        missing = False
        if key == "interests":
            missing = not isinstance(prefs.get("interests"), list) or len(prefs.get("interests", [])) == 0
        else:
            missing = not (isinstance(prefs.get(key), str) and prefs.get(key, "").strip())
        if not missing:
            continue

        recent_same = next((a for a in reversed(asked_log) if a.get("key") == key), None)
        if recent_same and (rounds - int(recent_same.get("round", -999))) < COOLDOWN_ROUNDS:
            continue

        state["last_question_key"] = key
        return QUESTION_BANK[key]

    state["last_question_key"] = None
    return "Share any detail you like (budget, climate, region, season, duration) — or type 'recommend'."


def ingest_node(state: ChatState) -> ChatState:
    state.setdefault("prefs", {})
    state.setdefault("rounds", 0)
    state.setdefault("enough", False)
    state.setdefault("force_recommend", False)
    state.setdefault("completed", False)
    state.setdefault("asked_log", [])
    state.setdefault("skips", [])
    state.setdefault("last_progress_round", 0)
    state.setdefault("last_question_key", None)

    t = (state.get("last_user_text") or "").strip()
    if not t:
        return state

    if t.lower() == "recommend":
        state["force_recommend"] = True

    if state.get("last_question_key") and _is_skip_answer(t):
        key = state["last_question_key"]
        skips = set(state.get("skips", []))
        skips.add(key)
        state["skips"] = sorted(list(skips))

    prev = json.dumps(state.get("prefs", {}), sort_keys=True)
    merged = llm_json(
        "CURRENT_PREFS = "
        + json.dumps(state.get("prefs", {}), ensure_ascii=False)
        + "\nUSER_TEXT = "
        + json.dumps(t, ensure_ascii=False)
        + "\n"
        + EXTRACT_PROMPT
    )
    if isinstance(merged, dict):
        state["prefs"] = merge_prefs(state.get("prefs", {}), merged)

    now = json.dumps(state.get("prefs", {}), sort_keys=True)
    if prev != now:
        state["last_progress_round"] = state.get("rounds", 0)

    state["rounds"] = state.get("rounds", 0) + 1
    return state


def decide_node(state: ChatState) -> ChatState:
    state["enough"] = True if state.get("force_recommend") else assess_enough(state.get("prefs", {}))
    return state


def ask_node(state: ChatState) -> ChatState:
    q = plan_next_question(state)
    if state.get("last_question_key"):
        state["asked_log"].append({"key": state["last_question_key"], "round": state.get("rounds", 0)})

    no_prog = state.get("rounds", 0) - state.get("last_progress_round", 0)
    if no_prog >= NO_PROGRESS_NUDGE and not state.get("enough"):
        q = "We can wrap up with what we have. Type 'recommend' to see city matches, or share one more detail."

    state["out_kind"] = "question"
    state["out_message"] = q
    state["out_recs"] = None
    return state


def recommend_node(state: ChatState) -> ChatState:
    try:
        recs = llm_json(
            "FINAL_PREFS = " + json.dumps(state.get("prefs", {}), ensure_ascii=False) + "\n" + RECOMMEND_PROMPT
        )
        if not isinstance(recs, list):
            raise ValueError("Expected a JSON array of recommendations.")
        state["completed"] = True
        state["out_kind"] = "recommendations"
        state["out_recs"] = recs
        state["out_message"] = "Here are your city matches:"
        return state
    except Exception as e:
        state["out_kind"] = "question"
        state["out_message"] = (
            "Add one more detail (e.g., budget, climate, or region) and say 'recommend' to proceed."
        )
        state["out_recs"] = None
        state["error"] = str(e)
        return state


def build_graph():
    g = StateGraph(ChatState)
    g.add_node("ingest", ingest_node)
    g.add_node("decide", decide_node)
    g.add_node("ask", ask_node)
    g.add_node("recommend", recommend_node)

    def router(s: ChatState) -> str:
        return "recommend" if s.get("enough") else "ask"

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "decide")
    g.add_conditional_edges("decide", router, {"ask": "ask", "recommend": "recommend"})
    g.add_edge("ask", END)
    g.add_edge("recommend", END)

    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)

initial_message = (
    "Hi! Tell me a bit about your trip goal and interests (e.g., beaches, museums, nightlife)."
)
