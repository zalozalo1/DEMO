from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from fastembed import TextEmbedding

load_dotenv()


@dataclass
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "wikivoyage_cities")
    qdrant_distance: str = os.getenv("QDRANT_DISTANCE", "COSINE").upper()
    wiki_lang: str = os.getenv("WIKIVOYAGE_LANG", "en")
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    user_agent: str = os.getenv(
        "USER_AGENT", "Wikivoyage-Qdrant/0.2 (contact: you@example.com)"
    )
    http_timeout: int = int(os.getenv("HTTP_TIMEOUT", "30"))
    http_max_retries: int = int(os.getenv("HTTP_MAX_RETRIES", "3"))
    http_backoff: float = float(os.getenv("HTTP_BACKOFF", "0.3"))

    def distance_enum(self) -> Distance:
        return {
            "COSINE": Distance.COSINE,
            "DOT": Distance.DOT,
            "EUCLID": Distance.EUCLID,
            "EUCLIDEAN": Distance.EUCLID,
            "L2": Distance.EUCLID,
        }.get(self.qdrant_distance, Distance.COSINE)


CFG = Settings()


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": CFG.user_agent})
    retry = Retry(
        total=CFG.http_max_retries,
        backoff_factor=CFG.http_backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def qdrant() -> QdrantClient:
    return QdrantClient(url=CFG.qdrant_url, api_key=CFG.qdrant_api_key)


_embedder: Optional[TextEmbedding] = None


def embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=CFG.embed_model)
    return _embedder


def _embed_dim() -> int:
    return len(next(embedder().embed(["dim"])))


def ensure_collection(dim: Optional[int] = None) -> None:
    client = qdrant()
    try:
        client.get_collection(CFG.qdrant_collection)
        return
    except UnexpectedResponse:
        d = dim or _embed_dim()
        client.create_collection(
            collection_name=CFG.qdrant_collection,
            vectors_config=VectorParams(size=d, distance=CFG.distance_enum()),
        )


def ensure_payload_indexes() -> None:
    """
    Create keyword indexes for exact-match filters we use (title/name).
    Safe to call repeatedly; errors are ignored when index already exists.
    """
    c = qdrant()
    for field in ("title", "name"):
        try:
            c.create_payload_index(
                collection_name=CFG.qdrant_collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


def stable_id(text: str) -> int:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(h[:15], 16)


def wiki_api() -> str:
    return f"https://{CFG.wiki_lang}.wikivoyage.org/w/api.php"


def get_first_image_url(title: str) -> Tuple[Optional[str], Optional[str]]:
    """Fallback: resolve first File:* used on the page and return (full, thumb)."""
    s = make_session()
    api = wiki_api()
    r = s.get(
        api,
        params={
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "images",
            "titles": title,
            "imlimit": "1",
        },
        timeout=CFG.http_timeout,
    )
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", [])
    files = (pages[0].get("images") if pages else None) or []
    if not files:
        return None, None
    file_title = files[0].get("title") or ""
    r2 = s.get(
        api,
        params={
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "titles": file_title,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": "640",
        },
        timeout=CFG.http_timeout,
    )
    r2.raise_for_status()
    pages2 = r2.json().get("query", {}).get("pages", [])
    if not pages2:
        return None, None
    ii = (pages2[0].get("imageinfo") or [{}])[0]
    return ii.get("url"), ii.get("thumburl")


def fetch_city_page(title: str) -> Optional[Dict[str, Any]]:
    """Fetch one city page (summary+image). If exact title missing, fall back to search top hit."""
    s = make_session()
    api = wiki_api()

    r = s.get(
        api,
        params={
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts|pageimages",
            "exintro": "1",
            "explaintext": "1",
            "piprop": "thumbnail|original",
            "pithumbsize": "640",
            "redirects": "1",
            "titles": title,
        },
        timeout=CFG.http_timeout,
    )
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", [])
    if pages and not pages[0].get("missing"):
        p = pages[0]
        t = p.get("title") or title
        extract = (p.get("extract") or "").strip()
        thumb = (p.get("thumbnail") or {})
        original = (p.get("original") or {})
        thumb_url = thumb.get("source")
        image_url = original.get("source") or thumb_url
        if not image_url:
            full_img, thumb_img = get_first_image_url(t)
            image_url = full_img or thumb_img
            thumb_url = thumb_img or full_img
        return {
            "title": t,
            "summary": extract,
            "image_url": image_url,
            "thumb_url": thumb_url,
            "page_url": f"https://{CFG.wiki_lang}.wikivoyage.org/wiki/{t.replace(' ', '_')}",
            "lang": CFG.wiki_lang,
            "source": "wikivoyage",
        }

    r2 = s.get(
        api,
        params={
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "list": "search",
            "srsearch": title,
            "srlimit": "1",
        },
        timeout=CFG.http_timeout,
    )
    r2.raise_for_status()
    hits = r2.json().get("query", {}).get("search", [])
    if not hits:
        return None
    best = hits[0].get("title") or title
    return fetch_city_page(best)


def find_city_exact(title: str) -> Optional[Dict[str, Any]]:
    """
    Try exact match on 'title' and then 'name' (both keyword-indexed).
    Return payload or None.
    """
    client = qdrant()
    try:
        ensure_payload_indexes()

        qfilter = Filter(must=[FieldCondition(key="title", match=MatchValue(value=title))])
        pts, _ = client.scroll(
            CFG.qdrant_collection, with_payload=True, limit=1, scroll_filter=qfilter
        )
        if pts:
            return pts[0].payload

        qfilter2 = Filter(must=[FieldCondition(key="name", match=MatchValue(value=title))])
        pts2, _ = client.scroll(
            CFG.qdrant_collection, with_payload=True, limit=1, scroll_filter=qfilter2
        )
        if pts2:
            return pts2[0].payload

    except UnexpectedResponse:

        try:
            pts, _ = client.scroll(
                CFG.qdrant_collection, with_payload=True, limit=512
            )
            for p in pts:
                pl = p.payload or {}
                if pl.get("title", "").lower() == title.lower() or pl.get("name", "").lower() == title.lower():
                    return pl
        except Exception:
            return None

    return None


def vector_search_city(query: str, k: int = 1) -> Optional[Dict[str, Any]]:
    vec = next(embedder().embed([query]))
    ensure_collection(dim=len(vec))
    client = qdrant()
    try:
        results = client.search(
            collection_name=CFG.qdrant_collection,
            query_vector=list(vec),
            limit=k,
            with_payload=True,
        )
    except UnexpectedResponse:
        return None
    if not results:
        return None
    return results[0].payload or None


def upsert_city_payload(payload: Dict[str, Any]) -> None:
    text = f"{payload.get('title','')} — {payload.get('summary','')}".strip(" —")
    vec = next(embedder().embed([text]))
    ensure_collection(dim=len(vec))
    ensure_payload_indexes()
    client = qdrant()
    client.upsert(
        collection_name=CFG.qdrant_collection,
        points=[
            PointStruct(
                id=stable_id(payload["title"]),
                vector=list(vec),
                payload=payload,
            )
        ],
    )


# ---------------- Public entry ----------------
def resolve_city(
    city: str, country: Optional[str] = None, lang: Optional[str] = None
) -> Dict[str, Any]:
    """
    Returns: { 'payload': {...}, 'was_existing': bool }
    Payload fields: title, summary, image_url, thumb_url, page_url, lang, source
    """
    try:
        ensure_collection()
    except Exception:
        pass

    title_candidates: List[str] = [city]
    if country:
        title_candidates += [f"{city}, {country}", f"{city} ({country})"]

    for t in title_candidates:
        got = find_city_exact(t)
        if got:
            return {"payload": got, "was_existing": True}

    approx = vector_search_city(city, k=1)
    if approx and approx.get("title", "").lower() == city.lower():
        return {"payload": approx, "was_existing": True}

    prev_lang = CFG.wiki_lang
    if lang and lang != prev_lang:
        CFG.wiki_lang = lang
    try:
        fresh = fetch_city_page(city)
    finally:
        CFG.wiki_lang = prev_lang  

    if not fresh:
        raise RuntimeError(f"Could not find a Wikivoyage page for '{city}'")

    upsert_city_payload(fresh)
    return {"payload": fresh, "was_existing": False}
