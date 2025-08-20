# app.py — Wikivoyage → Qdrant (solid, env-driven, pretty console)
# Works on Python 3.11+ (incl. 3.13). No Docker required.
#
# Install:
#   pip install qdrant-client fastembed onnxruntime requests python-dotenv tqdm rich
#
# .env (example; place next to this file):
#   QDRANT_URL=https://<your-cluster>.<region>.<provider>.cloud.qdrant.io   # or http://localhost:6333
#   QDRANT_API_KEY=<your_database_api_key>                                  # omit for local without auth
#   QDRANT_COLLECTION=wikivoyage_cities
#   QDRANT_DISTANCE=COSINE                                                  # COSINE | DOT | EUCLID
#   WIKIVOYAGE_LANG=en
#   WIKIVOYAGE_CATEGORY=Category:City articles
#   EMBED_MODEL=BAAI/bge-small-en-v1.5                                      # fast CPU, 384-d
#   INGEST_LIMIT=200
#   SLEEP_SECONDS=0.15
#   BATCH_SIZE=128
#   HTTP_TIMEOUT=30
#   HTTP_MAX_RETRIES=3
#   HTTP_BACKOFF=0.3
#   PROXY=
#   SEARCH_K=5
#   USER_AGENT=Wikivoyage-Qdrant/0.2 (contact: you@example.com)
#
# Commands:
#   python app.py config
#   python app.py verify
#   python app.py ingest
#   python app.py search "beach city"
#   python app.py stats
#   python app.py peek --n 5
#   python app.py backfill-images --max 5000
#   python app.py export --out wikivoyage_cities.ndjson

from __future__ import annotations
import os
import sys
import time
import json
import hashlib
import argparse
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
)
from fastembed import TextEmbedding

# ---------- Pretty console ----------
console = Console(highlight=False)

# ---------- Load .env sitting next to this file (always) ----------
ENV_PATH = Path(__file__).with_name(".env")
loaded = load_dotenv(dotenv_path=ENV_PATH, override=True)
if not loaded:
    console.print(Panel(f".env not found at: {ENV_PATH.resolve()}\nUsing process env / defaults.",
                        title="Warning", border_style="yellow"))

# ---------- Config ----------
@dataclass
class Settings:
    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "wikivoyage_cities")
    qdrant_distance: str = os.getenv("QDRANT_DISTANCE", "COSINE").upper()  # COSINE|DOT|EUCLID

    # Wikivoyage
    wiki_lang: str = os.getenv("WIKIVOYAGE_LANG", "en")
    wiki_category: str = os.getenv("WIKIVOYAGE_CATEGORY", "Category:City articles")

    # Embeddings
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

    # Ingest & HTTP
    ingest_limit: int = int(os.getenv("INGEST_LIMIT", "200"))
    sleep_seconds: float = float(os.getenv("SLEEP_SECONDS", "0.15"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "128"))
    http_timeout: int = int(os.getenv("HTTP_TIMEOUT", "30"))
    http_max_retries: int = int(os.getenv("HTTP_MAX_RETRIES", "3"))
    http_backoff: float = float(os.getenv("HTTP_BACKOFF", "0.3"))
    proxy: Optional[str] = os.getenv("PROXY") or None

    # Search
    search_k: int = int(os.getenv("SEARCH_K", "5"))

    # Misc
    user_agent: str = os.getenv("USER_AGENT", "Wikivoyage-Qdrant/0.2 (contact: you@example.com)")

    def distance_enum(self) -> Distance:
        mapping = {
            "COSINE": Distance.COSINE,
            "DOT": Distance.DOT,
            "EUCLID": Distance.EUCLID,
            "EUCLIDEAN": Distance.EUCLID,
            "L2": Distance.EUCLID,
        }
        return mapping.get(self.qdrant_distance, Distance.COSINE)

    def masked(self) -> Dict[str, str]:
        """Safe to print config (mask secrets)."""
        return {
            "QDRANT_URL": self.qdrant_url,
            "QDRANT_API_KEY": "***" if self.qdrant_api_key else "",
            "QDRANT_COLLECTION": self.qdrant_collection,
            "QDRANT_DISTANCE": self.qdrant_distance,
            "WIKIVOYAGE_LANG": self.wiki_lang,
            "WIKIVOYAGE_CATEGORY": self.wiki_category,
            "EMBED_MODEL": self.embed_model,
            "INGEST_LIMIT": str(self.ingest_limit),
            "SLEEP_SECONDS": str(self.sleep_seconds),
            "BATCH_SIZE": str(self.batch_size),
            "HTTP_TIMEOUT": str(self.http_timeout),
            "HTTP_MAX_RETRIES": str(self.http_max_retries),
            "HTTP_BACKOFF": str(self.http_backoff),
            "PROXY": "***" if self.proxy else "",
            "SEARCH_K": str(self.search_k),
            "USER_AGENT": self.user_agent,
        }

CFG = Settings()

# ---------- HTTP session with retries ----------
def make_session(cfg: Settings) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": cfg.user_agent})
    if cfg.proxy:
        s.proxies.update({"http": cfg.proxy, "https": cfg.proxy})
    retry = Retry(
        total=cfg.http_max_retries,
        backoff_factor=cfg.http_backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

# ---------- Utilities ----------
def stable_id(text: str) -> int:
    """Deterministic id from title (64-bit safe)."""
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(h[:15], 16)

def page_url(cfg: Settings, title: str) -> str:
    return f"https://{cfg.wiki_lang}.wikivoyage.org/wiki/" + title.replace(" ", "_")

def get_first_image_url(
    session: requests.Session, cfg: Settings, title: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Fallback: list the first 'File:*' used on the page and resolve its URLs.
    Returns (full_url, thumb_url).
    """
    api = f"https://{cfg.wiki_lang}.wikivoyage.org/w/api.php"
    r = session.get(
        api,
        params={
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "images",
            "titles": title,
            "imlimit": "1",
        },
        timeout=cfg.http_timeout,
    )
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", [])
    files = (pages[0].get("images") if pages else None) or []
    if not files:
        return None, None
    file_title = files[0].get("title")
    if not file_title:
        return None, None

    r2 = session.get(
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
        timeout=cfg.http_timeout,
    )
    r2.raise_for_status()
    pages2 = r2.json().get("query", {}).get("pages", [])
    if not pages2:
        return None, None
    ii = (pages2[0].get("imageinfo") or [{}])[0]
    return ii.get("url"), ii.get("thumburl")

def fetch_city_pages(cfg: Settings) -> Iterator[Dict]:
    """
    Enumerate city pages via MediaWiki Action API using a category.
    Yields: dict with title, summary, image_url, thumb_url, page_url, lang, source
    """
    endpoint = f"https://{cfg.wiki_lang}.wikivoyage.org/w/api.php"
    session = make_session(cfg)

    fetched, cont = 0, {}
    while True:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "categorymembers",
            "gcmtitle": cfg.wiki_category,
            "gcmtype": "page",
            "gcmlimit": "250",
            "prop": "extracts|pageimages",
            "exintro": "1",
            "explaintext": "1",
            "piprop": "thumbnail|original",
            "pithumbsize": "640",
        }
        params.update(cont)
        r = session.get(endpoint, params=params, timeout=cfg.http_timeout)
        r.raise_for_status()
        data = r.json()

        for p in data.get("query", {}).get("pages", []):
            title = p.get("title")
            if not title:
                continue
            extract = (p.get("extract") or "").strip()
            thumb = (p.get("thumbnail") or {})
            original = (p.get("original") or {})
            thumb_url = thumb.get("source")
            image_url = original.get("source") or thumb_url

            # Fallback to first file if still missing
            if not image_url:
                full_img, thumb_img = get_first_image_url(session, cfg, title)
                image_url = full_img or thumb_img
                thumb_url = thumb_img or full_img

            yield {
                "title": title,
                "summary": extract,
                "image_url": image_url,
                "thumb_url": thumb_url,
                "page_url": page_url(cfg, title),
                "lang": cfg.wiki_lang,
                "source": "wikivoyage",
            }
            fetched += 1
            if fetched >= cfg.ingest_limit:
                return
            time.sleep(cfg.sleep_seconds)

        if "continue" in data:
            cont = data["continue"]
        else:
            break

# ---------- Qdrant helpers ----------
def get_client(cfg: Settings) -> QdrantClient:
    return QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)

def ensure_collection(client: QdrantClient, cfg: Settings, dim: int):
    """Create collection if absent; validate vector size if present."""
    if not client.collection_exists(cfg.qdrant_collection):
        client.create_collection(
            collection_name=cfg.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=cfg.distance_enum()),
        )
    else:
        try:
            info = client.get_collection(cfg.qdrant_collection)
            vectors = getattr(info.config.params, "vectors", None)  # type: ignore[attr-defined]
            existing_dim = None
            if hasattr(vectors, "size"):
                existing_dim = vectors.size
            elif isinstance(vectors, dict) and vectors:
                existing_dim = list(vectors.values())[0].size
            if existing_dim is not None and existing_dim != dim:
                raise RuntimeError(
                    f"Collection '{cfg.qdrant_collection}' dim={existing_dim} != model dim={dim}. "
                    f"Change EMBED_MODEL or use a new collection name."
                )
        except Exception:
            pass

def count_points(client: QdrantClient, cfg: Settings) -> int:
    return client.count(collection_name=cfg.qdrant_collection, exact=True).count

def scroll_points(client: QdrantClient, cfg: Settings, limit: int = 5, offset=None):
    return client.scroll(
        collection_name=cfg.qdrant_collection, limit=limit, with_payload=True, offset=offset
    )

# ---------- Embedding ----------
def get_embedder(cfg: Settings) -> TextEmbedding:
    return TextEmbedding(model_name=cfg.embed_model)

# ---------- Commands ----------
def cmd_config():
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="white")
    for k, v in CFG.masked().items():
        table.add_row(k, v)
    console.print(Panel(table, title="Effective Configuration", border_style="cyan"))

def cmd_verify():
    try:
        c = get_client(CFG)
        cols = c.get_collections()
        table = Table(box=box.SIMPLE_HEAVY)
        table.add_column("Collection", style="bold green")
        for col in cols.collections or []:
            table.add_row(col.name)
        console.print(Panel(table, title=f"Connected to {CFG.qdrant_url}", border_style="green"))
    except Exception as e:
        console.print(Panel(str(e), title="Connection FAILED", border_style="red"))
        sys.exit(1)

def cmd_ingest():
    client = get_client(CFG)
    embedder = get_embedder(CFG)

    dim: Optional[int] = None
    batch: List[PointStruct] = []
    count = 0

    console.print(Panel(f"Ingesting up to {CFG.ingest_limit} city pages from Wikivoyage [{CFG.wiki_lang}]…",
                        border_style="magenta"))

    for item in tqdm(fetch_city_pages(CFG), total=CFG.ingest_limit, ncols=88, unit="pg"):
        text = f"{item['title']} — {item['summary']}" if item["summary"] else item["title"]
        vec = next(embedder.embed([text]))
        if dim is None:
            dim = len(vec)
            ensure_collection(client, CFG, dim)

        batch.append(PointStruct(id=stable_id(item["title"]), vector=list(vec), payload=item))
        if len(batch) >= CFG.batch_size:
            client.upsert(collection_name=CFG.qdrant_collection, points=batch)
            count += len(batch)
            batch.clear()

    if batch:
        client.upsert(collection_name=CFG.qdrant_collection, points=batch)
        count += len(batch)

    total = count_points(client, CFG)
    console.print(Panel(f"✅ Ingested {count} points. Collection total: {total}",
                        border_style="green"))

def _render_results(rows: List[Dict]):
    table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAVY)
    table.add_column("#", justify="right", width=3)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Title", style="bold")
    table.add_column("Page URL", style="blue")
    table.add_column("Image URL", style="blue")
    table.add_column("Summary (snippet)")
    for i, r in enumerate(rows, 1):
        table.add_row(
            str(i),
            f"{r['score']:.3f}",
            r["title"] or "",
            r["page_url"] or "",
            r["image_url"] or "",
            (r["summary"][:140] + "…") if r["summary"] and len(r["summary"]) > 140 else (r["summary"] or "")
        )
    console.print(table)

def cmd_search(query: str, k: Optional[int], lang: Optional[str]):
    k = k or CFG.search_k
    client = get_client(CFG)
    embedder = get_embedder(CFG)
    qv = next(embedder.embed([query]))

    qfilter = None
    if lang:
        qfilter = Filter(must=[FieldCondition(key="lang", match=MatchValue(value=lang))])

    results = client.search(
        collection_name=CFG.qdrant_collection,
        query_vector=list(qv),
        limit=k,
        with_payload=True,
        query_filter=qfilter,
    )

    # Fallback image fetch if missing
    session = make_session(CFG)
    rows: List[Dict] = []
    for r in results:
        p = r.payload or {}
        title = p.get("title", "")
        summary = p.get("summary") or ""
        page = p.get("page_url") or ""
        img = p.get("image_url") or p.get("thumb_url")
        if not img and title:
            full_img, thumb_img = get_first_image_url(session, CFG, title)
            img = full_img or thumb_img
        rows.append({
            "score": float(r.score),
            "title": title,
            "page_url": page,
            "image_url": img or "",
            "summary": summary
        })

    console.print(Panel.fit(f'🔎 Results for: "{query}" (top {k})', border_style="magenta"))
    _render_results(rows)

def cmd_stats():
    client = get_client(CFG)
    cnt = count_points(client, CFG)

    try:
        info = client.get_collection(CFG.qdrant_collection)
        vec_cfg = getattr(info.config.params, "vectors", None)  # type: ignore[attr-defined]
        dim = getattr(vec_cfg, "size", None)
        if dim is None and isinstance(vec_cfg, dict) and vec_cfg:
            dim = list(vec_cfg.values())[0].size
    except Exception:
        dim = None

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("Collection", CFG.qdrant_collection)
    table.add_row("Distance", CFG.qdrant_distance)
    table.add_row("Vector dim", str(dim or "?"))
    table.add_row("Points (exact)", str(cnt))
    console.print(Panel(table, title="Collection Stats", border_style="cyan"))

def cmd_peek(n: int):
    client = get_client(CFG)
    pts, _ = scroll_points(client, CFG, limit=n)
    rows = []
    for p in pts:
        pay = p.payload or {}
        rows.append({
            "score": 0.0,
            "title": str(pay.get("title") or ""),
            "page_url": str(pay.get("page_url") or ""),
            "image_url": str(pay.get("image_url") or pay.get("thumb_url") or ""),
            "summary": str(pay.get("summary") or "")
        })
    console.print(Panel.fit(f"First {len(rows)} points in '{CFG.qdrant_collection}'", border_style="yellow"))
    _render_results(rows)

def cmd_backfill_images(max_points: int):
    client = get_client(CFG)
    session = make_session(CFG)
    processed = 0
    offset = None
    updated = 0

    console.print(Panel(f"Backfilling image URLs for points missing images (max {max_points})…",
                        border_style="yellow"))
    while True:
        points, offset = client.scroll(
            collection_name=CFG.qdrant_collection,
            limit=256,
            with_payload=True,
            offset=offset
        )
        if not points:
            break

        for p in points:
            if processed >= max_points:
                break
            pay = p.payload or {}
            if pay.get("image_url") or pay.get("thumb_url"):
                processed += 1
                continue
            title = pay.get("title")
            if not title:
                processed += 1
                continue
            full_img, thumb_img = get_first_image_url(session, CFG, title)
            if full_img or thumb_img:
                client.set_payload(
                    collection_name=CFG.qdrant_collection,
                    payload={
                        "image_url": full_img or thumb_img,
                        "thumb_url": thumb_img or full_img
                    },
                    points=[p.id]
                )
                updated += 1
            processed += 1
        if processed >= max_points:
            break

    console.print(Panel(f"Done. Processed: {processed}, Updated: {updated}", border_style="green"))

def cmd_export(out_path: str):
    client = get_client(CFG)
    offset = None
    n = 0
    console.print(Panel(f"Exporting payloads to {out_path} (NDJSON)…", border_style="cyan"))
    with open(out_path, "w", encoding="utf-8") as f:
        while True:
            points, offset = client.scroll(
                collection_name=CFG.qdrant_collection,
                limit=512,
                with_payload=True,
                offset=offset
            )
            if not points:
                break
            for p in points:
                payload = p.payload or {}
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                n += 1
    console.print(Panel(f"Export complete. Wrote {n} lines.", border_style="green"))

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(
        description="Wikivoyage → Qdrant: ingest, search, and manage a city index (solid, env-driven)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("config", help="Show effective configuration")
    sub.add_parser("verify", help="Verify Qdrant connection")
    sub.add_parser("ingest", help="Fetch & index Wikivoyage city pages")
    sub.add_parser("stats", help="Show collection stats")
    p_peek = sub.add_parser("peek", help="Preview first N points (payload only)")
    p_peek.add_argument("--n", type=int, default=5)

    p_search = sub.add_parser("search", help="Semantic search")
    p_search.add_argument("query", type=str)
    p_search.add_argument("--k", type=int, default=None, help="Top-K results")
    p_search.add_argument("--lang", type=str, default=None, help="Filter by payload.lang")

    p_back = sub.add_parser("backfill-images", help="Fetch & persist image URL where missing")
    p_back.add_argument("--max", type=int, default=5000)

    p_export = sub.add_parser("export", help="Export payloads to NDJSON")
    p_export.add_argument("--out", type=str, default="wikivoyage_cities.ndjson")

    args = parser.parse_args()

    try:
        if args.cmd == "config":
            cmd_config()
        elif args.cmd == "verify":
            cmd_verify()
        elif args.cmd == "ingest":
            cmd_ingest()
        elif args.cmd == "stats":
            cmd_stats()
        elif args.cmd == "peek":
            cmd_peek(args.n)
        elif args.cmd == "search":
            cmd_search(args.query, k=args.k, lang=args.lang)
        elif args.cmd == "backfill-images":
            cmd_backfill_images(args.max)
        elif args.cmd == "export":
            cmd_export(args.out)
    except KeyboardInterrupt:
        console.print("[red]\nInterrupted by user.[/red]")
        sys.exit(130)
    except requests.HTTPError as e:
        console.print(Panel(f"HTTP error: {e}", border_style="red"))
        sys.exit(1)
    except Exception as e:
        console.print(Panel(f"Unexpected error: {e}", border_style="red"))
        sys.exit(1)

if __name__ == "__main__":
    main()
