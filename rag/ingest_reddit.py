"""
Phase 2 ingest: fetch Reddit travel tips via Pullpush (Pushshift successor)
and store them in the ChromaDB 'reddit_tips' collection.

Pullpush is a free, no-auth Reddit archive API.

Usage:
    # Ingest all target cities:
    python -m rag.ingest_reddit

    # Ingest specific cities only:
    python -m rag.ingest_reddit --cities "Lisbon,Cairo,Athens"

    # Limit posts per query (default 25):
    python -m rag.ingest_reddit --limit 50
"""
import argparse
import os
import sys
import time

import chromadb
import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.cities import CHROMA_DIR, EMBED_MODEL, REDDIT_COLLECTION, TARGET_CITIES

# ── Configuration ─────────────────────────────────────────────────────────────

PULLPUSH_SUBMISSIONS = "https://api.pullpush.io/reddit/search/submission/"
PULLPUSH_COMMENTS    = "https://api.pullpush.io/reddit/search/comment/"

SUBREDDITS = ["travel", "solotravel", "EuropeTravel", "foodtravel"]

# Search queries per city — {city} is replaced at runtime.
QUERY_TEMPLATES = [
    "{city} travel tips",
    "{city} food restaurant local",
    "{city} hidden gems secret",
    "{city} hotel accommodation stay",
]

RAW_DATA_DIR  = "./data/reddit"
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 80

_HEADERS = {
    "User-Agent": "MindfulTourismApp/1.0 (educational prototype)"
}

# Minimum score to include a post or comment (filters out low-quality content).
MIN_SCORE = 2

# Delay between API requests to be respectful of rate limits.
REQUEST_DELAY = 1.0  # seconds

# ── Pullpush API helpers ──────────────────────────────────────────────────────

def _fetch_submissions(query: str, subreddit: str, limit: int = 25) -> list[dict]:
    """
    Search Reddit submissions via Pullpush for a single subreddit.
    Returns a list of raw submission dicts.
    """
    params = {
        "q":         query,
        "subreddit": subreddit,
        "size":      min(limit, 100),
    }
    try:
        resp = requests.get(
            PULLPUSH_SUBMISSIONS, params=params,
            headers=_HEADERS, timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as exc:
        print(f"    [warn] submission search failed ({subreddit}): {exc}")
        return []


def _fetch_comments(query: str, subreddit: str, limit: int = 25) -> list[dict]:
    """
    Search Reddit comments via Pullpush for a single subreddit.
    Returns a list of raw comment dicts.
    """
    params = {
        "q":         query,
        "subreddit": subreddit,
        "size":      min(limit, 100),
    }
    try:
        resp = requests.get(
            PULLPUSH_COMMENTS, params=params,
            headers=_HEADERS, timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as exc:
        print(f"    [warn] comment search failed ({subreddit}): {exc}")
        return []


# ── Document building ─────────────────────────────────────────────────────────

def _submission_to_text(sub: dict) -> str | None:
    """Convert a submission dict to readable text. Returns None if too short."""
    title = sub.get("title", "").strip()
    body  = sub.get("selftext", "").strip()
    # Skip removed/deleted posts
    if body in ("[removed]", "[deleted]", ""):
        body = ""
    score = sub.get("score", 0)
    if score < MIN_SCORE:
        return None
    text = f"{title}\n\n{body}" if body else title
    # Skip very short posts (likely just a link or image)
    if len(text) < 50:
        return None
    subreddit = sub.get("subreddit", "unknown")
    return f"[Reddit r/{subreddit}, score:{score}]\n{text}"


def _comment_to_text(comment: dict) -> str | None:
    """Convert a comment dict to readable text. Returns None if too short."""
    body  = comment.get("body", "").strip()
    score = comment.get("score", 0)
    if score < MIN_SCORE or len(body) < 30:
        return None
    if body in ("[removed]", "[deleted]"):
        return None
    subreddit = comment.get("subreddit", "unknown")
    return f"[Reddit comment r/{subreddit}, score:{score}]\n{body}"


def _fetch_city_texts(city: str, limit: int) -> list[str]:
    """
    Fetch all Reddit texts (submissions + comments) for one city.
    Returns a deduplicated list of text strings.
    """
    seen: set[str] = set()
    texts: list[str] = []

    for template in QUERY_TEMPLATES:
        query = template.format(city=city)

        for subreddit in SUBREDDITS:
            # Submissions
            for sub in _fetch_submissions(query, subreddit, limit):
                text = _submission_to_text(sub)
                if text and text not in seen:
                    seen.add(text)
                    texts.append(text)
            time.sleep(REQUEST_DELAY)

            # Comments
            for comment in _fetch_comments(query, subreddit, limit):
                text = _comment_to_text(comment)
                if text and text not in seen:
                    seen.add(text)
                    texts.append(text)
            time.sleep(REQUEST_DELAY)

    return texts


# ── Ingest pipeline ───────────────────────────────────────────────────────────

def ingest(cities: list[str] | None = None, limit: int = 25):
    """
    Fetch Reddit data for cities and store in ChromaDB.
    Drops and recreates the collection to avoid duplicates.
    """
    cities = cities or TARGET_CITIES
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR,   exist_ok=True)

    # Drop existing collection
    try:
        raw_client = chromadb.PersistentClient(path=CHROMA_DIR)
        raw_client.delete_collection(REDDIT_COLLECTION)
        print(f"Dropped existing '{REDDIT_COLLECTION}' collection.")
    except Exception:
        pass

    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    all_docs: list[Document] = []

    for city in cities:
        print(f"\nFetching Reddit → {city} …", end=" ", flush=True)
        texts = _fetch_city_texts(city, limit)

        if not texts:
            print("SKIP (no data)")
            continue

        # Save raw text for inspection
        raw_path = os.path.join(RAW_DATA_DIR, f"{city.lower().replace(' ', '_')}.txt")
        with open(raw_path, "w", encoding="utf-8") as fh:
            fh.write("\n\n===\n\n".join(texts))

        # Chunk all texts together, then create Documents
        combined = "\n\n---\n\n".join(texts)
        chunks   = splitter.split_text(combined)
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "city": city,
                    "source": "reddit",
                    "chunk_index": i,
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        all_docs.extend(docs)
        print(f"OK  ({len(texts)} posts/comments → {len(docs)} chunks)")

    if not all_docs:
        print("\nNothing to ingest — exiting.")
        sys.exit(1)

    print(f"\nEmbedding and storing {len(all_docs)} chunks into '{REDDIT_COLLECTION}' …")
    Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name=REDDIT_COLLECTION,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Done. ChromaDB stored at: {CHROMA_DIR}")


# ── Add missing cities only ───────────────────────────────────────────────────

def add_cities(cities: list[str], limit: int = 25):
    """
    Fetch Reddit data for *cities* and ADD to the existing collection
    without rebuilding. Skips cities already present.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    store = Chroma(
        collection_name=REDDIT_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Find cities already in the collection
    existing = set()
    try:
        result = store.get(include=["metadatas"])
        for meta in result.get("metadatas", []):
            if meta and "city" in meta:
                existing.add(meta["city"])
    except Exception:
        pass

    new_docs: list[Document] = []

    for city in cities:
        if city in existing:
            print(f"Already in DB → {city}  (skipping)")
            continue

        print(f"\nFetching Reddit → {city} …", end=" ", flush=True)
        texts = _fetch_city_texts(city, limit)

        if not texts:
            print("SKIP (no data)")
            continue

        raw_path = os.path.join(RAW_DATA_DIR, f"{city.lower().replace(' ', '_')}.txt")
        with open(raw_path, "w", encoding="utf-8") as fh:
            fh.write("\n\n===\n\n".join(texts))

        combined = "\n\n---\n\n".join(texts)
        chunks   = splitter.split_text(combined)
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "city": city,
                    "source": "reddit",
                    "chunk_index": i,
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        new_docs.extend(docs)
        print(f"OK  ({len(texts)} posts/comments → {len(docs)} chunks)")

    if not new_docs:
        print("\nNothing new to add.")
        return

    print(f"\nAdding {len(new_docs)} chunks to existing '{REDDIT_COLLECTION}' collection …")
    # ChromaDB has a max batch size (~5000). Split into smaller batches.
    BATCH_SIZE = 5000
    for start in range(0, len(new_docs), BATCH_SIZE):
        batch = new_docs[start : start + BATCH_SIZE]
        store.add_documents(batch)
        print(f"  Batch {start // BATCH_SIZE + 1}: {len(batch)} chunks added")
    print("Done.")


# ── Ingest from local files only (no API calls) ──────────────────────────────

def ingest_local(cities: list[str] | None = None):
    """
    Read existing text files from data/reddit/ and embed into ChromaDB.
    Skips cities already in the collection. No API calls.
    """
    cities = cities or TARGET_CITIES
    os.makedirs(CHROMA_DIR, exist_ok=True)

    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    store = Chroma(
        collection_name=REDDIT_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Find cities already in the collection
    existing = set()
    try:
        result = store.get(include=["metadatas"])
        for meta in result.get("metadatas", []):
            if meta and "city" in meta:
                existing.add(meta["city"])
    except Exception:
        pass

    new_docs: list[Document] = []

    for city in cities:
        if city in existing:
            print(f"Already in DB → {city}  (skipping)")
            continue

        # Find the local file
        raw_path = os.path.join(RAW_DATA_DIR, f"{city.lower().replace(' ', '_')}.txt")
        if not os.path.exists(raw_path):
            # Try with accent-stripped filename
            import unicodedata
            norm = unicodedata.normalize("NFKD", city).encode("ASCII", "ignore").decode().lower()
            raw_path = os.path.join(RAW_DATA_DIR, f"{norm}.txt")

        if not os.path.exists(raw_path):
            print(f"No local file → {city}  (skipping)")
            continue

        with open(raw_path, "r", encoding="utf-8") as fh:
            content = fh.read()

        if not content.strip():
            print(f"Empty file → {city}  (skipping)")
            continue

        chunks = splitter.split_text(content)
        docs = [
            Document(
                page_content=chunk,
                metadata={"city": city, "source": "reddit", "chunk_index": i},
            )
            for i, chunk in enumerate(chunks)
        ]
        new_docs.extend(docs)
        print(f"Local file → {city}  ({len(docs)} chunks)")

    if not new_docs:
        print("\nNothing new to add.")
        return

    print(f"\nEmbedding and storing {len(new_docs)} chunks into '{REDDIT_COLLECTION}' …")
    BATCH_SIZE = 5000
    for start in range(0, len(new_docs), BATCH_SIZE):
        batch = new_docs[start : start + BATCH_SIZE]
        store.add_documents(batch)
        print(f"  Batch {start // BATCH_SIZE + 1}: {len(batch)} chunks added")
    print("Done.")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit (Pullpush) → ChromaDB ingest")
    parser.add_argument(
        "--cities",
        metavar="CITIES",
        help='Comma-separated city names (default: all TARGET_CITIES)',
    )
    parser.add_argument(
        "--add",
        metavar="CITIES",
        help='Comma-separated city names to ADD to existing collection',
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Embed from local text files only (no API calls). Use with --cities to filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Max submissions/comments per query per city (default: 25)",
    )
    args = parser.parse_args()

    if args.add:
        city_list = [c.strip() for c in args.add.split(",") if c.strip()]
        add_cities(city_list, limit=args.limit)
    elif getattr(args, 'local', False):
        city_list = None
        if args.cities:
            city_list = [c.strip() for c in args.cities.split(",") if c.strip()]
        ingest_local(cities=city_list)
    elif args.cities:
        city_list = [c.strip() for c in args.cities.split(",") if c.strip()]
        ingest(cities=city_list, limit=args.limit)
    else:
        ingest(limit=args.limit)
