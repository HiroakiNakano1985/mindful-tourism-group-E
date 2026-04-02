"""
Phase 2 ingest: fetch hotel and restaurant reviews from Google Places API (New)
and store them in the ChromaDB 'google_places' collection.

Uses the Places API (New) REST endpoints:
  - Text Search  → find places by city + category
  - Place Details → get reviews for each place

Usage:
    # Ingest all target cities:
    python -m rag.ingest_google_places

    # Ingest specific cities only:
    python -m rag.ingest_google_places --cities "Lisbon,Cairo,Athens"

    # Limit places per category per city (default 5):
    python -m rag.ingest_google_places --limit 10
"""
import argparse
import os
import sys
import time

import chromadb
import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag.cities import (
    CHROMA_DIR,
    EMBED_MODEL,
    GOOGLE_PLACES_COLLECTION,
    TARGET_CITIES,
)

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

TEXT_SEARCH_URL  = "https://places.googleapis.com/v1/places:searchText"
PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places/{place_id}"

# Categories to search for each city.
SEARCH_CATEGORIES = [
    ("hotel", "best hotels in {city}"),
    ("restaurant", "best restaurants in {city}"),
    ("attraction", "top attractions in {city}"),
]

RAW_DATA_DIR  = "./data/google_places"
REQUEST_DELAY = 0.5  # seconds between API calls

# ── API helpers ───────────────────────────────────────────────────────────────

def _text_search(query: str, limit: int = 5) -> list[dict]:
    """
    Search for places using Google Places Text Search (New).
    Returns a list of place dicts with id, displayName, rating, address.
    """
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.rating,"
            "places.formattedAddress,places.priceLevel"
        ),
    }
    body = {
        "textQuery": query,
        "pageSize": min(limit, 20),
        "languageCode": "en",
    }
    try:
        resp = requests.post(TEXT_SEARCH_URL, json=body, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json().get("places", [])
    except Exception as exc:
        print(f"    [warn] text search failed: {exc}")
        return []


def _get_place_reviews(place_id: str) -> list[dict]:
    """
    Fetch reviews for a place via Place Details (New).
    Returns up to 5 reviews (Google API limit).
    """
    url = PLACE_DETAILS_URL.format(place_id=place_id)
    headers = {
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "id,displayName,rating,reviews",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json().get("reviews", [])
    except Exception as exc:
        print(f"    [warn] place details failed for {place_id}: {exc}")
        return []


# ── Document building ─────────────────────────────────────────────────────────

def _review_to_text(review: dict, place_name: str) -> str | None:
    """
    Convert a Google Places review dict to readable text.
    Returns None if review has no text.
    """
    text_obj = review.get("text", {})
    text = text_obj.get("text", "").strip() if isinstance(text_obj, dict) else ""
    if not text or len(text) < 20:
        return None

    rating = review.get("rating", "?")
    author = review.get("authorAttribution", {}).get("displayName", "Anonymous")
    return f"[Google Review of {place_name}, rating:{rating}/5, by:{author}]\n{text}"


def _fetch_city_data(city: str, limit: int) -> list[tuple[str, str, str]]:
    """
    Fetch places and their reviews for one city.
    Returns a list of (text, category, place_name) tuples.
    """
    results: list[tuple[str, str, str]] = []

    for category, query_template in SEARCH_CATEGORIES:
        query = query_template.format(city=city)
        places = _text_search(query, limit)
        time.sleep(REQUEST_DELAY)

        for place in places:
            place_id = place.get("id", "")
            name_obj = place.get("displayName", {})
            place_name = name_obj.get("text", "Unknown") if isinstance(name_obj, dict) else "Unknown"
            rating = place.get("rating", "N/A")

            # Fetch reviews for this place
            reviews = _get_place_reviews(place_id)
            time.sleep(REQUEST_DELAY)

            for review in reviews:
                text = _review_to_text(review, place_name)
                if text:
                    results.append((text, category, place_name))

            # Also store a summary line even if no reviews
            if not reviews:
                address = place.get("formattedAddress", "")
                price = place.get("priceLevel", "")
                summary = (
                    f"[Google Places: {place_name}]\n"
                    f"Category: {category}, Rating: {rating}/5\n"
                    f"Address: {address}\n"
                    f"Price level: {price}"
                )
                results.append((summary, category, place_name))

    return results


# ── Ingest pipeline ───────────────────────────────────────────────────────────

def ingest(cities: list[str] | None = None, limit: int = 5):
    """
    Fetch Google Places data for cities and store in ChromaDB.
    Drops and recreates the collection to avoid duplicates.
    """
    if not API_KEY:
        print("ERROR: GOOGLE_PLACES_API_KEY not set in .env")
        sys.exit(1)

    cities = cities or TARGET_CITIES
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR,   exist_ok=True)

    # Drop existing collection
    try:
        raw_client = chromadb.PersistentClient(path=CHROMA_DIR)
        raw_client.delete_collection(GOOGLE_PLACES_COLLECTION)
        print(f"Dropped existing '{GOOGLE_PLACES_COLLECTION}' collection.")
    except Exception:
        pass

    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    all_docs: list[Document] = []

    for city in cities:
        print(f"\nFetching Google Places → {city} …", end=" ", flush=True)
        data = _fetch_city_data(city, limit)

        if not data:
            print("SKIP (no data)")
            continue

        # Save raw text for inspection
        raw_path = os.path.join(RAW_DATA_DIR, f"{city.lower().replace(' ', '_')}.txt")
        with open(raw_path, "w", encoding="utf-8") as fh:
            fh.write("\n\n===\n\n".join(t[0] for t in data))

        docs = [
            Document(
                page_content=text,
                metadata={
                    "city": city,
                    "source": "google_places",
                    "category": category,
                    "place_name": place_name,
                    "chunk_index": i,
                },
            )
            for i, (text, category, place_name) in enumerate(data)
        ]
        all_docs.extend(docs)
        print(f"OK  ({len(data)} reviews/entries → {len(docs)} docs)")

    if not all_docs:
        print("\nNothing to ingest — exiting.")
        sys.exit(1)

    print(f"\nEmbedding and storing {len(all_docs)} docs into '{GOOGLE_PLACES_COLLECTION}' …")
    Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name=GOOGLE_PLACES_COLLECTION,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Done. ChromaDB stored at: {CHROMA_DIR}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Places → ChromaDB ingest")
    parser.add_argument(
        "--cities",
        metavar="CITIES",
        help='Comma-separated city names (default: all TARGET_CITIES)',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max places per category per city (default: 5)",
    )
    args = parser.parse_args()

    city_list = None
    if args.cities:
        city_list = [c.strip() for c in args.cities.split(",") if c.strip()]

    ingest(cities=city_list, limit=args.limit)
