"""
Phase 1 ingest: fetch WikiVoyage articles for target cities and store them in
the ChromaDB 'wikivoyage' collection via LangChain.

Usage:
    # Full rebuild (drops existing collection, re-ingests all cities):
    python -m rag.ingest_wikivoyage

    # Add missing cities only (no rebuild):
    python -m rag.ingest_wikivoyage --add "Málaga,Kraków,Marrakech,Fez"
"""
import argparse
import os
import sys
import unicodedata

import chromadb
import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.cities import CHROMA_DIR, COLLECTION, EMBED_MODEL, TARGET_CITIES

# ── Configuration ─────────────────────────────────────────────────────────────

WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"
RAW_DATA_DIR   = "./data/wikivoyage"

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

_HEADERS = {
    "User-Agent": "MindfulTourismApp/1.0 (educational prototype; python-requests)"
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _city_to_filename(city: str) -> str:
    """
    Convert a city name to a safe ASCII filename.
    'Kraków' → 'krakow.txt', 'Málaga' → 'malaga.txt'
    """
    normalized = unicodedata.normalize("NFKD", city)
    ascii_name = normalized.encode("ASCII", "ignore").decode().lower()
    return (ascii_name or city.lower()).replace(" ", "_")


def fetch_wikivoyage(city: str) -> str:
    """
    Return the plain-text extract of the WikiVoyage article for *city*.
    Raises RuntimeError if no extract is found.
    """
    params = {
        "action":      "query",
        "titles":      city,
        "prop":        "extracts",
        "explaintext": True,
        "format":      "json",
    }
    resp = requests.get(WIKIVOYAGE_API, params=params, headers=_HEADERS, timeout=15)
    resp.raise_for_status()

    pages = resp.json()["query"]["pages"]
    page  = next(iter(pages.values()))

    if "extract" not in page or not page["extract"].strip():
        raise RuntimeError(f"No WikiVoyage extract found for '{city}'")

    return page["extract"]


def _load_embeddings() -> HuggingFaceEmbeddings:
    print(f"Loading embedding model: {EMBED_MODEL}")
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def _city_to_docs(city: str, splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    """Fetch and chunk one city. Returns [] on failure."""
    print(f"Fetching WikiVoyage → {city} …", end=" ", flush=True)
    try:
        text = fetch_wikivoyage(city)
    except Exception as exc:
        print(f"SKIP ({exc})")
        return []

    # Save raw text with a safe ASCII filename
    filename = _city_to_filename(city) + ".txt"
    raw_path = os.path.join(RAW_DATA_DIR, filename)
    with open(raw_path, "w", encoding="utf-8") as fh:
        fh.write(text)

    chunks = splitter.split_text(text)
    docs = [
        Document(
            page_content=chunk,
            metadata={"city": city, "source": "wikivoyage", "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]
    print(f"OK  ({len(docs)} chunks)")
    return docs


# ── Full rebuild ──────────────────────────────────────────────────────────────

def ingest():
    """
    Fetch, chunk, embed and store ALL target cities.
    Drops and recreates the collection to prevent duplicate chunks.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR,   exist_ok=True)

    # Drop existing collection to avoid duplicate chunks on re-run
    try:
        raw_client = chromadb.PersistentClient(path=CHROMA_DIR)
        raw_client.delete_collection(COLLECTION)
        print(f"Dropped existing '{COLLECTION}' collection.")
    except Exception:
        pass  # Collection did not exist yet

    embeddings = _load_embeddings()
    splitter   = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    all_docs: list[Document] = []
    for city in TARGET_CITIES:
        all_docs.extend(_city_to_docs(city, splitter))

    if not all_docs:
        print("Nothing to ingest — exiting.")
        sys.exit(1)

    print(f"\nEmbedding and storing {len(all_docs)} chunks into '{COLLECTION}' …")
    Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Done. ChromaDB stored at: {CHROMA_DIR}")


# ── Add missing cities only ───────────────────────────────────────────────────

def add_cities(cities: list[str]):
    """
    Fetch *cities* and add them to the existing collection without rebuilding.
    Skips cities that are already present to avoid duplicate chunks.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    embeddings = _load_embeddings()
    splitter   = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    store = Chroma(
        collection_name=COLLECTION,
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
        new_docs.extend(_city_to_docs(city, splitter))

    if not new_docs:
        print("Nothing new to add.")
        return

    print(f"\nAdding {len(new_docs)} chunks to existing '{COLLECTION}' collection …")
    store.add_documents(new_docs)
    print("Done.")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WikiVoyage → ChromaDB ingest")
    parser.add_argument(
        "--add",
        metavar="CITIES",
        help='Comma-separated city names to add to the existing collection, '
             'e.g. --add "Málaga,Kraków,Marrakech,Fez"',
    )
    args = parser.parse_args()

    if args.add:
        cities = [c.strip() for c in args.add.split(",") if c.strip()]
        add_cities(cities)
    else:
        ingest()
