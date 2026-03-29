"""
Phase 2 stub: web-crawling ingest for the 'web_crawled' ChromaDB collection.

This module will be implemented in Phase 2.  It is intentionally left as a
stub so that:
  - retriever.py can safely attempt to open the collection (gracefully missing),
  - the project structure is already in place for future work.

Planned data sources
--------------------
- travel blogs (e.g. Lonely Planet articles)
- tourism board pages
- crowd-sourced tips (e.g. TripAdvisor top-tips scrape, within T&C)
"""
from __future__ import annotations

# Collection name that Phase 2 will populate
COLLECTION = "web_crawled"

# ── Stub API ──────────────────────────────────────────────────────────────────

def crawl_and_ingest(cities: list[str], *, chroma_dir: str = "./data/chroma") -> None:
    """
    Crawl travel websites for *cities* and store chunks in ChromaDB.

    Phase 2 TODO:
      1. Build a URL list per city (tourism boards, travel blogs).
      2. Fetch HTML with `requests` + parse with `BeautifulSoup`.
      3. Clean & chunk the text (LangChain RecursiveCharacterTextSplitter).
      4. Embed with HuggingFaceEmbeddings and upsert into the
         '{COLLECTION}' ChromaDB collection.
    """
    raise NotImplementedError(
        "Phase 2: web crawling ingest is not yet implemented. "
        "Run rag/ingest_wikivoyage.py for Phase 1 data."
    )
