"""
RAG retriever: searches all available ChromaDB collections and returns
a combined context string for the LLM.

Collections queried (in order):
  1. 'wikivoyage'  – Phase 1 data
  2. 'web_crawled' – Phase 2 data (skipped gracefully if not yet ingested)
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag.cities import CHROMA_DIR, CITY_ALIASES, EMBED_MODEL

# ── Configuration ─────────────────────────────────────────────────────────────

COLLECTIONS = ["wikivoyage", "web_crawled"]
N_RESULTS   = 5   # chunks to retrieve per city key per collection

# ── Retriever class ───────────────────────────────────────────────────────────

class TravelRetriever:
    """
    Semantic retriever that searches multiple ChromaDB collections.

    Parameters
    ----------
    chroma_dir : str
        Path to the persistent ChromaDB directory.
    """

    def __init__(self, chroma_dir: str = CHROMA_DIR) -> None:
        self.chroma_dir = chroma_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self._stores: dict[str, Chroma] = {}

    # ── private helpers ───────────────────────────────────────────────────────

    def _load_store(self, collection_name: str) -> Chroma | None:
        """Return a Chroma store for *collection_name*, or None if unavailable."""
        if collection_name in self._stores:
            return self._stores[collection_name]
        try:
            store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_dir,
            )
            self._stores[collection_name] = store
            return store
        except Exception as exc:
            print(f"[retriever] Could not open '{collection_name}': {exc}")
            return None

    def _normalise_city(self, city: str) -> str:
        """
        Strip country suffix: 'Lisbon, Portugal' → 'Lisbon'.
        ChromaDB metadata stores just the bare city name.
        """
        return city.split(",")[0].strip()

    def _city_keys(self, city: str) -> list[str]:
        """
        Return the list of metadata keys to search for *city*.
        Handles aliases (e.g. Malta ↔ Valletta) so both collections
        are searched when either name is selected.
        """
        key = self._normalise_city(city)
        return CITY_ALIASES.get(key, [key])

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve(self, city: str, query: str, n_results: int = N_RESULTS) -> str:
        """
        Retrieve relevant text chunks for *city* using semantic similarity.

        Strategy
        --------
        1. For each city key (including aliases), run a filtered similarity
           search in every available collection.
        2. If the filtered search returns nothing for ALL keys, return a
           fallback message — do NOT fall back to an unfiltered search,
           which would mix in unrelated cities and mislead the LLM.
        3. Deduplicate and concatenate results.

        Returns a single string ready to be injected into the LLM prompt.
        """
        city_keys  = self._city_keys(city)
        collected: list[str] = []

        for name in COLLECTIONS:
            store = self._load_store(name)
            if store is None:
                continue

            for key in city_keys:
                try:
                    docs = store.similarity_search(
                        query,
                        k=n_results,
                        filter={"city": key},
                    )
                except Exception:
                    docs = []

                for doc in docs:
                    chunk = doc.page_content.strip()
                    if chunk and chunk not in collected:
                        collected.append(chunk)

        if not collected:
            return (
                f"No stored travel data found for {city}. "
                "Please generate advice based on your general knowledge of this destination."
            )

        return "\n\n---\n\n".join(collected)
