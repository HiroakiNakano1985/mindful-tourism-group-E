"""
RAG retriever: searches ChromaDB collections and returns
a combined context string for the LLM.

Collections:
  - 'wikivoyage'    – general city guides (Phase 1)
  - 'reddit_tips'   – niche travel tips from Reddit (Phase 2)
  - 'google_places' – hotel/restaurant reviews (Phase 2, future)

The `retrieve()` method accepts a `collections` parameter so callers
can choose which sources to search per section (e.g. etiquette from
wikivoyage + reddit, hotels from wikivoyage only).
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag.cities import ALL_COLLECTIONS, CHROMA_DIR, CITY_ALIASES, EMBED_MODEL

# ── Configuration ─────────────────────────────────────────────────────────────

N_RESULTS = 5   # chunks to retrieve per city key per collection

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
        """Strip country suffix: 'Lisbon, Portugal' → 'Lisbon'."""
        return city.split(",")[0].strip()

    def _city_keys(self, city: str) -> list[str]:
        """Return metadata keys for *city*, including aliases."""
        key = self._normalise_city(city)
        return CITY_ALIASES.get(key, [key])

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        city: str,
        query: str,
        collections: list[str] | None = None,
        n_results: int = N_RESULTS,
    ) -> str:
        """
        Retrieve relevant text chunks for *city* using semantic similarity.

        Parameters
        ----------
        city : str
            Destination city (may include country suffix).
        query : str
            Semantic search query.
        collections : list[str] | None
            Which ChromaDB collections to search.
            Defaults to ALL_COLLECTIONS if not specified.
        n_results : int
            Number of chunks per city key per collection.

        Returns a single string ready to be injected into the LLM prompt.
        """
        search_collections = collections or ALL_COLLECTIONS
        city_keys  = self._city_keys(city)
        collected: list[str] = []

        for name in search_collections:
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
