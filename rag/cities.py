"""
Shared constants and city list for the Mindful Tourism App.

Centralising EMBED_MODEL, CHROMA_DIR, COLLECTION and TARGET_CITIES here
prevents the two-file duplication that would cause silent RAG failures if
one side is changed without updating the other.
"""

# Embedding model — must be identical at ingest time AND retrieval time.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ChromaDB persistence directory (relative to project root).
CHROMA_DIR = "./data/chroma"

# Collection name for WikiVoyage data.
COLLECTION = "wikivoyage"

# Cities whose WikiVoyage data is stored under multiple metadata keys.
# When searching for a key city, all aliases are searched together.
CITY_ALIASES: dict[str, list[str]] = {
    "Malta":    ["Malta", "Valletta"],
    "Valletta": ["Malta", "Valletta"],
}

# Canonical list of ingestable destinations.
# Names must match WikiVoyage article titles exactly.
TARGET_CITIES: list[str] = [
    # Iberian Peninsula
    "Lisbon",
    "Porto",
    "Madrid",
    "Seville",
    "Granada",
    "Valencia",
    "Málaga",
    "Bilbao",
    # France
    "Paris",
    "Nice",
    "Lyon",
    "Bordeaux",
    "Marseille",
    # Italy
    "Rome",
    "Florence",
    "Venice",
    "Naples",
    "Milan",
    "Palermo",
    # Low Countries & Central Europe
    "Amsterdam",
    "Brussels",
    "Vienna",
    "Prague",
    "Budapest",
    "Kraków",
    # Switzerland
    "Zurich",
    "Geneva",
    # British Isles
    "Edinburgh",
    "Dublin",
    # Balkans & Adriatic
    "Athens",
    "Thessaloniki",
    "Dubrovnik",
    "Split",
    "Kotor",
    "Sarajevo",
    "Ljubljana",
    # Greek Islands
    "Santorini",
    "Mykonos",
    "Rhodes",
    "Corfu",
    "Crete",
    # Mediterranean Islands
    "Malta",
    "Valletta",
    # Morocco
    "Marrakech",
    "Casablanca",
    "Fez",
    "Tangier",
    # Egypt
    "Cairo",
    "Alexandria",
    "Luxor",
    # Tunisia & Algeria
    "Tunis",
    "Algiers",
]
