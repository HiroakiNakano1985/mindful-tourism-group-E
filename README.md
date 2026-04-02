# 🌍 Mindful Tourism App — Group E

---

## Overview

This prototype helps users discover travel destinations mindfully. It works in two stages:

- **Stage 1** — The user describes a trip in natural language. Gemini 2.5 Flash recommends 4 destinations from a curated list of 52 cities, each with a photo, vibe tags, flight time, and price estimate.
- **Stage 2** — The user selects a city. A RAG pipeline retrieves local knowledge from three data sources (WikiVoyage, Reddit, Google Places) via ChromaDB, and 5 parallel LLM calls generate:
  - **City Info** — language, currency, timezone, and climate at a glance
  - **Recommended Places** — tailored to the user's interests, with photos, an interactive Folium map, and a "Mindful Moment" for each spot
  - **Where to Stay** — hotels extracted from Google Places reviews, with star ratings and Booking.com links
  - **Local Etiquette** — specific, positive insider tips (not generic warnings)
  - **Mindful Pacing** — 8 tips across 6 categories: timing, local connection, movement, doing nothing, local scale, and sustainability

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit (sidebar layout, 2-column cards, interactive maps) |
| LLM | Google Gemini 2.5 Flash Lite (JSON mode, 5 parallel calls) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | ChromaDB (persistent, 3 collections) |
| RAG | LangChain + LangChain-Chroma |
| Maps | Folium + OpenStreetMap (Nominatim + Google Places geocoding fallback) |
| Photos | Google Places Photos API |
| Data sources | WikiVoyage API, Reddit (Pullpush API), Google Places API (New) |

---

## RAG Architecture

Each section of the travel guide searches different ChromaDB collections to get the most relevant context:

```
ChromaDB
├── wikivoyage       ← city overviews, general info (52 cities)
├── reddit_tips      ← niche tips, local knowledge, hidden gems
└── google_places    ← hotel/restaurant reviews with ratings

Section-specific retrieval:
  📍 Recommended Places  → wikivoyage + reddit_tips
  🏨 Where to Stay       → wikivoyage + google_places
  🙏 Local Etiquette     → wikivoyage + reddit_tips
  🌿 Mindful Pacing      → wikivoyage + reddit_tips
```

---

## LLM Architecture

Stage 2 uses 5 parallel Gemini calls for quality and speed:

```
User selects a city
        ↓
  4 RAG queries (section-specific)
        ↓
  ┌─────────────────────────────────────────────────────┐
  │ Parallel LLM calls (ThreadPoolExecutor, 5 workers)  │
  │                                                     │
  │  _generate_city_info()  → language, currency, etc.  │
  │  _generate_places()     → recommended places        │
  │  _generate_hotels()     → hotels from reviews       │
  │  _generate_etiquette()  → etiquette tips            │
  │  _generate_pacing()     → mindful pacing tips       │
  └─────────────────────────────────────────────────────┘
        ↓
  Combined result → Streamlit UI
```

---

## Key Features

- **Vibe Tags** — Stage 1 city cards show emoji-tagged vibes (e.g. "🍷 Wine marathon vineyards") instead of generic descriptions. Tags carry through to Stage 2 to ensure recommended places match.
- **Mindful Moments** — Each recommended place includes a specific suggestion for how to *experience* it, not just visit it (e.g. "Walk through without buying anything first. Come back to the stall with the best smell.").
- **Combined Map** — All places (numbered) and hotels (lettered, color-coded by category) shown on a single interactive map.
- **Categorized Pacing** — Tips are grouped by category (⏰ Timing, 🤝 Connection, 🚶 Movement, ☕ Doing Nothing, 📍 Local Scale, 🌱 Sustainability) with color-coded cards.
- **Google Places Photos** — City and place images sourced from Google Places for comprehensive coverage.

---

## File Structure

```
mindful-tourism-group-E/
├── app.py                        # Streamlit main app (sidebar + 2-stage UI)
├── requirements.txt
├── .env.example                  # API token template
├── .gitignore
├── .gitattributes                # Git LFS tracking rules
│
├── llm/
│   └── client.py                 # Gemini API (Stage 1 + 5 parallel Stage 2 calls)
│
├── rag/
│   ├── cities.py                 # Shared constants, city list, collection names
│   ├── ingest_wikivoyage.py      # WikiVoyage API → ChromaDB
│   ├── ingest_reddit.py          # Reddit (Pullpush API) → ChromaDB
│   ├── ingest_google_places.py   # Google Places API → ChromaDB
│   ├── ingest_crawler.py         # Web crawling stub (future)
│   └── retriever.py              # Section-specific multi-collection retrieval
│
└── data/                         # Stored via Git LFS
    ├── wikivoyage/               # Raw WikiVoyage text
    ├── reddit/                   # Raw Reddit posts/comments
    ├── google_places/            # Raw Google Places reviews
    └── chroma/                   # ChromaDB vector store (3 collections)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API tokens

```bash
cp .env.example .env
```

Edit `.env`:

```
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_PLACES_API_KEY=AIza_xxxxxxxxxxxxxxxxxxxxxxxx
```

- HuggingFace token (for embeddings): https://huggingface.co/settings/tokens
- Gemini API key: https://aistudio.google.com/apikey
- Google Places API key: https://console.cloud.google.com/

### 3. Data setup

The `data/` directory is included via Git LFS, so **no ingest is needed after cloning**. Just run:

```bash
git lfs pull
```

To rebuild or update data manually:

```bash
# WikiVoyage (all cities)
python -m rag.ingest_wikivoyage

# Reddit (all cities via Pullpush — no API key needed)
python -m rag.ingest_reddit

# Reddit (add specific cities without rebuilding)
python -m rag.ingest_reddit --add "CityName1,CityName2"

# Reddit (embed from local files only, no API calls)
python -m rag.ingest_reddit --local

# Google Places (requires GOOGLE_PLACES_API_KEY)
python -m rag.ingest_google_places
```

### 4. Run the app

```bash
streamlit run app.py --server.runOnSave true
```

---

## Target Cities

52 destinations across Europe and North Africa:

> Lisbon, Porto, Madrid, Seville, Granada, Valencia, Málaga, Bilbao,
> Paris, Nice, Lyon, Bordeaux, Marseille,
> Rome, Florence, Venice, Naples, Milan, Palermo,
> Amsterdam, Brussels, Vienna, Prague, Budapest, Kraków,
> Zurich, Geneva, Edinburgh, Dublin,
> Athens, Thessaloniki, Dubrovnik, Split, Kotor, Sarajevo, Ljubljana,
> Santorini, Mykonos, Rhodes, Corfu, Crete,
> Malta, Valletta,
> Marrakech, Casablanca, Fez, Tangier,
> Cairo, Alexandria, Luxor, Tunis, Algiers

---

## Roadmap

- [ ] Foursquare API integration for hyper-local tips
- [ ] User preference memory across sessions
- [ ] PDF export of travel guides
- [ ] Weather / best season display per city
- [ ] Flight search links (Skyscanner / Google Flights)

---

## Notes

- `data/` is managed with Git LFS. Run `git lfs pull` after cloning.
- Gemini 2.5 Flash Lite is used with JSON mode (`response_mime_type="application/json"`) for reliable structured output.
- Reddit data is fetched via Pullpush (community Reddit archive) — no Reddit API key required.
- Google Places API has a $200/month free credit — more than enough for prototyping.
- Geocoding uses Nominatim (free) with Google Places API as fallback for better coverage.
- Photos are sourced from Google Places Photos API with 24-hour caching.

---
---

# 🌍 Mindful Tourism App — Group E（日本語）

---

## 概要

このプロトタイプは、ユーザーが「マインドフル」に旅先を選ぶことをサポートします。2つのステージで動作します：

- **Stage 1** — ユーザーが自然言語で旅のリクエストを入力。Gemini 2.5 Flashが52都市のリストから4件の候補を推薦し、写真・バイブタグ・フライト時間・費用目安を表示します。
- **Stage 2** — ユーザーが都市を選択。3つのデータソース（WikiVoyage・Reddit・Google Places）からRAGパイプラインが情報を検索し、5つの並列LLM呼び出しで以下を生成します：
  - **都市情報** — 言語・通貨・タイムゾーン・気候を一目で表示
  - **おすすめスポット** — ユーザーの興味に合わせた場所提案（写真・インタラクティブマップ・「Mindful Moment」付き）
  - **ホテル情報** — Google Placesレビューから抽出（星レーティング・Booking.comリンク付き）
  - **ローカルエチケット** — ポジティブで具体的なインサイダー tips
  - **マインドフルペーシング** — 6カテゴリ（時間帯・地元接点・移動・何もしない・地元感覚・サステナビリティ）の8つの tips

---

## 技術スタック

| レイヤー | 技術 |
|--------|------|
| フロントエンド | Streamlit（サイドバー、2カラムカード、インタラクティブマップ） |
| LLM | Google Gemini 2.5 Flash Lite（JSONモード、5並列呼び出し） |
| 埋め込みモデル | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| ベクターDB | ChromaDB（永続化、3コレクション） |
| RAG | LangChain + LangChain-Chroma |
| 地図 | Folium + OpenStreetMap（Nominatim + Google Places ジオコーディング） |
| 写真 | Google Places Photos API |
| データソース | WikiVoyage API、Reddit（Pullpush API）、Google Places API（New） |

---

## 主な機能

- **バイブタグ** — Stage 1 の都市カードに絵文字付きタグ（例：「🍷 Wine marathon vineyards」）を表示。Stage 2 でもタグに対応した場所を推薦。
- **Mindful Moment** — 各スポットに「訪れる」だけでなく「体験する」ための具体的な提案を付与。
- **統合マップ** — おすすめスポット（番号付き）とホテル（カテゴリ色分け）を1つのインタラクティブマップに表示。
- **カテゴリ別ペーシング** — tips を6カテゴリ（⏰ 時間帯、🤝 地元接点、🚶 移動、☕ 何もしない、📍 地元感覚、🌱 サステナビリティ）で色分け表示。
- **Google Places 写真** — 都市画像・スポット画像ともに Google Places から取得し、カバレッジを最大化。

---

## セットアップ手順

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. APIトークンの設定

```bash
cp .env.example .env
```

`.env` を編集：

```
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_PLACES_API_KEY=AIza_xxxxxxxxxxxxxxxxxxxxxxxx
```

- HuggingFace トークン（埋め込みモデル用）: https://huggingface.co/settings/tokens
- Gemini API キー: https://aistudio.google.com/apikey
- Google Places API キー: https://console.cloud.google.com/

### 3. データのセットアップ

`data/` ディレクトリは Git LFS で管理されているため、**クローン後に ingest は不要**です：

```bash
git lfs pull
```

### 4. アプリの起動

```bash
streamlit run app.py --server.runOnSave true
```

---

## 対応都市

ヨーロッパ・北アフリカの52都市：

> Lisbon, Porto, Madrid, Seville, Granada, Valencia, Málaga, Bilbao,
> Paris, Nice, Lyon, Bordeaux, Marseille,
> Rome, Florence, Venice, Naples, Milan, Palermo,
> Amsterdam, Brussels, Vienna, Prague, Budapest, Kraków,
> Zurich, Geneva, Edinburgh, Dublin,
> Athens, Thessaloniki, Dubrovnik, Split, Kotor, Sarajevo, Ljubljana,
> Santorini, Mykonos, Rhodes, Corfu, Crete,
> Malta, Valletta,
> Marrakech, Casablanca, Fez, Tangier,
> Cairo, Alexandria, Luxor, Tunis, Algiers

---

## 注意事項

- `data/` は Git LFS で管理されています。クローン後に `git lfs pull` を実行してください。
- Gemini 2.5 Flash Lite を JSON モードで使用し、安定した構造化出力を実現しています。
- Reddit データは Pullpush（コミュニティ運営アーカイブ）経由で取得。API キー不要。
- Google Places API は月 $200 の無料クレジットあり。
- 写真は Google Places Photos API から取得し、24時間キャッシュ。
