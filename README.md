# 🌍 Mindful Tourism App — Group E

---

## Overview

This prototype helps users discover travel destinations mindfully. It works in two stages:

- **Stage 1** — The user describes a trip in natural language. Gemini 3 Flash recommends 4 destinations from a curated list of 52 cities, with estimated flight times, costs, city images, and tailored reasons.
- **Stage 2** — The user selects a city. A RAG pipeline retrieves local knowledge from three data sources (WikiVoyage, Reddit, Google Places) via ChromaDB, and 4 parallel LLM calls generate:
  - **Recommended Places** — tailored to the user's interests, plotted on an interactive Folium map
  - **Where to Stay** — hotels extracted from Google Places reviews, with Booking.com links and map markers
  - **Local Etiquette** — specific, surprising insider tips written in a witty tone (not generic advice)
  - **Mindful Pacing** — practical tips covering timing, local connections, scenic routes, "doing nothing" spots, and local pace of life

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit (sidebar layout, 2-column cards, interactive maps) |
| LLM | Google Gemini 3 Flash Preview (JSON mode, 4 parallel calls) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | ChromaDB (persistent, 3 collections) |
| RAG | LangChain + LangChain-Chroma |
| Maps | Folium + OpenStreetMap (with Nominatim + Google Places geocoding fallback) |
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
  🙏 Local Etiquette     → wikivoyage + reddit_tips (tourist mistakes, scams)
  🌿 Mindful Pacing      → wikivoyage + reddit_tips (timing, quiet spots, local pace)
```

---

## LLM Architecture

Stage 2 uses 4 parallel Gemini calls instead of a single monolithic call, improving both quality and speed:

```
User selects a city
        ↓
  4 RAG queries (section-specific)
        ↓
  ┌─────────────────────────────────────────────────────┐
  │ Parallel LLM calls (ThreadPoolExecutor)             │
  │                                                     │
  │  _generate_places()    → recommended_places         │
  │  _generate_hotels()    → hotels                     │
  │  _generate_etiquette() → etiquette                  │
  │  _generate_pacing()    → pacing_advice              │
  └─────────────────────────────────────────────────────┘
        ↓
  Combined result → Streamlit UI
```

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
│   └── client.py                 # Gemini API (Stage 1 + 4 parallel Stage 2 calls)
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
- Gemini 3 Flash Preview is used with JSON mode (`response_mime_type="application/json"`) for reliable structured output.
- Reddit data is fetched via Pullpush (community Reddit archive) — no Reddit API key required.
- Google Places API has a $200/month free credit — more than enough for prototyping.
- Geocoding uses Nominatim (free) with Google Places API as fallback for better coverage.

---
---

# 🌍 Mindful Tourism App — Group E（日本語）

---

## 概要

このプロトタイプは、ユーザーが「マインドフル」に旅先を選ぶことをサポートします。2つのステージで動作します：

- **Stage 1** — ユーザーが自然言語で旅のリクエストを入力。Gemini 3 Flashが52都市のリストから4件の候補を推薦し、フライト時間・費用目安・都市画像を表示します。
- **Stage 2** — ユーザーが都市を選択。3つのデータソース（WikiVoyage・Reddit・Google Places）からRAGパイプラインが情報を検索し、4つの並列LLM呼び出しで以下を生成します：
  - **おすすめスポット** — ユーザーの興味に合わせた場所提案（Foliumインタラクティブマップ付き）
  - **ホテル情報** — Google Placesレビューから抽出（Booking.comリンク・マップマーカー付き）
  - **ローカルエチケット** — ウィットに富んだ都市固有のインサイダー tips
  - **マインドフルペーシング** — 時間帯・地元との接点・景色の良いルート・「何もしない」スポット・地元の時間感覚

---

## 技術スタック

| レイヤー | 技術 |
|--------|------|
| フロントエンド | Streamlit（サイドバー、2カラムカード、インタラクティブマップ） |
| LLM | Google Gemini 3 Flash Preview（JSONモード、4並列呼び出し） |
| 埋め込みモデル | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| ベクターDB | ChromaDB（永続化、3コレクション） |
| RAG | LangChain + LangChain-Chroma |
| 地図 | Folium + OpenStreetMap（Nominatim + Google Places ジオコーディング） |
| データソース | WikiVoyage API、Reddit（Pullpush API）、Google Places API（New） |

---

## RAG アーキテクチャ

ガイドの各セクションごとに異なる ChromaDB コレクションを検索し、最適なコンテキストを取得します：

```
ChromaDB
├── wikivoyage       ← 都市の概要・基本情報（52都市）
├── reddit_tips      ← ニッチなクチコミ・裏知識・隠れた名所
└── google_places    ← ホテル・レストランのレビュー・評価

セクション別検索:
  📍 おすすめスポット      → wikivoyage + reddit_tips
  🏨 ホテル情報           → wikivoyage + google_places
  🙏 ローカルエチケット    → wikivoyage + reddit_tips
  🌿 マインドフルペーシング → wikivoyage + reddit_tips
```

---

## LLM アーキテクチャ

Stage 2 では単一の巨大な呼び出しではなく、4つの並列 Gemini 呼び出しを使用し、品質と速度を両立：

```
ユーザーが都市を選択
        ↓
  4つの RAG クエリ（セクション別）
        ↓
  ┌─────────────────────────────────────────────────────┐
  │ 並列 LLM 呼び出し（ThreadPoolExecutor）              │
  │                                                     │
  │  _generate_places()    → おすすめスポット             │
  │  _generate_hotels()    → ホテル情報                  │
  │  _generate_etiquette() → ローカルエチケット           │
  │  _generate_pacing()    → マインドフルペーシング        │
  └─────────────────────────────────────────────────────┘
        ↓
  統合結果 → Streamlit UI
```

---

## ファイル構成

```
mindful-tourism-group-E/
├── app.py                        # Streamlit メインアプリ（サイドバー + 2ステージ UI）
├── requirements.txt
├── .env.example                  # APIトークンのテンプレート
├── .gitignore
├── .gitattributes                # Git LFS 追跡ルール
│
├── llm/
│   └── client.py                 # Gemini API（Stage 1 + 4並列 Stage 2）
│
├── rag/
│   ├── cities.py                 # 共有定数・都市リスト・コレクション名
│   ├── ingest_wikivoyage.py      # WikiVoyage API → ChromaDB
│   ├── ingest_reddit.py          # Reddit（Pullpush API）→ ChromaDB
│   ├── ingest_google_places.py   # Google Places API → ChromaDB
│   ├── ingest_crawler.py         # Webクローリングスタブ（将来用）
│   └── retriever.py              # セクション別マルチコレクション検索
│
└── data/                         # Git LFS で管理
    ├── wikivoyage/               # WikiVoyage 生テキスト
    ├── reddit/                   # Reddit 投稿・コメント
    ├── google_places/            # Google Places レビュー
    └── chroma/                   # ChromaDB ベクターストア（3コレクション）
```

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

データを手動で再構築・更新する場合：

```bash
# WikiVoyage（全都市）
python -m rag.ingest_wikivoyage

# Reddit（全都市、Pullpush API経由 — APIキー不要）
python -m rag.ingest_reddit

# Reddit（特定都市だけ追加）
python -m rag.ingest_reddit --add "都市名1,都市名2"

# Reddit（ローカルファイルからベクトライズのみ、API呼び出しなし）
python -m rag.ingest_reddit --local

# Google Places（GOOGLE_PLACES_API_KEY が必要）
python -m rag.ingest_google_places
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

## ロードマップ

- [ ] Foursquare API 連携（ハイパーローカル tips）
- [ ] セッションをまたいだユーザー好み記憶機能
- [ ] 旅行ガイドの PDF エクスポート
- [ ] 天気・ベストシーズン表示
- [ ] フライト検索リンク（Skyscanner / Google Flights）

---

## 注意事項

- `data/` は Git LFS で管理されています。クローン後に `git lfs pull` を実行してください。
- Gemini 3 Flash Preview を JSON モード（`response_mime_type="application/json"`）で使用し、安定した構造化出力を実現しています。
- Reddit データは Pullpush（コミュニティ運営の Reddit アーカイブ）経由で取得しています。Reddit API キーは不要です。
- Google Places API は月 $200 の無料クレジットがあり、プロトタイプ用途では十分です。
- ジオコーディングは Nominatim（無料）を使用し、Google Places API をフォールバックとして併用しています。
