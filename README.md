# 🌍 Mindful Tourism App — Group E

---

## Overview

This prototype helps users discover travel destinations mindfully. It works in two stages:

- **Stage 1** — The user describes a trip in natural language. An LLM recommends 3–4 destinations from a curated city list, with estimated flight times, costs, and city images.
- **Stage 2** — The user selects a city. A RAG pipeline retrieves local knowledge from three data sources (WikiVoyage, Reddit, Google Places) via ChromaDB, and the LLM generates:
  - **Recommended Places** — tailored to the user's interests, with embedded Google Maps
  - **Where to Stay** — hotels with reviews from Google Places, with Booking.com links
  - **Local Etiquette** — specific, surprising insider tips (not generic advice)
  - **Mindful Pacing** — practical timing, transport, and "doing nothing" tips

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit (sidebar layout, 2-column cards) |
| LLM | HuggingFace Inference API (`Qwen/Qwen2.5-7B-Instruct`) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | ChromaDB (persistent, 3 collections) |
| RAG | LangChain + LangChain-Chroma |
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
  🌿 Mindful Pacing      → wikivoyage + reddit_tips (timing, transport, quiet spots)
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
│   └── client.py                 # HuggingFace Inference API (Stage 1 & 2)
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
    └── chroma/                   # ChromaDB vector store
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
GOOGLE_PLACES_API_KEY=AIza_xxxxxxxxxxxxxxxxxxxxxxxx
```

- HuggingFace token: https://huggingface.co/settings/tokens
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
- The HuggingFace free tier has rate limits. Requests may occasionally be slow or time out.
- Reddit data is fetched via Pullpush (community Reddit archive) — no Reddit API key required.
- Google Places API has a $200/month free credit — more than enough for prototyping.

---
---

# 🌍 Mindful Tourism App — Group E（日本語）

---

## 概要

このプロトタイプは、ユーザーが「マインドフル」に旅先を選ぶことをサポートします。2つのステージで動作します：

- **Stage 1** — ユーザーが自然言語で旅のリクエストを入力。LLMがキュレーションされた都市リストから3〜4件の候補を推薦し、フライト時間・費用目安・都市画像を表示します。
- **Stage 2** — ユーザーが都市を選択。3つのデータソース（WikiVoyage・Reddit・Google Places）からRAGパイプラインがChromaDB経由で情報を検索し、LLMが以下を生成します：
  - **おすすめスポット** — ユーザーの興味に合わせた場所提案（Google Maps埋め込み付き）
  - **ホテル情報** — Google Placesのレビュー付き（Booking.comリンク付き）
  - **ローカルエチケット** — 観光客がやりがちなミス、具体的で驚きのある tips
  - **マインドフルペーシング** — 時間帯・移動・「何もしない」提案など実用的な tips

---

## 技術スタック

| レイヤー | 技術 |
|--------|------|
| フロントエンド | Streamlit（サイドバー、2カラムカード） |
| LLM | HuggingFace Inference API（`Qwen/Qwen2.5-7B-Instruct`） |
| 埋め込みモデル | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| ベクターDB | ChromaDB（永続化、3コレクション） |
| RAG | LangChain + LangChain-Chroma |
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
│   └── client.py                 # HuggingFace Inference API 呼び出し
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
    └── chroma/                   # ChromaDB ベクターストア
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
GOOGLE_PLACES_API_KEY=AIza_xxxxxxxxxxxxxxxxxxxxxxxx
```

- HuggingFace トークン: https://huggingface.co/settings/tokens
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
- HuggingFace の無料枠はレートリミットがあります。まれにリクエストが遅延・タイムアウトする場合があります。
- Reddit データは Pullpush（コミュニティ運営の Reddit アーカイブ）経由で取得しています。Reddit API キーは不要です。
- Google Places API は月 $200 の無料クレジットがあり、プロトタイプ用途では十分です。
