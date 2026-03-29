# 🌍 Mindful Tourism App — Group E

---

## Overview

This prototype helps users discover travel destinations mindfully. It works in two stages:

- **Stage 1** — The user describes a trip in natural language. An LLM recommends 3–4 destinations from a curated city list, along with estimated flight times and costs.
- **Stage 2** — The user selects a city. A RAG pipeline retrieves local knowledge from WikiVoyage (via ChromaDB) and the LLM generates recommended places, hotel suggestions, local etiquette tips, and mindful pacing advice.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | HuggingFace Inference API (`Qwen/Qwen2.5-7B-Instruct`) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | ChromaDB (persistent) |
| RAG | LangChain + LangChain-Chroma |
| Data source (Phase 1) | WikiVoyage API |
| Data source (Phase 2) | Web crawling — stub implemented, to be added |

---

## File Structure

```
mindful-tourism-group-E/
├── app.py                      # Streamlit main app (Stage 1 + Stage 2 UI)
├── requirements.txt
├── .env.example                # API token template
├── .gitignore
│
├── llm/
│   └── client.py               # HuggingFace Inference API calls (Stage 1 & 2)
│
├── rag/
│   ├── cities.py               # Shared constants: city list, EMBED_MODEL, CHROMA_DIR
│   ├── ingest_wikivoyage.py    # WikiVoyage scraping → ChromaDB ingest
│   ├── ingest_crawler.py       # Phase 2 web crawling stub
│   └── retriever.py            # RAG retrieval logic (multi-collection)
│
└── data/                       # Auto-generated (excluded from git)
    ├── wikivoyage/             # Raw WikiVoyage text files
    ├── crawled/                # Phase 2 crawled data (empty)
    └── chroma/                 # ChromaDB vector store
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API token

```bash
cp .env.example .env
```

Edit `.env` and add your HuggingFace token:

```
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get your token at: https://huggingface.co/settings/tokens

### 3. Ingest WikiVoyage data

Run once to populate ChromaDB with ~50 European and North African cities:

```bash
python -m rag.ingest_wikivoyage
```

To add specific cities without rebuilding the entire collection:

```bash
python -m rag.ingest_wikivoyage --add "CityName1,CityName2"
```

### 4. Run the app

```bash
streamlit run app.py --server.runOnSave true
```

`--server.runOnSave true` enables hot-reload on file save.

---

## Target Cities

~53 destinations across Europe and North Africa:

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

- **Phase 2** — Web crawling ingest (`rag/ingest_crawler.py`) to enrich hotel and attraction data beyond WikiVoyage
- **Phase 2** — Booking.com / hotel API integration for real-time availability and pricing
- **Phase 2** — User preference memory across sessions

---

## Notes

- `data/` directory is excluded from git. Run `ingest_wikivoyage.py` after cloning to regenerate it.
- The HuggingFace free tier has rate limits. Requests may occasionally be slow or time out.

---
---

# 🌍 Mindful Tourism App — Group E（日本語）

---

## 概要

このプロトタイプは、ユーザーが「マインドフル」に旅先を選ぶことをサポートします。2つのステージで動作します：

- **Stage 1** — ユーザーが自然言語で旅のリクエストを入力。LLMがキュレーションされた都市リストから3〜4件の候補を推薦し、フライト時間・費用目安を表示します。
- **Stage 2** — ユーザーが都市を選択。RAGパイプラインがChromaDB経由でWikiVoyageのローカル情報を検索し、LLMがおすすめスポット・ホテル情報・地元エチケット・ペーシングアドバイスを生成します。

---

## 技術スタック

| レイヤー | 技術 |
|--------|------|
| フロントエンド | Streamlit |
| LLM | HuggingFace Inference API（`Qwen/Qwen2.5-7B-Instruct`） |
| 埋め込みモデル | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| ベクターDB | ChromaDB（永続化） |
| RAG | LangChain + LangChain-Chroma |
| データソース（Phase 1） | WikiVoyage API |
| データソース（Phase 2） | Webクローリング（スタブ実装済み、追加予定） |

---

## ファイル構成

```
mindful-tourism-group-E/
├── app.py                      # Streamlit メインアプリ（Stage 1 + Stage 2 UI）
├── requirements.txt
├── .env.example                # APIトークンのテンプレート
├── .gitignore
│
├── llm/
│   └── client.py               # HuggingFace Inference API 呼び出し（Stage 1・2共通）
│
├── rag/
│   ├── cities.py               # 共有定数：都市リスト・EMBED_MODEL・CHROMA_DIR
│   ├── ingest_wikivoyage.py    # WikiVoyageスクレイピング → ChromaDB投入
│   ├── ingest_crawler.py       # Phase 2 Webクローリングスタブ
│   └── retriever.py            # RAG検索ロジック（マルチコレクション対応）
│
└── data/                       # 自動生成（git管理外）
    ├── wikivoyage/             # WikiVoyage生テキスト
    ├── crawled/                # Phase 2用クロールデータ（空）
    └── chroma/                 # ChromaDB ベクターストア
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

`.env` を開いてHuggingFaceトークンを追記してください：

```
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

トークンの取得はこちら：https://huggingface.co/settings/tokens

### 3. WikiVoyageデータの投入

初回のみ実行します（ヨーロッパ・北アフリカ約50都市を ChromaDB に投入）：

```bash
python -m rag.ingest_wikivoyage
```

コレクションを再構築せずに特定の都市だけ追加する場合：

```bash
python -m rag.ingest_wikivoyage --add "都市名1,都市名2"
```

### 4. アプリの起動

```bash
streamlit run app.py --server.runOnSave true
```

`--server.runOnSave true` を付けると、ファイル保存時に自動でリロードされます。

---

## 対応都市

ヨーロッパ・北アフリカの約53都市：

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

- **Phase 2** — Webクローリングingest（`rag/ingest_crawler.py`）でWikiVoyage以外のホテル・観光情報を拡充
- **Phase 2** — Booking.com / ホテルAPI連携によるリアルタイム空き状況・料金表示
- **Phase 2** — セッションをまたいだユーザー好み記憶機能

---

## 注意事項

- `data/` ディレクトリはgit管理外です。クローン後に `ingest_wikivoyage.py` を実行して再生成してください。
- HuggingFace の無料枠はレートリミットがあります。まれにリクエストが遅延・タイムアウトする場合があります。
