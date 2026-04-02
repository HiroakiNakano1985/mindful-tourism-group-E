# 🌍 Mindful Tourism App — Group E

---
http://34.247.194.233:8501/  

## The Problem

Travel discovery has become strangely repetitive.

Search any destination and you'll see the same "Top 10 things to do." The same restaurants. The same hotel lists. ChatGPT gives you the same generic answers. Guidebooks are outdated the day they're printed.

But the best travel advice has always come from **a friend who's actually been there** — someone who tells you *"skip the main square, walk two blocks east, there's a family-run place where the owner will talk your ear off about local wine."*

That kind of knowledge exists. It's scattered across Reddit threads, buried in Google reviews, hidden in local travel forums. But no one has time to read 500 Reddit posts before a trip.

## The Solution

**This app aggregates real traveller experiences and delivers them as a personalised, mindful travel guide.**

Instead of generic "Top 10" lists, it provides:

- 🍷 **"The winery requires reservations — max 2 per day or you'll be too drunk to enjoy them"** (from Reddit)
- 🏨 **"Breakfast buffet is incredible, ask for a river-view room on 6F"** (from Google Reviews)
- ☕ **"Sit in the back courtyard for 10 minutes without your phone — you'll notice the art on the ceiling that everyone misses"** (Mindful Moment)

The key difference from ChatGPT: **every recommendation is backed by real data** — Reddit posts from travellers who've been there, and Google Places reviews from actual guests. Not hallucinated, not generic, not the same answer everyone else gets.

> **In one sentence: "The advice a well-travelled friend would give you, powered by real data."**

---

## How It Works

**Stage 1 — Discover** — Describe your ideal trip in natural language. The app recommends 4 destinations with vibe tags (e.g. "🌙 Dawn-till-noon raves", "🍷 Wine marathon vineyards") so you can feel the destination before choosing it.

**Stage 2 — Explore** — Select a city and get a personalised mindful guide:

| Section | What you get | Data source |
|---------|-------------|-------------|
| 📍 **Recommended Places** | 5-8 spots tailored to your interests, each with a photo and a "Mindful Moment" | WikiVoyage + Reddit |
| 🏨 **Where to Stay** | Hotels with real review highlights and Booking.com links | Google Places reviews |
| 🙏 **Local Etiquette** | Specific insider tips, not "respect local customs" | WikiVoyage + Reddit |
| 🌿 **Mindful Pacing** | Timing, local connections, scenic routes, "doing nothing" spots, sustainability | WikiVoyage + Reddit |
| 🗺️ **Interactive Map** | All places and hotels plotted on one map | Nominatim + Google Places |

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

## What Makes This Different

| | Guidebook | ChatGPT | **This App** |
|--|-----------|---------|-------------|
| "Best food in Bordeaux?" | Top 5 Michelin restaurants | Top 5 Michelin restaurants | "The Saturday market in Chartrons has a fish stall where the owner gives free samples" |
| "Hotels in Budapest?" | Star rating + location | Star rating + location | "Corinthia's breakfast buffet is insane. The spa closes early though — go before 5pm" |
| "What to do in Amsterdam?" | Canal tour, Rijksmuseum, Anne Frank | Canal tour, Rijksmuseum, Anne Frank | "Take the free ferry to NDSM at sunrise. Rent an electric boat instead of a canal tour — same view, no crowds" |

### Key design choices:

- **Vibe Tags, not descriptions** — "🌙 Dawn-till-noon raves" tells you more than 3 paragraphs of text
- **Mindful Moments** — Each spot comes with a suggestion for how to *experience* it, not just visit it
- **Categorized Pacing** — Tips grouped by what matters: ⏰ Timing, 🤝 Local Connection, 🚶 Movement, ☕ Doing Nothing, 📍 Local Scale, 🌱 Sustainability
- **Data-backed** — Every recommendation traces back to Reddit posts or Google reviews, not LLM imagination

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

## 問題

旅行の情報収集は、どこを見ても同じ内容の繰り返しです。

どの都市を検索しても「おすすめ観光スポット10選」。同じレストラン。同じホテル。ChatGPT に聞いても同じ一般的な回答。ガイドブックは印刷された日に古くなります。

でも、最高の旅行アドバイスはいつも **「実際に行ったことがある友達」** から聞くもの — *「メイン通りは飛ばして、2ブロック東に歩いて。家族経営の店があって、オーナーが地元ワインについて延々と語ってくれるよ」* みたいな。

そういう知識は存在します。Reddit のスレッドに散らばり、Google レビューに埋もれ、ローカルな旅行フォーラムに隠れています。でも旅行前に500件の Reddit 投稿を読む時間は誰にもありません。

## 解決策

**実際の旅行者の体験を集約し、パーソナライズされたマインドフルな旅行ガイドとして提供するアプリです。**

一般的な「おすすめ10選」ではなく：

- 🍷 **「ワイナリーは予約制。1日2軒が限界、3軒目は酔って味がわからない」**（Reddit より）
- 🏨 **「朝食ビュッフェが最高。6Fのリバーサイドの部屋を頼んで」**（Google レビューより）
- ☕ **「奥の中庭に座って10分間スマホを見ないで — 天井のアートに気づくはず」**（Mindful Moment）

ChatGPT との最大の違い：**すべての推薦が実データに基づいている** — 実際に行った旅行者の Reddit 投稿と、実際の宿泊者の Google レビュー。ハルシネーションなし、一般論なし。

> **一言で言うと：「行ったことのある友達のアドバイスを、実データで実現するアプリ」**

---

## 使い方

**Stage 1 — 発見** — 旅のリクエストを自然言語で入力。4都市がバイブタグ付き（例：「🌙 朝まで続くレイブ」「🍷 ワイン畑マラソン」）で提案されます。

**Stage 2 — 探索** — 都市を選択すると、パーソナライズされたマインドフルガイドが生成されます：

| セクション | 内容 | データソース |
|-----------|------|-------------|
| 📍 **おすすめスポット** | 興味に合わせた5-8箇所、写真と「Mindful Moment」付き | WikiVoyage + Reddit |
| 🏨 **ホテル情報** | 実レビューのハイライトと Booking.com リンク | Google Places レビュー |
| 🙏 **ローカルエチケット** | 「地元文化を尊重しよう」ではない、具体的なインサイダー tips | WikiVoyage + Reddit |
| 🌿 **マインドフルペーシング** | 時間帯・地元接点・景色のいいルート・「何もしない」スポット・サステナビリティ | WikiVoyage + Reddit |
| 🗺️ **インタラクティブマップ** | 全スポットとホテルを1つの地図に表示 | Nominatim + Google Places |

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

## 何が違うのか

| | ガイドブック | ChatGPT | **このアプリ** |
|--|-----------|---------|-------------|
| 「ボルドーで美味しいもの」 | ミシュラン星付き5選 | ミシュラン星付き5選 | 「土曜のシャルトロン市場の魚屋が試食させてくれる」 |
| 「ブダペストのホテル」 | 星の数とロケーション | 星の数とロケーション | 「Corinthia の朝食ビュッフェは最高。スパは17時前に行って」 |
| 「アムステルダムの観光」 | 運河ツアー、国立美術館 | 運河ツアー、国立美術館 | 「日の出に NDSM 行きの無料フェリーに乗って。運河ツアーより電動ボートレンタルのほうが同じ景色で人がいない」 |

### 設計の特徴

- **バイブタグ** — 長い説明文ではなく「🌙 朝まで続くレイブ」で体験を伝える
- **Mindful Moment** — 各スポットに「体験する方法」の提案を付与
- **カテゴリ別ペーシング** — ⏰ 時間帯、🤝 地元接点、🚶 移動、☕ 何もしない、📍 地元感覚、🌱 サステナビリティ
- **データに基づく推薦** — すべての情報が Reddit 投稿か Google レビューに由来。LLM の想像ではない

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
