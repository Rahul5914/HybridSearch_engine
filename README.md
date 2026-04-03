# HybridSearch_engine
# 🔍 Hybrid Candidate Search Engine

> **Live Demo:** [https://hybridsearchengine-7esyyx45j4nzmyyfl5jvqf.streamlit.app](https://hybridsearchengine-7esyyx45j4nzmyyfl5jvqf.streamlit.app)

A production-grade semantic search system built for the **GrowthVenture Backend Engineering Intern Take-Home Assignment**. Type any natural language query and get back the top 20 most relevant candidates from a pool of **1,742 resumes** — ranked by relevance with a detailed breakdown of *why* each result was returned.

---

## ✨ What It Does

- Accepts **any natural language query** — precise or vague
- Returns **top 20 candidates** ranked by a 3-stage hybrid pipeline
- For every result shows:
  - **Confidence score** (cross-encoder sigmoid output as %)
  - **Keyword rank** — where BM25 placed this candidate
  - **Semantic rank** — where the bi-encoder placed this candidate
  - **Extracted evidence** — the 2 most relevant sentences pulled from their resume

---

## 🏗️ System Architecture

```
Natural Language Query
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1a — BM25  (Lexical Retrieval)               │
│  Tokenises query + resumes. Exact keyword matching  │
│  via Okapi BM25 with TF-IDF + length normalisation. │
└──────────────────────┬──────────────────────────────┘
                       │  ranked list (all 1,742)
┌─────────────────────────────────────────────────────┐
│  Stage 1b — Bi-Encoder  (Semantic Retrieval)        │
│  all-MiniLM-L6-v2 encodes query → cosine similarity │
│  against pre-built 384-dim resume embeddings.       │
└──────────────────────┬──────────────────────────────┘
                       │  ranked list (all 1,742)
┌─────────────────────────────────────────────────────┐
│  Stage 2 — Reciprocal Rank Fusion                   │
│  Merges both lists. Score = Σ 1/(k + rank), k=60.  │
│  Produces a single fused top-100 candidate pool.   │
└──────────────────────┬──────────────────────────────┘
                       │  top 100 candidates
┌─────────────────────────────────────────────────────┐
│  Stage 3 — Cross-Encoder Reranker                   │
│  ms-marco-MiniLM-L-6-v2 reads each (query, resume) │
│  pair jointly. Full bidirectional attention.        │
│  Final confidence score via sigmoid transform.      │
└──────────────────────┬──────────────────────────────┘
                       │
                 Top 20 Results
        (score breakdown + extracted evidence)
```

---

## 🛠️ Tech Stack

| Component | Library / Model |
|-----------|----------------|
| Semantic embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` |
| Reranker | `sentence-transformers` · `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Lexical retrieval | `rank_bm25` · Okapi BM25 |
| Rank fusion | Reciprocal Rank Fusion (k=60) |
| Web UI | `streamlit` |
| Index storage | `pickle` (15.6 MB) |
| Language | Python 3.10+ |

---

## 📁 Repository Structure

```
HybridSearch_engine/
├── app.py                        # Streamlit web app (main entry point)
├── requirements.txt              # Python dependencies
├── resume_engine.pkl             # Pre-built search index (1,742 candidates)
├── HybridSearch_Final.ipynb      # Colab notebook to rebuild the index
├── Analysis_Document_Final.docx  # Technical analysis document (2 pages)
└── README.md                     # This file
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Rahul5914/HybridSearch_engine.git
cd HybridSearch_engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) — the app loads the pre-built index automatically.

> **Note:** `resume_engine.pkl` must be in the same directory as `app.py`. It is already included in this repo.

---

## 📓 Rebuild the Index from Scratch (Optional)

Only needed if you want to re-index with new candidate data:

1. Open `HybridSearch_Final.ipynb` in [Google Colab](https://colab.research.google.com)
2. Upload `Candidates_and_Jobs.xlsx` when prompted
3. Run all cells — takes ~5 minutes on CPU, ~1 minute on GPU
4. Download the generated `resume_engine.pkl`
5. Replace the existing file in this repo

---

## 💡 Example Queries

The system handles queries ranging from highly precise to deliberately vague:

| Type | Query |
|------|-------|
| Precise | `senior backend engineer, 4+ years, Python and Go, Bangalore` |
| Conceptual | `someone who can own our payments infra, has worked at a fintech before` |
| Vague | `founding engineer type generalist, 2-5 yrs, startup experience, wears multiple hats` |
| Negative intent | `ML engineer who's done production deployment, not just notebooks` |
| Domain-specific | `data engineer who's worked with event streaming at scale, preferably Kafka` |
| Stack-qualified | `fullstack but actually good at backend, not just a React dev who can write an API` |

---

## 📊 Why a 3-Stage Hybrid?

| Approach | Strength | Weakness |
|---------|----------|----------|
| BM25 only | Fast, exact keyword hits | Fails on vague / conceptual queries |
| Bi-Encoder only | Understands meaning and intent | Misses precise keyword requirements |
| Cross-Encoder on all 1,742 | Most accurate | ~90s per query — unusable |
| **This system** | **Best of all three** | **~4s per query** |

- **BM25** catches `Kafka`, `Bangalore`, `Go` precisely where semantic models struggle with rare nouns
- **Bi-Encoder** understands `"payments infra"` even when the resume says `"transaction processing at a fintech startup"`
- **RRF** fuses both ranked lists without discarding either signal — candidates strong in both rank highest
- **Cross-Encoder** reads each shortlisted (query, resume) pair together with full attention — like a human reviewer — for the final ranking

---

## ⚠️ Known Limitations

| Failure | Why | Planned Fix |
|--------|-----|-------------|
| Negation (`"not a job hopper"`) | Neither BM25 nor MiniLM understand exclusion | LLM-based query rewriting to extract negative constraints |
| Implicit tenure signals | Employment duration requires date-range parsing, not text matching | Pre-compute `longest_tenure` and `num_companies` at index time |
| Company-concept queries (`"like a Razorpay engineer"`) | MiniLM doesn't know what skills a specific company implies | LLM query expansion before embedding |

---

## 📄 Analysis Document

`Analysis_Document_Final.docx` covers:

- Full system design rationale and alternatives considered
- Sample queries with top-5 results and per-rank explanations
- Detailed failure case analysis with proposed fixes
- Roadmap for what to build next with a full week

---

*Built for the GrowthVenture Backend Engineering Intern Take-Home Assignment.*
