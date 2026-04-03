import os, pickle, re, textwrap, warnings
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Candidate Search", page_icon="🔍", layout="wide")

@st.cache_resource(show_spinner="Loading AI models…")
def load_models():
    bi = SentenceTransformer("all-MiniLM-L6-v2")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return bi, ce

@st.cache_resource(show_spinner="Loading candidate index…")
def load_engine(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def rrf(lex, sem, k=60):
    s = {}
    for r, i in enumerate(lex): s[i] = s.get(i, 0) + 1 / (k + r + 1)
    for r, i in enumerate(sem): s[i] = s.get(i, 0) + 1 / (k + r + 1)
    return sorted(s, key=s.get, reverse=True)

def top_sentences(bi_enc, q_emb, text, n=2):
    sents = [s.strip() for s in re.split(r"(?<=[.!?]) +|\n+", text) if len(s.strip()) > 25]
    if not sents:
        return text[:250]
    embs = bi_enc.encode(sents, normalize_embeddings=True)
    sc   = embs @ q_emb[0]
    top  = sorted(np.argsort(sc)[-n:])
    return " … ".join(sents[i] for i in top)

def search(query, data, bi, ce, top_k=20, pool=100):
    corpus, embeddings, bm25, df = data["corpus"], data["embeddings"], data["bm25"], data["df"]
    tokens    = re.sub(r"[^a-z0-9 ]", " ", query.lower()).split()
    lex_sc    = np.array(bm25.get_scores(tokens))
    lex_ranks = np.argsort(lex_sc)[::-1]
    q_emb     = bi.encode([query], normalize_embeddings=True)
    sem_sc    = embeddings @ q_emb[0]
    sem_ranks = np.argsort(sem_sc)[::-1]
    fused     = rrf(lex_ranks.tolist(), sem_ranks.tolist())[:pool]
    pairs     = [[query, corpus[i]] for i in fused]
    ce_sc     = ce.predict(pairs, show_progress_bar=False)
    ranked    = sorted(zip(fused, ce_sc), key=lambda x: x[1], reverse=True)
    results   = []
    for rank, (idx, score) in enumerate(ranked[:top_k], 1):
        conf    = round(float(1 / (1 + np.exp(-score))) * 100, 1)
        lex_pos = int(np.where(lex_ranks == idx)[0][0]) + 1
        sem_pos = int(np.where(sem_ranks == idx)[0][0]) + 1
        ev      = top_sentences(bi, q_emb, corpus[idx])
        results.append({"rank": rank, "name": df.iloc[idx]["candidate_name"],
                        "conf": conf, "lex": lex_pos, "sem": sem_pos,
                        "evidence": ev, "resume": corpus[idx]})
    return results

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🔍 Candidate Search Engine")
st.caption("3-stage hybrid pipeline · BM25 → Bi-Encoder → Cross-Encoder reranker")

with st.sidebar:
    st.header("⚙️ Settings")
    pkl   = st.text_input("Engine file", value="resume_engine.pkl")
    top_k = st.slider("Results to show", 5, 20, 10)
    st.divider()
    st.markdown("**Example queries**")
    examples = [
        "senior backend engineer Python Go Bangalore",
        "ML engineer production deployment not notebooks",
        "React Native developer consumer tech startup",
        "founding engineer generalist 2-5 yrs startup",
        "data engineer Kafka event streaming at scale",
        "fresh grad strong fundamentals side projects",
        "payments infra engineer fintech experience",
        "fullstack actually good at backend not just React",
        "devops understands networking not yaml pushers",
        "senior person 3+ years same company",
    ]
    for q in examples:
        if st.button(q, use_container_width=True, key=q[:25]):
            st.session_state["q"] = q
    st.divider()
    st.markdown("**How it works**\n1. 🔤 BM25 — keyword match\n2. 🧠 Bi-Encoder — semantic\n3. 🔀 RRF — fuses both lists\n4. ✅ Cross-Encoder — deep rerank")

if not os.path.exists(pkl):
    st.warning(f"⚠️ `{pkl}` not found.")
    up = st.file_uploader("Upload resume_engine.pkl", type=["pkl"])
    if up:
        open("resume_engine.pkl", "wb").write(up.read())
        st.success("✅ Uploaded! Refresh the page.")
    st.stop()

bi, ce = load_models()
data   = load_engine(pkl)
st.sidebar.success(f"✅ {len(data['corpus'])} candidates loaded")

query = st.text_input("Search query", value=st.session_state.get("q", ""),
    placeholder="e.g. senior backend engineer, Python, 4+ years, Bangalore")

if st.button("🔍 Search", type="primary", use_container_width=True) and query.strip():
    st.session_state["q"] = query
    with st.spinner("Running 3-stage search pipeline…"):
        results = search(query, data, bi, ce, top_k)
    st.subheader(f"Top {len(results)} results for: *{query}*")
    st.divider()
    for r in results:
        conf  = r["conf"]
        color = "#2ecc71" if conf >= 65 else ("#f39c12" if conf >= 45 else "#e74c3c")
        label = "Strong" if conf >= 65 else ("Moderate" if conf >= 45 else "Weak")
        bar   = "█" * int(conf / 5) + "░" * (20 - int(conf / 5))
        with st.expander(f"#{r['rank']:02d}  {r['name']}  —  {label} match ({conf}%)",
                         expanded=(r["rank"] <= 3)):
            c1, c2 = st.columns([1, 3])
            with c1:
                st.metric("Confidence", f"{conf}%")
                st.markdown(f"<b style='color:{color}'>{label} match</b>", unsafe_allow_html=True)
                st.code(bar, language=None)
                st.caption(f"🔤 Keyword rank: #{r['lex']}")
                st.caption(f"🧠 Semantic rank: #{r['sem']}")
            with c2:
                st.info(f"**Why this candidate matched:**\n\n{r['evidence']}")
                st.text_area("Resume excerpt", textwrap.shorten(r["resume"], 600, placeholder="…"),
                             height=130, disabled=True, key=f"res_{r['rank']}")
                with st.expander("📄 View full resume"):
                    st.text(r["resume"])
