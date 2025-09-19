# app.py (stable CPU build: persistence + explore + chart + exports)
import os, io, json, csv
from pathlib import Path
import streamlit as st

from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, Settings, Document,
    StorageContext, load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI as LIOpenAI

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

# --- CPU-only (fixes meta-tensor issue) ---
DEVICE = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Paths
PDF_DIR = Path(r"C:\Users\krupa\Desktop\Bootcamp\Generative_Ai\Data")
PDF_DIR.mkdir(parents=True, exist_ok=True)
STORE_DIR = Path("./storage")
EMB_CACHE = PDF_DIR / "embedding_model"
EMB_CACHE.mkdir(parents=True, exist_ok=True)

# Keys
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

PROMPT = PromptTemplate(
    "Use the provided context to answer the question.\n"
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context_str}\n\nQuestion: {query_str}\nAnswer:"
)

st.title("🤖 Home Appliances Chatbot")
st.caption("Upload PDFs → build index → ask → see sources. Persisted index + Explore + chart + exports.")

# Sidebar controls
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox(
    "Groq model (if using GROQ_API_KEY)",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"], index=0
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.2, 0.15, 0.05)
chunk_size_ui = st.sidebar.slider("Chunk size (chars)", 400, 1500, 800, 50)
chunk_overlap_ui = st.sidebar.slider("Chunk overlap", 0, 300, 150, 10)
k_candidates = st.sidebar.slider("Recall (candidates before rerank)", 5, 50, 30, 5)
k_final = st.sidebar.slider("Final answers to consider (k)", 1, 10, 3)
use_reranker = st.sidebar.checkbox("Use reranker (BGE base)", True)
show_chunks = st.sidebar.checkbox("Show retrieved chunks under answers", True)
if st.sidebar.button("Reset storage (full rebuild)"):
    import shutil
    shutil.rmtree(STORE_DIR, ignore_errors=True)
    for k in ("index","page_texts","page_labels","pages_df","chat","last_sources"):
        st.session_state.pop(k, None)
    st.sidebar.success("Storage cleared. Upload PDFs and Build again.")

# Upload PDFs
uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded:
    for f in uploaded:
        (PDF_DIR / f.name).write_bytes(f.read())
    st.success(f"Saved {len(uploaded)} file(s) to {PDF_DIR}")

with st.expander("📄 Files in data folder"):
    files = sorted(p.name for p in PDF_DIR.glob("*.pdf"))
    st.write(files or "No PDFs yet.")

def make_llm():
    if GROQ_KEY.startswith("gsk_"):
        return Groq(model=model_name, api_key=GROQ_KEY, temperature=temperature)
    if OPENAI_KEY.startswith("sk-"):
        os.environ["OPENAI_API_KEY"] = OPENAI_KEY
        return LIOpenAI(model="gpt-4o-mini", temperature=temperature)
    st.error("No valid API key found. Set GROQ_API_KEY (gsk_…) or OPENAI_API_KEY (sk-…).")
    st.stop()

llm = make_llm()

# Global embedder (+CPU, local cache)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=str(EMB_CACHE),
    device=DEVICE,
)

@st.cache_resource(show_spinner=True)
def build_and_persist(pdf_dir: Path, store_dir: Path, chunk_size: int, chunk_overlap: int):
    page_docs, rows = [], []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        with open(pdf_path, "rb") as f:
            r = PdfReader(f)
            for i, page in enumerate(r.pages, start=1):
                text = page.extract_text() or ""
                page_docs.append(Document(text=text, metadata={"file_name": pdf_path.name, "page": i}))
            rows.append({"file": pdf_path.name, "pages": len(r.pages)})

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embed = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(EMB_CACHE),
        device=DEVICE,
    )
    index = VectorStoreIndex.from_documents(page_docs, transformations=[splitter], embed_model=embed)

    store_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(store_dir))

    texts = [d.text for d in page_docs]
    labels = [f"{d.metadata.get('file_name')} (p.{d.metadata.get('page')})" for d in page_docs]
    pages_df = pd.DataFrame(rows)
    return index, texts, labels, pages_df

def load_persisted(store_dir: Path):
    if store_dir.exists():
        storage = StorageContext.from_defaults(persist_dir=str(store_dir))
        return load_index_from_storage(storage)
    return None

if st.button("🔧 Build / Rebuild Index", type="primary"):
    with st.spinner("Indexing & saving…"):
        idx, texts, labels, pages_df = build_and_persist(PDF_DIR, STORE_DIR, chunk_size_ui, chunk_overlap_ui)
        st.session_state.index = idx
        st.session_state.page_texts = texts
        st.session_state.page_labels = labels
        st.session_state.pages_df = pages_df
    st.success("Index ready!")

if "index" not in st.session_state:
    persisted = load_persisted(STORE_DIR)
    if persisted:
        st.session_state.index = persisted
        texts, labels, rows = [], [], []
        for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
            with open(pdf_path, "rb") as f:
                r = PdfReader(f)
                for i, page in enumerate(r.pages, start=1):
                    texts.append(page.extract_text() or "")
                    labels.append(f"{pdf_path.name} (p.{i})")
                rows.append({"file": pdf_path.name, "pages": len(r.pages)})
        st.session_state.page_texts = texts
        st.session_state.page_labels = labels
        st.session_state.pages_df = pd.DataFrame(rows)

# ---- Chat (always visible) ----
st.subheader("💬 Chat")
if "chat" not in st.session_state:
    st.session_state.chat = []
for role, text in st.session_state.chat:
    st.chat_message("user" if role == "You" else "assistant").write(text)

has_index = "index" in st.session_state
has_key = (os.getenv("GROQ_API_KEY","").startswith("gsk_")) or (os.getenv("OPENAI_API_KEY","").startswith("sk-"))
ready_to_chat = has_index and has_key

if not has_key:
    st.info("Paste a valid GROQ_API_KEY (gsk_…) or OPENAI_API_KEY (sk-…) before chatting.")
elif not has_index:
    st.info("Upload PDFs and click **Build / Rebuild Index** to enable the chat.")

user_q = st.chat_input(
    "Ask something about your PDFs…" if ready_to_chat else "Build the index to enable chat…",
    disabled=not ready_to_chat
)

if ready_to_chat:
    node_post = [SentenceTransformerRerank(
        model="BAAI/bge-reranker-base",
        top_n=k_final,
        device=DEVICE
    )] if use_reranker else []

    query_engine = st.session_state.index.as_query_engine(
        llm=llm,
        text_qa_template=PROMPT,
        similarity_top_k=k_candidates,
        node_postprocessors=node_post,
        response_mode="tree_summarize",
    )

    if user_q:
        st.session_state.chat.append(("You", user_q))
        with st.chat_message("user"):
            st.write(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp = query_engine.query(user_q)

            src, src_rows = [], []
            for s in getattr(resp, "source_nodes", [])[:max(1, k_final)]:
                meta = s.node.metadata or {}
                name = meta.get("file_name", "source")
                pg = meta.get("page")
                snip = (s.node.get_content() or "")[:250].replace("\n", " ")
                score = getattr(s, "score", None)
                src.append(f"- **{name} (p.{pg})**: {snip}…")
                src_rows.append({"file_name": name, "page": pg, "score": score, "snippet": snip})
            answer = str(resp)
            if src:
                answer += "\n\n---\n**Sources**\n" + "\n".join(src)

            st.write(answer)
            st.session_state.chat.append(("Bot", answer))
            st.session_state.last_sources = src_rows

# ---- Exports ----
st.markdown("### ⬇️ Export")
chat_json = json.dumps(
    [{"role": r, "content": t} for (r, t) in st.session_state.get("chat", [])],
    ensure_ascii=False, indent=2
)
st.download_button("Download chat (JSON)", data=chat_json,
                   file_name="chat_history.json", mime="application/json")

import io as _io
csv_buf = _io.StringIO()
writer = csv.writer(csv_buf); writer.writerow(["role", "content"])
for r, t in st.session_state.get("chat", []): writer.writerow([r, t])
st.download_button("Download chat (CSV)", data=csv_buf.getvalue(),
                   file_name="chat_history.csv", mime="text/csv")

if st.session_state.get("last_sources"):
    src_df = pd.DataFrame(st.session_state.last_sources)
    st.dataframe(src_df, use_container_width=True, height=200)
    st.download_button("Download sources (CSV)", data=src_df.to_csv(index=False),
                       file_name="last_answer_sources.csv", mime="text/csv")
    txt_payload = "\n\n---\n\n".join(
        f"{row['file_name']} (p.{row['page']}):\n{row['snippet']}"
        for row in st.session_state.last_sources
    )
    st.download_button("Download retrieved chunks (TXT)", data=txt_payload,
                       file_name="last_answer_chunks.txt", mime="text/plain")

# ---- Explore ----
st.divider()
st.subheader("🔎 Explore pages (keyword search)")
q = st.text_input("Search keywords across all pages", "")
n_hits = st.slider("Show top N", 3, 50, 10)
texts = st.session_state.get("page_texts", [])
labels = st.session_state.get("page_labels", [])
if q and texts:
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(texts)
    sims = linear_kernel(vec.transform([q]), X).ravel()
    order = sims.argsort()[::-1]; top_idx = order[:n_hits]
    rows = [{
        "rank": rank,
        "label": labels[i],
        "score": float(sims[i]),
        "snippet": (texts[i] or "")[:600].replace("\n", " ")
    } for rank, i in enumerate(top_idx, start=1)]
    hits_df = pd.DataFrame(rows)
    st.dataframe(hits_df, use_container_width=True, height=min(500, 56*len(rows)+40))
    st.download_button("⬇️ Download hits (CSV)",
                       data=hits_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"explore_{q.strip().replace(' ','_')}_hits.csv",
                       mime="text/csv")
    txt_payload = "\n\n---\n\n".join(
        f"[{r['rank']}] {r['label']}  score={r['score']:.6f}\n{r['snippet']}" for r in rows
    )
    st.download_button("⬇️ Download hits (TXT)", data=txt_payload,
                       file_name=f"explore_{q.strip().replace(' ','_')}_hits.txt",
                       mime="text/plain")

# ---- Mini chart ----
if st.session_state.get("pages_df") is not None and not st.session_state.pages_df.empty:
    st.divider()
    st.subheader("📊 Pages per PDF")
    fig = px.bar(st.session_state.pages_df, x="file", y="pages", title="Pages per PDF")
    st.plotly_chart(fig, use_container_width=True)
