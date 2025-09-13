import sys
import os
import time
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from pubmed import PubMedRetriever
import html
import tempfile

# Patch sqlite3 for ChromaDB compatibility
try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except ImportError:
    pass

try:
    from groq import Groq
    from groq.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False


# ---------------------------
# Page config MUST come first
# ---------------------------
st.set_page_config(page_title="Healthcare Research Assistant", page_icon="🩺", layout="wide")


# ---------------------------
# Configuration
# ---------------------------
COLLECTION_NAME = "pubmed_if_articles"
# CHROMA_DIR = "./chroma_data"
CHROMA_DIR = os.path.join(tempfile.gettempdir(), "chroma_data")
DEFAULT_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
SEARCH_DEFAULT = "intermittent fasting and diabetes"
MAX_SEARCH = 200
RAG_N_RESULTS = 8
PER_DOC_CHAR_LIMIT = 2000
TOTAL_CHAR_LIMIT = 20000


# ---------------------------
# Cached resources
# ---------------------------
@st.cache_resource
def get_chroma_client(path: str = CHROMA_DIR):
    os.makedirs(path, exist_ok=True)
    if hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=path)
    elif hasattr(chromadb, "EphemeralClient"):
        return chromadb.EphemeralClient()
    else:
        raise ImportError("Your ChromaDB version does not support PersistentClient or EphemeralClient.")

@st.cache_resource
def get_embedder(model_name: str = DEFAULT_MODEL):
    return SentenceTransformer(model_name)

@st.cache_resource
def get_groq_client():
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    if not GROQ_AVAILABLE:
        st.warning("Groq SDK not installed or failed to import; generation will be disabled.")
        return None
    return Groq(api_key=key)


# ---------------------------
# Utility functions
# ---------------------------
def get_or_create_collection(client, name: str):
    try:
        return client.get_collection(name)
    except Exception:
        try:
            return client.create_collection(name=name)
        except Exception:
            return client.get_or_create_collection(name)

def pmid_search_and_fetch(search_term: str, max_results: int = 50):
    pmids = PubMedRetriever.search_pubmed_articles(search_term, max_results=max_results)
    if not pmids:
        return []
    articles = PubMedRetriever.fetch_pubmed_abstracts(pmids)
    return articles

def prepare_doc_and_meta(article: dict):
    pmid = str(article.get("pmid", ""))
    title = article.get("title", "No Title")
    abstract_field = article.get("abstract", {})
    if isinstance(abstract_field, dict):
        abstract_text = " ".join([t for t in abstract_field.values() if t])
    else:
        abstract_text = str(abstract_field or "")
    doc_text = f"{title}\n\n{abstract_text}".strip()
    meta = {
        "pmid": pmid,
        "title": title,
        "journal": article.get("journal", "Unknown Journal"),
        "authors": ", ".join(article.get("authors", [])) if isinstance(article.get("authors"), list) else article.get("authors", "No Authors"),
        "publication_date": article.get("publication_date", "Unknown Date")
    }
    return doc_text, meta

# ---------------------------
# Session state defaults
# ---------------------------
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'ask_in_progress' not in st.session_state:
    st.session_state.ask_in_progress = False


# ---------------------------
# CSS tweaks
# ---------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem !important;
    max-width: 70% !important;
    margin-left: auto;
    margin-right: auto;
}
.stApp .main {
    padding-top: 0rem !important;
}
.reportview-container .main .block-container {
    padding-top: 0rem !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------
# Header
# ---------------------------
st.markdown(f"""
<h1 style='color:royalblue; margin-bottom:0; font-size:25px;'>Healthcare Research Assistant</h1>
<p style='color:gray; margin-top:0; font-size:15px; margin-bottom:0;'>💊 Search PubMed, ingest selected articles to ChromaDB, and query via semantic search + optional Groq LLaMA-3</p>
""", unsafe_allow_html=True)
st.markdown("<div style='margin-top:6px; margin-bottom:6px;'></div>", unsafe_allow_html=True)


# ---------------------------
# Home Page
# ---------------------------
def home_page():
    col1, col2 = st.columns([16, 2])
    with col1:
        query = st.text_input(
            "💬 Type your question about intermittent fasting / metabolic disorders:",
            value="",
            key="main_query"
        )
    with col2:
        st.markdown("<div style='margin-top:28px;'>", unsafe_allow_html=True)
        ask_btn = st.button("🩺 Ask", key="ask_home")
        st.markdown("</div>", unsafe_allow_html=True)

    result_placeholder = st.empty()

    if ask_btn:
        st.session_state.ask_in_progress = True

    if st.session_state.ask_in_progress:
        with result_placeholder.container():
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                client = get_chroma_client()
                collection = get_or_create_collection(client, COLLECTION_NAME)
                embedder = get_embedder()
                groq_client = get_groq_client()

                with st.spinner("🧩 Embedding query..."):
                    q_emb = embedder.encode([query])[0]
                with st.spinner("📚 Retrieving from vector store..."):
                    results = collection.query(
                        query_embeddings=[q_emb.tolist()],
                        n_results=RAG_N_RESULTS,
                        include=["documents", "metadatas", "distances"]
                    )
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                dists = results.get("distances", [[]])[0] if "distances" in results else [None] * len(docs)

                # ✅ Check if any data exists in vector DB
                try:
                    if collection.count() == 0:
                        st.info(
                            "ℹ️ No data found in vector database. Please go to **Search Articles** page, search for articles and ingest them first.")
                        return
                except Exception:
                    st.info(
                        "ℹ️ No data found in vector database. Please go to **Search Articles** page, search for articles and ingest them first.")
                    return

                # --- RAG Answer before Retrieved documents ---
                if groq_client:
                    context_parts, total_chars = [], 0
                    for meta, doc in zip(metas, docs):
                        excerpt = doc[:PER_DOC_CHAR_LIMIT]
                        block = f"PMID: {meta.get('pmid', '')}\nTitle: {meta.get('title', '')}\nJournal: {meta.get('journal', '')}\nDate: {meta.get('publication_date', '')}\n\n{excerpt}\n\n"
                        if total_chars + len(block) > TOTAL_CHAR_LIMIT:
                            break
                        context_parts.append(block)
                        total_chars += len(block)

                    if context_parts:
                        context_text = "\n".join(context_parts)
                        system_msg = (
                            "You are an evidence-based medical research assistant. Use the provided research context as primary source. Do NOT invent facts. When making a claim, cite supporting PMID(s). If evidence is insufficient, state that clearly."
                        )
                        user_prompt = (
                            f"Question: {query}\n\nContext:\n{context_text}\n\nProvide a concise evidence-based answer (3-6 sentences), list PMIDs supporting each claim, and give a short confidence (High/Moderate/Low) with reason."
                        )

                        info_msg = st.info("⏳ Generating answer (may take a few seconds)...")
                        try:
                            messages = [
                                ChatCompletionSystemMessageParam(role="system", content=system_msg),
                                ChatCompletionUserMessageParam(role="user", content=user_prompt)
                            ]
                            chat_completion = groq_client.chat.completions.create(
                                model=os.environ['GROQ_MODEL'],
                                messages=messages,
                                temperature=0.0,
                                max_tokens=1000
                            )
                            final_answer = chat_completion.choices[0].message.content.strip()

                            # st.subheader("🧠 Answer")
                            # st.write(final_answer)

                            # escape HTML and convert newlines to <br> for safe display
                            safe_answer = html.escape(final_answer).replace("\n", "<br>")

                            info_msg.empty()

                            # show answer inside a pretty card
                            st.markdown(f"""
                                <div style="
                                    border:1px solid #e5e7eb;
                                    border-radius:12px;
                                    padding:16px 20px;
                                    margin-bottom:20px;
                                    background-color:#ffffff;
                                    box-shadow:0 2px 6px rgba(0,0,0,0.05);
                                ">
                                    <h4 style="margin:0; font-size:20px; color:#1e3a8a;">🧠 Answer</h4>
                                    <div style="margin-top:2px; font-size:14px; color:#111827; line-height:1.45;">
                                        {safe_answer}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                        except Exception as e:
                            info_msg.empty()
                            st.error(f"Generation failed: {e}")

                        st.markdown("<br>", unsafe_allow_html=True)
                        # ✅ Beautiful UI for Retrieved Evidence List
                        st.markdown(
                            '<div style="font-size:20px; font-weight:600; margin-bottom:8px;">📄 Retrieved Evidence List from PubMed</div>',
                            unsafe_allow_html=True
                        )
                        for i, (meta, doc, dist) in enumerate(zip(metas, docs, dists), start=1):
                            sim_text = f"{dist:.2f}" if isinstance(dist, (int, float)) else "N/A"
                            st.markdown(f"""
                            <div style="
                                border:1px solid #e5e7eb;
                                border-radius:12px;
                                padding:16px 20px;
                                margin-bottom:20px;
                                background-color:#ffffff;
                                box-shadow:0 2px 6px rgba(0,0,0,0.05);
                            ">
                                <h4 style="margin:0; font-size:16px; color:#1e3a8a;">{i}. {meta.get("title", "No Title")}</h4>
                                <p style="margin:1px 0; font-size:13px; color:#374151;">
                                    📖 <b>Journal:</b> {meta.get("journal", "")} &nbsp;|&nbsp; <b>Year:</b> {meta.get("publication_date", "")}
                                </p>
                                <p style="margin:6px 0; font-size:13px; color:#4b5563;">
                                    🔑 <b>PMID:</b> {meta.get("pmid", "")} &nbsp;|&nbsp; 🎯 <b>Similarity:</b> {sim_text}
                                </p>
                                <details style="margin:8px 0; font-size:13px;">
                                    <summary style="cursor:pointer; color:#2563eb;">📖 Preview abstract</summary>
                                    <p style="margin-top:6px; color:#111827;">{doc[:4000]}</p>
                                </details>
                            </div>
                            """, unsafe_allow_html=True)

        st.session_state.ask_in_progress = False


# ---------------------------
# Search + Ingest Page (same as your original, unchanged)
# ---------------------------

def search_articles_page():
    st.markdown("""
    <style>
    .search-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 22px;
        background-color: #ffffff;
        box-shadow: 0 3px 10px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .search-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    .search-title { font-size: 16px; font-weight:600; color:#1d4ed8; margin:0; }
    .search-meta { font-size:13px; color:#4b5563; margin-top:4px; }
    .search-snippet { font-size:13px; color:#111827; margin-top:6px; }
    .search-divider { border-top:1px dashed #e5e7eb; margin:12px 0; }
    </style>
    """, unsafe_allow_html=True)

    # --- Input controls row ---
    with st.container():
        client = get_chroma_client()
        collection = get_or_create_collection(client, COLLECTION_NAME)
        embedder = get_embedder()
        groq_client = get_groq_client()

        col1, col2 = st.columns([6, 2])
        with col1:
            search_term = st.text_input("🧬 PubMed search term", value=SEARCH_DEFAULT, key="search_term_input")
        with col2:
            max_results = st.slider("📊 Max results", min_value=5, max_value=MAX_SEARCH, value=50, step=5,
                                    key="max_results_slider")
        search_btn = st.button("🔎 Search PubMed", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []
    if "selected_pmids" not in st.session_state:
        st.session_state["selected_pmids"] = set()

    if search_btn:
        with st.spinner("🔄 Searching PubMed..."):
            articles = pmid_search_and_fetch(search_term, max_results=max_results)
            st.session_state["search_results"] = articles
            st.session_state["selected_pmids"] = set()
        st.success(f"✅ Found {len(articles)} articles.")

    results = st.session_state.get("search_results", [])
    if results:
        per_page = 5
        total_pages = (len(results) + per_page - 1) // per_page

        col_a, col_b = st.columns([14, 2])
        pad_px = 28
        with col_a:
            page = st.number_input("📑 Browse results and select articles to ingest.", min_value=1, max_value=total_pages, value=1, step=1, key="results_page")
        with col_b:
            # same spacer so button baseline matches number_input
            st.markdown(f'<div style="height:{pad_px}px"></div>', unsafe_allow_html=True)
            if st.button("Select All", key="select_all_page"):
                start_idx = (page - 1) * per_page
                end_idx = start_idx + per_page
                page_results = results[start_idx:end_idx]
                for art in page_results:
                    st.session_state["selected_pmids"].add(str(art.get("pmid", "")))

        st.markdown("<br>", unsafe_allow_html=True)

        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_results = results[start_idx:end_idx]

        # --- Keep this loop as requested ---
        for art in page_results:
            pmid = str(art.get("pmid", ""))
            title = art.get("title", "No Title")
            journal = art.get("journal", "")
            year = art.get("publication_date", "")
            abstract_field = art.get("abstract", {})
            if isinstance(abstract_field, dict):
                snippet = next((v for v in abstract_field.values() if v), "")[:300]
            else:
                snippet = str(abstract_field or "")[:300]

            key = f"sel_{pmid}"

            with st.container():
                col1, col2 = st.columns([55, 2])

                with col1:
                    st.markdown(
                        f"""
                                <div style="
                                    border:1px solid #e5e7eb;
                                    border-radius:12px;
                                    padding:16px 20px;
                                    margin-bottom:30px;
                                    background-color:#ffffff;
                                    box-shadow:0 2px 6px rgba(0,0,0,0.05);
                                ">
                                    <h4 style="margin:0; font-size:16px; color:#1e3a8a;">{title}</h4>
                                    <p style="margin:1px 0px 0px 0px; font-size:13px; color:#374151;">
                                        📖 <b>Journal:</b> {journal} &nbsp;|&nbsp; <b>Year:</b> {year}
                                    </p>
                                    <p style="margin:6px 0; font-size:13px; color:#4b5563;">
                                        🔑 <b>PMID:</b> {pmid}
                                    </p>
                                    <details style="margin:8px 1px 0px 3px; font-size:13px;">
                                        <summary style="cursor:pointer; color:#2563eb;">📖 Preview snippet</summary>
                                        <p style="margin-top:6px; color:#111827;">{snippet or "No abstract available"}</p>
                                    </details>
                                </div>
                                """,
                        unsafe_allow_html=True
                    )

                with col2:
                    checked = st.checkbox(
                        "", key=key, value=(pmid in st.session_state["selected_pmids"]),
                        label_visibility="collapsed"
                    )

            # sync state
            if checked:
                st.session_state["selected_pmids"].add(pmid)
            else:
                st.session_state["selected_pmids"].discard(pmid)
    else:
        st.info("ℹ️ No search results yet. Use the search box above.")

    # --- Footer buttons ---
    st.markdown("<br>", unsafe_allow_html=True)
    col_d, col_e = st.columns([1,1])
    with col_d:
        ingest_btn = st.button("➕ Ingest Selected Articles", use_container_width=True, type="primary")
    with col_e:
        clear_btn = st.button("🗑️ Clear All Articles", use_container_width=True)

    # Clear confirmation
    if clear_btn:
        st.session_state["confirm_delete"] = True

    if st.session_state.get("confirm_delete"):
        st.warning("⚠️ Are you sure? This will permanently delete the collection.")
        col1_del, col2_del = st.columns([1,1])
        with col1_del:
            if st.button("✅ Yes, delete it", use_container_width=True):
                try:
                    client.delete_collection(COLLECTION_NAME)
                    st.success("All articles cleared.")
                except Exception as e:
                    st.error(f"Failed to delete collection: {e}")
                st.session_state["confirm_delete"] = False
        with col2_del:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state["confirm_delete"] = False

    # Ingest logic
    if ingest_btn:
        selected = list(st.session_state.get("selected_pmids", []))
        if not selected:
            st.warning("⚠️ No articles selected for ingestion.")
        else:
            pmid2article = {str(a.get("pmid")): a for a in st.session_state.get("search_results", [])}
            to_ingest = [pmid2article.get(p) for p in selected if pmid2article.get(p)]
            try:
                existing = set(collection.get()["ids"])
            except Exception:
                existing = set()
            filtered = [a for a in to_ingest if a and str(a.get("pmid")) not in existing]
            if not filtered:
                st.info("ℹ️ No new documents to ingest (all already present).")
            else:
                st.info(f"⏳ Ingesting {len(filtered)} new docs (skipping existing).")
                progress = st.progress(0)
                BATCH = 16
                added = 0
                for i in range(0, len(filtered), BATCH):
                    batch = filtered[i:i + BATCH]
                    docs, metas, ids = [], [], []
                    for art in batch:
                        doc_text, meta = prepare_doc_and_meta(art)
                        if not doc_text.strip():
                            continue
                        ids.append(meta["pmid"])
                        docs.append(doc_text)
                        metas.append(meta)
                    if not docs:
                        continue
                    with st.spinner(f"🧪 Embedding batch {i // BATCH + 1}/{(len(filtered) + BATCH - 1) // BATCH}"):
                        embeddings = embedder.encode(docs, show_progress_bar=False)
                        emb_list = [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]
                        collection.add(ids=ids, documents=docs, embeddings=emb_list, metadatas=metas)
                    added += len(ids)
                    progress.progress(min(100, int(100 * (i + BATCH) / max(1, len(filtered)))))
                    time.sleep(0.1)
                progress.empty()
                st.success(f"✅ Added {added} new documents to database.")


# ---------------------------
# Navigation Tabs
# ---------------------------
tabs = st.tabs(["Home", "Add More PubMed Articles"])
with tabs[0]:
    st.session_state.page = 'home'
    home_page()
with tabs[1]:
    st.session_state.page = 'add'
    search_articles_page()
