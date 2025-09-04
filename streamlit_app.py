# streamlit_app.py
"""
MediAssist AI — Streamlit UI for PubMed -> Chroma ingestion and RAG queries (Groq LLaMA-3)

Requirements:
 - Put this file next to your provided pubmed.py
 - pip install streamlit chromadb sentence-transformers torch groq python-dotenv
 - Create .env with GROQ_API_KEY (optional; ingestion + retrieval work without it)

Notes:
 - Uses chromadb.PersistentClient(path="./chroma_data") so collections persist across runs.
 - Uses SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO') for biomedical embeddings.
 - Avoids re-downloading model by caching with st.cache_resource.
"""

import os
import time
import json
from typing import List

import streamlit as st
from dotenv import load_dotenv

# PubMed helper you supplied
from pubmed import PubMedRetriever

# Embeddings & Vector DB
from sentence_transformers import SentenceTransformer
import chromadb

# Groq (optional)
try:
    from groq import Groq
    from groq.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

# ---------------------------
# Configuration (tweakable)
# ---------------------------
COLLECTION_NAME = "pubmed_if_articles"
CHROMA_DIR = "./chroma_data"
DEFAULT_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
SEARCH_DEFAULT = "intermittent fasting AND type 2 diabetes"
MAX_SEARCH = 200  # safety cap for UI
# How many docs to pass to LLM and how many chars per doc
RAG_N_RESULTS = 8
PER_DOC_CHAR_LIMIT = 2000
TOTAL_CHAR_LIMIT = 20000

# ---------------------------
# Cached resources
# ---------------------------
@st.cache_resource
def get_chroma_client(path: str = CHROMA_DIR):
    # persistent client for ChromaDB v1.x
    return chromadb.PersistentClient(path=path)

@st.cache_resource
def get_embedder(model_name: str = DEFAULT_MODEL):
    # Loading model may take time the first run; caching avoids reloads
    return SentenceTransformer(model_name)

@st.cache_resource
def get_groq_client():
    # load env and initialize Groq (returns None if not configured/available)
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
            # fallback (some chroma versions implement get_or_create)
            return client.get_or_create_collection(name)

def pmid_search_and_fetch(search_term: str, max_results: int = 50):
    """
    Search PubMed and return list of article dicts (same format as pubmed.fetch_pubmed_abstracts).
    WARNING: PubMed API rate-limit — keep max_results moderate in UI.
    """
    pmids = PubMedRetriever.search_pubmed_articles(search_term, max_results=max_results)
    if not pmids:
        return []
    articles = PubMedRetriever.fetch_pubmed_abstracts(pmids)
    return articles

def prepare_doc_and_meta(article: dict):
    """
    Returns (doc_text, meta_dict) using full abstract when available.
    """
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
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="MediAssist AI — PubMed RAG", layout="wide")
st.title("MediAssist AI — PubMed Search & RAG")
st.caption("Search PubMed, ingest selected articles to ChromaDB, and query via semantic search + Groq LLaMA-3 (optional).")

# Sidebar: Search & ingest
st.sidebar.header("Document Search & Ingest")
search_term = st.sidebar.text_input("PubMed search term", value=SEARCH_DEFAULT)
max_results = st.sidebar.slider("Max results to retrieve", min_value=5, max_value=MAX_SEARCH, value=50, step=5)

if "search_results" not in st.session_state:
    st.session_state["search_results"] = []  # list of article dicts
if "selected_pmids" not in st.session_state:
    st.session_state["selected_pmids"] = set()

search_btn = st.sidebar.button("Search PubMed")

if search_btn:
    with st.spinner("Searching PubMed..."):
        articles = pmid_search_and_fetch(search_term, max_results=max_results)
        st.session_state["search_results"] = articles
        st.session_state["selected_pmids"] = set()  # reset selections
    st.sidebar.success(f"Found {len(articles)} articles.")

# Show search results with checkboxes
st.sidebar.markdown("### Results")
if st.session_state["search_results"]:
    for art in st.session_state["search_results"]:
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
        checked = st.sidebar.checkbox(f"{title} — {journal} ({year})", key=key, value=(pmid in st.session_state["selected_pmids"]))
        if checked:
            st.session_state["selected_pmids"].add(pmid)
        else:
            st.session_state["selected_pmids"].discard(pmid)
        # small expandable snippet
        with st.sidebar.expander("Preview snippet", expanded=False):
            st.write(snippet or "No abstract available")
else:
    st.sidebar.info("No search results yet. Use the search box above.")

# Ingest button
client = get_chroma_client()
collection = get_or_create_collection(client, COLLECTION_NAME)
embedder = get_embedder()
ingest_btn = st.sidebar.button("Ingest selected into Vector Store")

if ingest_btn:
    selected = list(st.session_state.get("selected_pmids", []))
    if not selected:
        st.sidebar.warning("No articles selected for ingestion.")
    else:
        # build mapping pmid->article from search results
        pmid2article = {str(a.get("pmid")): a for a in st.session_state.get("search_results", [])}
        to_ingest = [pmid2article.get(p) for p in selected if pmid2article.get(p)]
        # duplicates check
        try:
            existing = set(collection.get()["ids"])
        except Exception:
            existing = set()
        # filter out duplicates
        filtered = [a for a in to_ingest if a and str(a.get("pmid")) not in existing]
        if not filtered:
            st.sidebar.info("No new documents to ingest (all were already present).")
        else:
            st.sidebar.info(f"Ingesting {len(filtered)} new docs (skipping existing).")
            progress = st.sidebar.progress(0)
            BATCH = 16
            added = 0
            for i in range(0, len(filtered), BATCH):
                batch = filtered[i:i+BATCH]
                docs = []
                metas = []
                ids = []
                for art in batch:
                    doc_text, meta = prepare_doc_and_meta(art)
                    if not doc_text.strip():
                        continue
                    ids.append(meta["pmid"])
                    docs.append(doc_text)
                    metas.append(meta)
                if not docs:
                    continue
                with st.spinner(f"Embedding batch {i//BATCH + 1}/{(len(filtered)+BATCH-1)//BATCH}"):
                    embeddings = embedder.encode(docs, show_progress_bar=False)
                    # convert embeddings to python lists
                    emb_list = [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]
                    collection.add(ids=ids, documents=docs, embeddings=emb_list, metadatas=metas)
                added += len(ids)
                progress.progress(min(100, int(100 * (i + BATCH) / max(1, len(filtered)))))
                time.sleep(0.1)
            progress.empty()
            st.sidebar.success(f"Added {added} new documents to '{COLLECTION_NAME}'.")

# Main area: query & results
st.markdown("---")
st.header("Query the ingested documents")
query = st.text_input("Type your question about intermittent fasting / metabolic disorders:", value="", key="main_query")

col1, col2 = st.columns([3,1])
with col2:
    if GROQ_AVAILABLE and get_groq_client():
        st.success("Groq available")
    else:
        st.info("Groq not configured. RAG generation disabled (retrieval-only).")

ask_btn = st.button("Ask")

if ask_btn:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        # 1) embed query
        with st.spinner("Embedding query..."):
            q_emb = embedder.encode([query])[0]
        # 2) query chroma
        with st.spinner("Retrieving from vector store..."):
            results = collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=RAG_N_RESULTS,
                include=["documents", "metadatas", "distances"]
            )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0] if "distances" in results else [None]*len(docs)

        st.subheader("Retrieved documents")
        for i, (meta, doc, dist) in enumerate(zip(metas, docs, dists), start=1):
            with st.expander(f"{i}. {meta.get('title','No Title')} — {meta.get('journal','')} ({meta.get('publication_date','')})"):
                st.write(f"**PMID:** {meta.get('pmid','')}")
                st.write(f"**Authors:** {meta.get('authors','')}")
                st.write(f"**Distance (similarity):** {dist}")
                st.markdown("**Excerpt / Abstract:**")
                st.write(doc[:4000])  # show up to 4000 chars

        # 3) If Groq configured, call LLaMA-3 for RAG answer
        groq_client = get_groq_client()
        if not groq_client:
            st.info("Groq client not available. Showing retrieval results only. Set GROQ_API_KEY in .env to enable generation.")
        else:
            # Build context by concatenating per-doc slices (respect TOTAL_CHAR_LIMIT)
            context_parts = []
            total_chars = 0
            for meta, doc in zip(metas, docs):
                excerpt = doc[:PER_DOC_CHAR_LIMIT]
                block = f"PMID: {meta.get('pmid','')}\nTitle: {meta.get('title','')}\nJournal: {meta.get('journal','')}\nDate: {meta.get('publication_date','')}\n\n{excerpt}\n\n"
                if total_chars + len(block) > TOTAL_CHAR_LIMIT:
                    break
                context_parts.append(block)
                total_chars += len(block)
            if not context_parts:
                st.warning("No usable context available for generation.")
            else:
                context_text = "\n".join(context_parts)
                # Compose a careful prompt
                system_msg = (
                    "You are an evidence-based medical research assistant. Use the provided research context "
                    "as primary source. Do NOT invent facts. When making a claim, cite supporting PMID(s). "
                    "If evidence is insufficient, state that clearly."
                )
                user_prompt = (
                    f"Question: {query}\n\nContext:\n{context_text}\n\n"
                    "Provide a concise evidence-based answer (3-6 sentences), list PMIDs supporting each claim, "
                    "and give a short confidence (High/Moderate/Low) with reason."
                )

                st.info("Generating answer via Groq LLaMA-3 (may take a few seconds)...")
                try:
                    # Use typed messages for SDK / type-hint compliance
                    messages = [
                        ChatCompletionSystemMessageParam(role="system", content=system_msg),
                        ChatCompletionUserMessageParam(role="user", content=user_prompt)
                    ]
                    chat_completion = groq_client.chat.completions.create(
                        model=os.environ['GROQ_MODEL'],  # faster than 70b; change if you want
                        messages=messages,
                        temperature=0.0,
                        max_tokens=1000
                    )

                    final_answer = chat_completion.choices[0].message.content.strip()
                    st.subheader("RAG Answer (LLaMA-3)")
                    st.write(final_answer)

                except Exception as e:
                    st.error(f"Generation failed: {e}")

# Footer with quick actions
st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Show collection stats"):
        try:
            info = collection.count()
            st.write(info)
        except Exception as e:
            st.error(f"Failed to get collection stats: {e}")
with col_b:
    if st.button("Delete collection (CAREFUL)"):
        st.session_state["confirm_delete"] = True

    if st.session_state.get("confirm_delete", False):
        st.warning("Are you sure? This will permanently delete the collection.")
        if st.button("Yes, delete it"):
            try:
                client.delete_collection(COLLECTION_NAME)
                st.success(f"Deleted collection '{COLLECTION_NAME}'.")
                st.session_state["confirm_delete"] = False
            except Exception as e:
                st.error(f"Failed to delete collection: {e}")
        if st.button("Cancel"):
            st.session_state["confirm_delete"] = False