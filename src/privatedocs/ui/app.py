"""Streamlit UI for PrivateDocs-LLM with expandable source citations and metrics."""

import os
import time

import requests
import streamlit as st

st.set_page_config(
    page_title="PrivateDocs-LLM | Confidential Document Q&A",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        background-color: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #38bdf8;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 8px;
    }
    .citation-card {
        background-color: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-title">🔒 PrivateDocs-LLM</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Self-hosted, air-gapped document intelligence benchmarked on AMD Instinct™ MI300X</div>',
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input(
        "API Base URL",
        value=os.getenv("API_BASE_URL", "http://localhost:8000"),
        help="FastAPI Gateway endpoint URL",
    )
    api_key = st.text_input(
        "API Key",
        value=os.getenv("USER_API_KEY", "user-secret-dev-key-change-in-prod"),
        type="password",
        help="X-API-Key header credential",
    )

    st.subheader("Retrieval Controls")
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        options=["hybrid", "dense", "bm25"],
        index=0,
        help="Dense (cosine), BM25 (keyword), or Hybrid (Reciprocal Rank Fusion)",
    )
    top_k = st.slider("Top Context Chunks (k)", min_value=1, max_value=10, value=5)

    doc_options = [
        "All Documents (No Filter)",
        "sec_apple_10k_2023",
        "sec_microsoft_10k_2023",
        "sec_nvidia_10k_2024",
        "sec_alphabet_10k_2023",
        "sec_amazon_10k_2023",
        "sec_amd_10k_2023",
        "sec_meta_10k_2023",
        "whitehouse_eo_14110_ai",
        "sec_cybersecurity_guidance_2018",
        "sec_cybersecurity_final_rule_2023",
    ]
    selected_doc = st.selectbox(
        "Filter Document / Company",
        options=doc_options,
        index=0,
        help="Restrict retrieval to chunks from a specific document or filing",
    )
    filter_doc_id = None if selected_doc.startswith("All Documents") else selected_doc

    st.divider()
    st.markdown("**System Health & Observability**")
    try:
        health_resp = requests.get(f"{api_url.rstrip('/')}/healthz", timeout=2)
        if health_resp.status_code == 200:
            st.success("API Gateway: Healthy")
        else:
            st.warning(f"API Gateway: HTTP {health_resp.status_code}")
    except Exception:
        st.error("API Gateway: Offline")

    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.rerun()

# Chat session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I am PrivateDocs Assistant. Ask me any question regarding your confidential internal documents.",
            "sources": [],
            "metrics": None,
        }
    ]

# Render chat messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("metrics"):
            m = msg["metrics"]
            st.markdown(
                f'<span class="metric-badge">⏱️ {m.get("latency_ms", 0):.0f} ms</span>'
                f'<span class="metric-badge">📥 Prompt: {m.get("prompt_tokens", 0)}</span>'
                f'<span class="metric-badge">📤 Generated: {m.get("completion_tokens", 0)}</span>'
                f'<span class="metric-badge">🔤 Total: {m.get("total_tokens", m.get("tokens", 0))}</span>',
                unsafe_allow_html=True,
            )
        if msg.get("sources"):
            with st.expander(f"📚 Verified Sources & Citations ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(
                        f"""<div class="citation-card">
                        <strong>[{s.get('doc_id')}, Page {s.get('page')}]</strong> (Score: {s.get('score', 0):.4f})<br/>
                        <em>Source: {s.get('source')}</em><br/>
                        <p style="margin-top: 6px; color: #cbd5e1;">"{s.get('snippet')}"</p>
                        </div>""",
                        unsafe_allow_html=True,
                    )

# User input handling
if prompt := st.chat_input("Ask a question about the documents..."):
    # Append user question
    st.session_state["messages"].append(
        {"role": "user", "content": prompt, "sources": [], "metrics": None}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API Gateway
    with st.chat_message("assistant"):
        with st.spinner("Retrieving confidential context and generating grounded answer..."):
            endpoint = f"{api_url.rstrip('/')}/v1/ask"
            headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
            from typing import Any

            payload: dict[str, Any] = {
                "question": prompt,
                "top_k": top_k,
                "mode": retrieval_mode,
                "doc_id": filter_doc_id,
                "stream": False,
            }

            try:
                t0 = time.perf_counter()
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()
                    answer_text = data.get("answer", "")
                    sources = data.get("sources", [])
                    metrics = {
                        "latency_ms": data.get("latency_ms", elapsed_ms),
                        "prompt_tokens": data.get("prompt_tokens", 0),
                        "completion_tokens": data.get("completion_tokens", 0),
                        "total_tokens": data.get("total_tokens", data.get("tokens", 0)),
                    }

                    st.markdown(answer_text)
                    st.markdown(
                        f'<span class="metric-badge">⏱️ {metrics["latency_ms"]:.0f} ms</span>'
                        f'<span class="metric-badge">📥 Prompt: {metrics["prompt_tokens"]} tokens</span>'
                        f'<span class="metric-badge">📤 Generated: {metrics["completion_tokens"]} tokens</span>'
                        f'<span class="metric-badge">🔤 Total: {metrics["total_tokens"]}</span>',
                        unsafe_allow_html=True,
                    )

                    if sources:
                        with st.expander(f"📚 Verified Sources & Citations ({len(sources)})"):
                            for s in sources:
                                st.markdown(
                                    f"""<div class="citation-card">
                                    <strong>[{s.get('doc_id')}, Page {s.get('page')}]</strong> (Score: {s.get('score', 0):.4f})<br/>
                                    <em>Source: {s.get('source')}</em><br/>
                                    <p style="margin-top: 6px; color: #cbd5e1;">"{s.get('snippet')}"</p>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "sources": sources,
                            "metrics": metrics,
                        }
                    )
                elif resp.status_code == 429:
                    st.error("⚠️ Rate limit exceeded. Please wait a moment before sending another request.")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as exc:
                st.error(f"Failed to communicate with API Gateway: {exc}")
