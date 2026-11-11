import os

import httpx
import pandas as pd
import streamlit as st

GATEWAY = os.environ.get("LEDGERLIGHT_GATEWAY_URL", "http://127.0.0.1:8000")
RETRIEVAL = os.environ.get("LEDGERLIGHT_RETRIEVAL_URL", "http://127.0.0.1:8001")

st.set_page_config(page_title="Ledgerlight", page_icon="📑", layout="wide")
st.title("Ledgerlight")
st.caption("Financial documents, with sources. Search 5,230 curated passages from company filings.")


@st.cache_data(ttl=10)
def metadata():
    response = httpx.get(RETRIEVAL + "/health", timeout=15)
    response.raise_for_status()
    return response.json()


try:
    health = metadata()
except httpx.HTTPError:
    st.info("The local search service is starting. Start Ledgerlight and refresh this page.")
    st.stop()

with st.sidebar:
    st.subheader("Sources")
    filters = {}
    for key, label in [
        ("ticker", "Company ticker"),
        ("year", "Filing year"),
        ("form", "Filing form"),
    ]:
        selected = st.selectbox(label, ["Any", *health["filters"][key]])
        if selected != "Any":
            filters[key] = selected
    st.caption("Filters apply before retrieval. Page references appear in each citation.")

ask_tab, rank_tab = st.tabs(["Ask", "How it was ranked"])
with ask_tab:
    with st.form("ask_form"):
        question = st.text_area(
            "Ask about the filings",
            height=100,
            max_chars=2000,
            placeholder="Which business segments does Johnson & Johnson operate?",
        )
        submitted = st.form_submit_button("Find an answer", type="primary")
    if submitted and question.strip():
        with st.spinner("Reading the sources…"):
            try:
                response = httpx.post(
                    GATEWAY + "/ask", json={"question": question, "filters": filters}, timeout=300
                )
                response.raise_for_status()
                st.session_state["result"] = response.json()
                st.session_state["question"] = question
                st.session_state["filters"] = filters.copy()
            except httpx.HTTPError:
                st.error(
                    "A local service could not finish the request. Check the service status and retry."
                )
    result = st.session_state.get("result")
    if result:
        st.subheader(st.session_state["question"])
        if filters != st.session_state["filters"]:
            st.info("The result below uses the filters from the last submitted question.")
        if st.session_state["filters"]:
            st.caption(" · ".join(f"{k}: {v}" for k, v in st.session_state["filters"].items()))
        st.markdown(result["answer"], unsafe_allow_html=False)
        st.caption(
            f"{result['latencies']['e2e']:.2f} seconds · {len(result['citations'])} cited sources"
        )
        for citation in result["citations"]:
            source_labels = [
                f"{s['ticker']} · {s['form']} · {s['year']} · p. {s['page']}"
                for s in citation["sources"]
            ]
            with st.expander(f"[{citation['number']}] " + " / ".join(source_labels)):
                st.text(citation["text"])
        if result["abstained"]:
            st.caption("Try a more specific question or adjust the source filters.")

with rank_tab:
    result = st.session_state.get("result")
    if result:
        st.caption(
            "Candidate scores for the last submitted question. Score scales differ by stage."
        )
        st.dataframe(
            pd.DataFrame(result["ranking"]["trace"]), hide_index=True, use_container_width=True
        )
        timings = pd.DataFrame(
            [
                {"stage": k, "seconds": round(v, 4)}
                for k, v in result["ranking"]["latencies"].items()
            ]
        )
        st.dataframe(timings, hide_index=True, use_container_width=True)
        st.json(result["ranking"]["versions"], expanded=False)
    else:
        st.info("Ask a question to inspect its retrieval and ranking scores.")

st.divider()
st.caption(
    "The collection contains selected passages, not complete filings or all of EDGAR. "
    "Source URLs are not supplied by the dataset."
)
