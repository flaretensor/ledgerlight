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
    if submitted and not question.strip():
        st.warning("Enter a question first.")
    elif submitted:
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
        if result["abstained"]:
            st.info("No answer found in the indexed passages.")
        else:
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
        st.caption(f"Ranking for: {st.session_state['question']}")
        st.subheader("Candidate passages")
        st.caption(
            "bm25: lexical BM25; dense: embedding similarity; rrf: reciprocal-rank fusion; "
            "ce: cross-encoder; ltr: learned CatBoost ranking; *_rank: 1 is best. "
            "Score scales differ by stage."
        )
        ranking = result["ranking"]
        candidates = pd.DataFrame(ranking["trace"])
        if "ltr_rank" in candidates:
            candidates = candidates.sort_values("ltr_rank", kind="stable")
        if "results" in ranking and "passage_id" in candidates:
            evidence = {item["passage_id"]: item for item in ranking["results"]}
            candidates["sent_to_model"] = candidates["passage_id"].isin(evidence)
            for key in ["ticker", "year", "form"]:
                values = {
                    pid: " / ".join(
                        dict.fromkeys(
                            str(source[key])
                            for source in item.get("sources", [])
                            if source.get(key) is not None
                        )
                    )
                    for pid, item in evidence.items()
                }
                if any(values.values()):
                    candidates[key] = candidates["passage_id"].map(values)
            st.caption(
                "sent_to_model marks the returned evidence passages supplied to the LLM. "
                "Ticker/year/form summarize source records in that evidence; blank means "
                "the metadata was not returned."
            )
        st.dataframe(candidates, hide_index=True, width="stretch")
        st.subheader("Stage timings")
        timings = pd.DataFrame(
            [
                {"stage": k, "seconds": round(v, 4)}
                for k, v in result["ranking"]["latencies"].items()
            ]
        )
        st.dataframe(timings, hide_index=True, width="stretch")
        st.subheader("Pipeline versions")
        st.json(result["ranking"]["versions"], expanded=False)
    else:
        st.info("Ask a question to inspect its retrieval and ranking scores.")

st.divider()
st.caption(
    "The collection contains selected passages, not complete filings or all of EDGAR. "
    "Source URLs are not supplied by the dataset."
)
