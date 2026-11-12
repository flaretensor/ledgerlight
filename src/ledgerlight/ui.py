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
    st.subheader("Filters")
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
        st.caption(f"Question: {st.session_state['question']}")
        if filters != st.session_state["filters"]:
            st.info("The result below uses the filters from the last submitted question.")
        if st.session_state["filters"]:
            st.caption(
                "Filters: "
                + " · ".join(f"{k}: {v}" for k, v in st.session_state["filters"].items())
            )
        if result["abstained"]:
            st.info("No answer found in the indexed passages.")
        else:
            st.markdown(result["answer"], unsafe_allow_html=False)
        latency_caption = f"{result['latencies']['e2e']:.3f} seconds"
        if not result["abstained"]:
            count = len(result["citations"])
            latency_caption += f" · {count} cited {'source' if count == 1 else 'sources'}"
        st.caption(latency_caption)
        for citation in result["citations"]:
            source_labels = [
                f"{s['ticker']} · {s['form']} · {s['year']} · p. {s['page']}"
                for s in citation["sources"]
            ]
            with st.expander(f"[{citation['number']}] " + " / ".join(source_labels)):
                st.text(citation["text"])
        if result["abstained"]:
            st.caption(
                "The retrieved passages did not support an answer. Rephrasing the question "
                "or changing the filters may help. See the ranking tab for the retrieved passages."
            )

with rank_tab:
    result = st.session_state.get("result")
    if result:
        st.caption(f"Ranking for: {st.session_state['question']}")
        st.subheader("Candidate passages")
        st.caption(
            "bm25: word matching; dense: embedding similarity; rrf: combined ranks; "
            "ce: cross-encoder; ltr: CatBoost. Rank 1 is best; score scales differ."
        )
        if st.session_state["filters"]:
            st.caption(
                "Filters applied before retrieval: "
                + " · ".join(f"{k}: {v}" for k, v in st.session_state["filters"].items())
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
                    candidates[key] = (
                        candidates["passage_id"].map(values).fillna("—").replace("", "—")
                    )
            st.caption(
                "sent_to_model marks the returned evidence passages supplied to the LLM. "
                "Ticker/year/form summarize returned source records; — means metadata "
                "was not returned. Cited marks passages referenced by the answer."
            )
        if "passage_id" in candidates:
            cited = {item["passage_id"] for item in result["citations"]}
            candidates["cited"] = candidates["passage_id"].isin(cited)
            flags = [key for key in ["sent_to_model", "cited"] if key in candidates]
            columns = [key for key in candidates if key not in flags]
            anchor = "ltr_rank" if "ltr_rank" in columns else "passage_id"
            position = columns.index(anchor) + 1
            candidates = candidates[columns[:position] + flags + columns[position:]]
        st.dataframe(candidates, hide_index=True, width="stretch", height="content")
        st.caption(
            "Full retrieved passages are shown below. The LLM receives selected excerpts, "
            "which may be shortened to fit its context."
        )
        for passage in ranking.get("results", []):
            if passage.get("text"):
                source_labels = [
                    f"{source.get('ticker') or '—'} · {source.get('form') or '—'} · "
                    f"{source.get('year') or '—'} · p. {source.get('page') or '—'}"
                    for source in passage.get("sources", [])
                ]
                label = passage["passage_id"]
                if source_labels:
                    label += " · " + " / ".join(source_labels)
                with st.expander(label):
                    st.text(passage["text"])
        st.subheader("Retrieval-stage timings")
        latency_parts = []
        if "e2e" in result["latencies"]:
            latency_parts.append(f"Gateway end-to-end: {result['latencies']['e2e']:.3f} s")
        if "generation" in result["latencies"]:
            latency_parts.append(
                f"LLM generation (including any retry): {result['latencies']['generation']:.3f} s"
            )
        if latency_parts:
            st.caption(" · ".join(latency_parts))
        timings = ranking["latencies"]
        summary_keys = [key for key in ["queue", "registry", "total", "e2e"] if key in timings]
        stages = [key for key in ["bm25", "embed", "qdrant", "ce", "ltr"] if key in timings]
        stages += [key for key in timings if key not in stages + summary_keys]
        if stages:
            stage_rows = [{"stage": key, "seconds": round(timings[key], 3)} for key in stages]
            st.dataframe(
                pd.DataFrame(stage_rows),
                hide_index=True,
                width="stretch",
                height="content",
                column_config={"seconds": st.column_config.NumberColumn(format="%.3f")},
            )
        if summary_keys:
            labels = {
                "queue": "Retrieval queue wait",
                "registry": "Registry lookup",
                "total": "Retrieval engine total",
                "e2e": "Retrieval service end-to-end",
            }
            st.caption("Retrieval summary: totals overlap the stages; do not add these rows.")
            summary_rows = [
                {"summary": labels[key], "seconds": round(timings[key], 3)} for key in summary_keys
            ]
            st.dataframe(
                pd.DataFrame(summary_rows),
                hide_index=True,
                width="stretch",
                height="content",
                column_config={"seconds": st.column_config.NumberColumn(format="%.3f")},
            )
        st.subheader("Pipeline versions")
        st.caption("Recorded model, index, and runtime versions for this response.")
        st.json(result["ranking"]["versions"], expanded=False)
    else:
        st.info("Ask a question to inspect its retrieval and ranking scores.")

st.divider()
st.caption(
    "The collection contains selected passages, not complete filings or all of EDGAR. "
    "Source URLs are not supplied by the dataset."
)
