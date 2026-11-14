import json
import os
import re
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

GATEWAY = os.environ.get("LEDGERLIGHT_GATEWAY_URL", "http://127.0.0.1:8000")
RETRIEVAL = os.environ.get("LEDGERLIGHT_RETRIEVAL_URL", "http://127.0.0.1:8001")
COMPANIES = {
    "ABBV": "AbbVie",
    "AMGN": "Amgen",
    "LNG": "Cheniere Energy",
    "CVX": "Chevron",
    "COP": "ConocoPhillips",
    "CTRA": "Coterra Energy",
    "DVN": "Devon Energy",
    "LLY": "Eli Lilly",
    "ET": "Energy Transfer",
    "EPD": "Enterprise Products",
    "EOG": "EOG Resources",
    "XOM": "ExxonMobil",
    "F": "Ford",
    "GM": "General Motors",
    "JNJ": "Johnson & Johnson",
    "KMI": "Kinder Morgan",
    "MPC": "Marathon Petroleum",
    "MRK": "Merck",
    "MRNA": "Moderna",
    "PFE": "Pfizer",
    "TSLA": "Tesla",
    "WMB": "Williams Companies",
}
REPORT_TYPES = {"10-K": "Annual report (10-K)", "10-Q": "Quarterly report (10-Q)"}
HEADER_HELP = (
    "The search covers 5,230 passages from annual (10-K) and quarterly (10-Q) reports "
    "that 22 U.S. energy, healthcare, and automotive companies filed with the U.S. "
    "Securities and Exchange Commission (SEC) in 2022–2025.\n\n"
    "The passages come from FinRank, a public retrieval dataset of questions and labeled "
    "excerpts from these reports. They are excerpts, not complete reports, so some "
    "questions can't be answered here. "
    "The dataset has no links to the original filings; each source shows the company, "
    "report type, filing year, and page instead. Passage text is shown as stored in the "
    "dataset, including leftover markup.\n\n"
    "This is a retrieval-augmented generation (RAG) pipeline: BM25 and dense retrieval, "
    "reciprocal rank fusion, a cross-encoder, a CatBoost ranker, and a small local LLM "
    "(Qwen3.5-2B) that writes the answer from the top passages. The LLM can make mistakes; "
    "check the sources. The Search details tab shows each stage."
)
SOURCE_NAVIGATION = """<script>
if (!window.ledgerlightSourceNavigation) {
  window.ledgerlightSourceNavigation = true;
  document.addEventListener("click", async event => {
    if (!(event.target instanceof Element)) return;
    const link = event.target.closest("a");
    if (!link) return;
    const url = new URL(link.href, location.href);
    if (url.origin !== location.origin || url.pathname !== location.pathname ||
        !/^#source-\\d+$/.test(url.hash)) return;
    const source = document.getElementById(url.hash.slice(1));
    if (!source) return;
    event.preventDefault();
    const details = source.closest("details");
    if (!details) return;
    await Promise.allSettled(details.getAnimations().map(animation => animation.finished));
    if (!details.open) {
      details.querySelector("summary")?.click();
    }
    for (let frame = 0; frame < 120; frame++) {
      await new Promise(requestAnimationFrame);
      const content = details.querySelector('[data-testid="stExpanderDetails"]');
      const summary = details.querySelector("summary");
      if (details.open && content && !content.inert && summary &&
          content.getBoundingClientRect().height > 0 &&
          details.getBoundingClientRect().height + 1 >=
          summary.getBoundingClientRect().height + content.getBoundingClientRect().height) break;
    }
    await Promise.allSettled(details.getAnimations().map(animation => animation.finished));
    await new Promise(requestAnimationFrame);
    location.hash = url.hash;
    details.scrollIntoView({block: "start"});
  });
}
</script>"""
CANDIDATE_COLUMNS = [
    ("passage_id", "Passage", "Passage ID in the FinRank dataset."),
    ("ltr_rank", "CatBoost rank", "Final rank."),
    ("sent_to_model", "In context", "The passage was sent to the LLM."),
    ("cited", "Cited", "The answer cites this passage."),
    (
        "ce_rank",
        "CE rank",
        "Rank by the cross-encoder (CE), a MiniLM model that reads the question and a passage "
        "window together and outputs one relevance score.",
    ),
    ("rrf_rank", "RRF rank", "Rank after reciprocal rank fusion of the BM25 and dense rankings."),
    ("bm25_rank", "BM25 rank", "Rank in BM25 keyword retrieval."),
    (
        "dense_rank",
        "Dense rank",
        "Rank in dense retrieval: embedding similarity between the question and the "
        "passage's best window (BGE-small).",
    ),
    (
        "ltr",
        "CatBoost score",
        "Output of the CatBoost ranker. Higher is better; only the order matters.",
    ),
    ("ce", "CE score", "Score from the cross-encoder (CE), a relevance logit. Higher is better."),
    (
        "rrf",
        "RRF score",
        "Sum of reciprocal ranks from BM25 and dense retrieval. Higher is better.",
    ),
    ("bm25", "BM25 score", "BM25 score. Higher is better."),
    (
        "dense",
        "Dense score",
        "Embedding similarity between the question and the passage's best window. Higher is better.",
    ),
]


def company_name(ticker):
    return COMPANIES.get(str(ticker).upper(), str(ticker))


def company_option(ticker):
    if ticker is None:
        return "All companies"
    if ticker.upper() in COMPANIES:
        return f"{company_name(ticker)} ({ticker})"
    return ticker


def source_label(sources, bold=False):
    labels = []
    for source in sources:
        page = re.sub(r"(?<=\d)\s*[-–—]\s*(?=\d)", "–", str(source["page"]))
        company = company_name(source["ticker"])
        labels.append(
            f"{'**' + company + '**' if bold else company} · "
            f"{REPORT_TYPES.get(source['form'], source['form'])} · "
            f"filed {source['year']} · p. {page}"
        )
    return " / ".join(labels)


def linked_answer(answer, citations):
    labels = {str(c["number"]): f"[{c['number']}] " + source_label(c["sources"]) for c in citations}

    def link(match):
        space, number = match.group(1), match.group(2)
        if number not in labels:
            return match.group(0)
        title = labels[number].replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        return f'{chr(160) if space else ""}[{number}](#source-{number} "{title}")'

    return re.sub(r"([ \t]*)\[(\d+)\]", link, answer)


def duration(seconds):
    if seconds < 0.001:
        return "<1 ms"
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"


def passage_body(passage, clipped=False):
    windows = passage.get("windows")
    if windows and not clipped:
        st.markdown("**Excerpt sent to the LLM**")
        st.text("\n\n".join(windows))
        st.markdown("**Full passage**")
    st.text(passage["text"])


def version_values(versions, prefix=""):
    flattened = {}
    for key, value in versions.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            flattened.update(version_values(value, dotted))
        else:
            flattened[dotted] = value
    return flattened


def versions_line(versions):
    values = version_values(versions)
    segments = [
        f"{label} v{values[key]}"
        for key, label in [
            ("pipeline", "Pipeline"),
            ("components.encoder", "encoder"),
            ("components.cross_encoder", "cross-encoder"),
        ]
        if key in values and values[key] is not None
    ]
    if values.get("method") is not None and values.get("components.ranker") is not None:
        segments.append(f"{values['method']} ranker v{values['components.ranker']}")
    if values.get("backend") is not None:
        segments.append(str(values["backend"]))
    return " · ".join(segments)


def versions_table(versions):
    values = version_values(versions)
    labels = {
        "pipeline": "Pipeline",
        "method": "Ranking method",
        "components.encoder": "Encoder",
        "components.cross_encoder": "Cross-encoder",
        "components.ranker": "Ranker",
        "collection": "Collection",
        "index": "Index",
        "backend": "Backend",
        "llm.revision": "LLM revision",
        "llm.sha256": "LLM SHA-256",
        "llm.runtime_build": "LLM runtime build",
    }
    rows = []
    for key in [*labels, *(key for key in values if key not in labels)]:
        if key not in values:
            continue
        value = values[key]
        value = (
            json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        )
        if key == "llm.sha256":
            value = value.removeprefix("sha256:")
        rows.append({"Component": labels.get(key, key), "Version": value})
    return pd.DataFrame(rows, columns=["Component", "Version"])


st.set_page_config(
    page_title="Ledgerlight",
    page_icon=Path(__file__).parent / "assets" / "favicon.png",
    layout="centered",
)
st.html(
    """<style>
    [data-testid="stMainBlockContainer"]{max-width:768px;padding:3rem 24px 2rem}
    [data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p,
    [role="tooltip"], [role="tooltip"] p,
    [data-testid="stWidgetLabel"] p,[data-baseweb="select"]{color:#1f1f1f!important;opacity:1}
    [data-testid="stCaptionContainer"] p{font-size:15px}
    .st-key-answer_stats [data-testid="stCaptionContainer"] p{font-size:13px}
    [data-testid="stTooltipIcon"],[data-testid="stTooltipIcon"] button{color:#5f6368;opacity:1}
    [role="tooltip"] p{margin-bottom:0}
    [role="tooltip"] [data-testid="stMarkdownContainer"]{margin-bottom:0}
    [role="tooltip"] p+p{margin-top:.5em}
    [data-testid="stTextAreaRootElement"]{background:#fff;border:1px solid #c7cace;
        border-radius:16px;box-shadow:0 2px 8px #1f1f1f0a}
    [data-testid="stTextArea"] textarea{font-size:17px;background:transparent;border:0;
        padding:12px 14px}
    [data-testid="stTextArea"] textarea::placeholder{color:#6b7177;opacity:1}
    a[href^="#source-"]{background:#fdecea;color:#a8190d;font-size:12px;font-weight:600;
        border-radius:6px;padding:0 6px;text-decoration:none}
    a[href^="#"]:not([href^="#source-"]){color:#df2f1f}
    [data-testid="stExpander"] details{border:0;background:#f5f5f5;border-radius:12px}
    [data-testid="stSpinnerIcon"]{border-color:#df2f1f33!important;border-top-color:#df2f1f!important}
    [data-testid="stExpander"] [data-testid="stText"]{font-family:inherit;font-size:14.5px;
        line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere}
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p{font-size:14px}
    .st-key-candidate_heading{margin-bottom:2rem}
    .st-key-candidate_heading [data-testid="stHorizontalBlock"]{flex-wrap:nowrap}
    .st-key-candidate_heading [data-testid="stColumn"]:first-child{flex:1;min-width:0}
    .st-key-candidate_heading [data-testid="stColumn"]:last-child{flex:0 0 136px;min-width:0;width:136px}
    </style>"""
)
st.title("Ledgerlight")
st.caption(
    "Ask questions about the financial reports of 22 U.S. companies. Every answer cites its sources.",
    help=HEADER_HELP,
)


@st.cache_data(ttl=10)
def metadata():
    response = httpx.get(RETRIEVAL + "/health", timeout=15)
    response.raise_for_status()
    return response.json()


try:
    health = metadata()
except httpx.HTTPError:
    st.info("The app is still starting. Refresh this page in a minute.")
    st.stop()

ask_tab, rank_tab = st.tabs(["Ask", "Search details"])
with ask_tab:
    with st.form("ask_form", border=False):
        question = st.text_area(
            "Question",
            label_visibility="collapsed",
            height=80,
            max_chars=2000,
            placeholder=(
                "Ask in English, e.g., What factors influence the incentives GM uses to "
                "promote its products?"
            ),
        )
        company_col, form_col, year_col, submit_col = st.columns(
            [2.4, 2.2, 1.5, 0.7], vertical_alignment="bottom", gap="small"
        )
        with company_col:
            ticker = st.selectbox(
                "Company",
                [
                    None,
                    *sorted(
                        {"ABBV" if t.upper() == "ABBV" else t for t in health["filters"]["ticker"]},
                        key=lambda t: company_name(t).casefold(),
                    ),
                ],
                format_func=company_option,
                help=(
                    "Search one company's reports only. Useful when your question doesn't "
                    "name the company. The code in parentheses is the company's stock ticker."
                ),
            )
        with form_col:
            form = st.selectbox(
                "Report type",
                [None, *REPORT_TYPES],
                format_func=lambda value: "All types" if value is None else REPORT_TYPES[value],
                help=(
                    "U.S. public companies file a 10-K once a year and a shorter 10-Q after "
                    "each of the other three quarters. Annual reports have the most detail."
                ),
            )
        with year_col:
            year = st.selectbox(
                "Filing year",
                [None, *health["filters"]["year"]],
                format_func=lambda value: "All years" if value is None else value,
                help=(
                    "The year the report was filed, not the year it covers. An annual report "
                    "for 2024 is usually filed in early 2025, so choose 2025 to find it. "
                    "The dataset estimates the year for a few reports."
                ),
            )
        with submit_col:
            submitted = st.form_submit_button("Ask", type="primary")
    filters = {}
    if ticker is not None:
        filters["ticker"] = ticker
    if form is not None:
        filters["form"] = form
    if year is not None:
        filters["year"] = year
    if submitted and not question.strip():
        st.session_state["previous_result"] = True
        st.warning("Enter a question first.")
    elif submitted:
        with st.spinner("Searching the reports…"):
            try:
                response = httpx.post(
                    GATEWAY + "/ask", json={"question": question, "filters": filters}, timeout=300
                )
                response.raise_for_status()
                st.session_state["result"] = response.json()
                st.session_state["question"] = question
                st.session_state["filters"] = filters.copy()
                st.session_state["previous_result"] = False
            except httpx.HTTPError:
                st.session_state["previous_result"] = True
                st.error("Something went wrong. Try again in a moment.")
    result = st.session_state.get("result")
    if result:
        if st.session_state.get("previous_result", False):
            st.caption(f"Previous question: {st.session_state['question']}")
        if result["abstained"]:
            st.info(
                "No answer found in these reports. Try rephrasing the question or removing filters."
            )
        else:
            st.markdown(
                linked_answer(result["answer"], result["citations"]), unsafe_allow_html=False
            )
        seconds = result["latencies"]["e2e"]
        with st.container(key="answer_stats"):
            if result["abstained"]:
                st.caption(f"{seconds:.1f} s")
            else:
                count = len(result["citations"])
                st.caption(f"{count} {'source' if count == 1 else 'sources'} · {seconds:.1f} s")
        for citation in result["citations"]:
            number = int(citation["number"])
            with st.expander(
                f":primary-background[**{number}**] " + source_label(citation["sources"], bold=True)
            ):
                st.html(
                    f'<span id="source-{number}"></span>'
                    + (SOURCE_NAVIGATION if number == 1 else ""),
                    unsafe_allow_javascript=number == 1,
                )
                passage_body(citation, result.get("generation", {}).get("context_clipped", False))

with rank_tab:
    result = st.session_state.get("result")
    if result:
        st.caption(
            "How the answer was found.",
            help=(
                "Two retrievers search the passages: BM25 (keyword matching) and dense "
                "retrieval (BGE-small embeddings in Qdrant). Reciprocal rank fusion (RRF) "
                "merges their rankings into one candidate list. A cross-encoder scores each "
                "question–passage pair, and a CatBoost learning-to-rank model produces the "
                "final order from all of these signals. The LLM (Qwen3.5-2B) reads the top "
                "passages and writes the answer."
            ),
        )
        ranking = result["ranking"]
        candidates = pd.DataFrame(ranking["trace"])
        if "ltr_rank" in candidates:
            candidates = candidates.sort_values("ltr_rank", kind="stable")
        if "passage_id" not in candidates:
            candidates["passage_id"] = pd.Series(dtype=str)
        evidence = {item["passage_id"] for item in ranking.get("results", [])}
        candidates["sent_to_model"] = candidates["passage_id"].isin(evidence)
        cited = {item["passage_id"] for item in result["citations"]}
        candidates["cited"] = candidates["passage_id"].isin(cited)
        for key in ["sent_to_model", "cited"]:
            candidates[key] = candidates[key].map({True: "✓", False: ""})
        columns = [key for key, _, _ in CANDIDATE_COLUMNS]
        candidates = candidates.reindex(columns=columns)
        sent_count = len(ranking.get("results", []))
        st.write(
            f"BM25 + dense retrieval → RRF: {len(candidates)} candidates → cross-encoder → "
            f"CatBoost ranker → top {sent_count} → LLM"
        )
        st.subheader(
            "LLM context",
            help=(
                f"The top {sent_count} passages by final rank. Each passage is split into "
                "overlapping windows of up to 448 tokens. The excerpt sent to the LLM is "
                "the window closest to the question by embedding similarity, one per "
                "passage, not the full passage."
            ),
        )
        for rank, passage in enumerate(ranking.get("results", []), 1):
            if passage.get("text"):
                label = (
                    f"**{rank}.** {source_label(passage['sources'], bold=True)} · "
                    f"{passage['passage_id']}"
                )
                with st.expander(label):
                    passage_body(
                        passage, result.get("generation", {}).get("context_clipped", False)
                    )
        with st.container(key="candidate_heading"):
            heading_col, scores_col = st.columns([3, 1], vertical_alignment="center")
            with heading_col:
                st.subheader(
                    "Candidates",
                    help=f"The {len(candidates)} candidates after rank fusion, ordered by final "
                    f"rank. The top {sent_count} went to the LLM.",
                )
            with scores_col:
                show_scores = st.toggle("Show scores", value=False)
        visible_columns = columns if show_scores else columns[:8]
        column_config = {}
        for key, label, help_text in CANDIDATE_COLUMNS:
            if key == "passage_id":
                column = st.column_config.TextColumn(label, help=help_text)
            elif key in {"sent_to_model", "cited"}:
                column = st.column_config.TextColumn(label, help=help_text, alignment="center")
            else:
                column = st.column_config.NumberColumn(
                    label, help=help_text, format="%d" if key.endswith("_rank") else "%.3f"
                )
            column_config[key] = column
        st.dataframe(
            candidates[visible_columns].style.set_properties(
                subset=["sent_to_model", "cited"], color="#1f1f1f", **{"text-align": "center"}
            ),
            hide_index=True,
            width="content",
            height=400,
            column_order=visible_columns,
            column_config=column_config,
        )
        st.subheader(
            "Timing",
            help=(
                "Wall-clock time of each stage for this request. Retrieval total covers "
                "the five stages above it; Total also includes LLM generation."
            ),
        )
        timings = ranking["latencies"]
        steps = [
            ("BM25", timings.get("bm25")),
            ("Query embedding", timings.get("embed")),
            ("Dense retrieval", timings.get("qdrant")),
            ("Cross-encoder", timings.get("ce")),
            ("CatBoost ranker", timings.get("ltr")),
            ("Retrieval total", timings.get("e2e")),
            ("LLM generation", result["latencies"].get("generation")),
            ("Total", result["latencies"].get("e2e")),
        ]
        st.dataframe(
            pd.DataFrame(
                [
                    {"Step": name, "Time": duration(value)}
                    for name, value in steps
                    if value is not None
                ],
                columns=["Step", "Time"],
            ),
            hide_index=True,
            width="content",
            height="content",
        )
        st.subheader(
            "Versions",
            help=(
                "Model versions that served this answer. The numbers are versions in the "
                "local MLflow registry, not model names."
            ),
        )
        versions = ranking.get("versions", {})
        line = versions_line(versions)
        if line:
            st.write(line)
        with st.expander("All versions"):
            st.dataframe(
                versions_table(versions),
                hide_index=True,
                width="content",
                height="content",
                column_config={
                    "Component": st.column_config.TextColumn(width=140),
                    "Version": st.column_config.TextColumn(width=548),
                },
            )
    else:
        st.write("Ask a question first. This tab shows how its answer was found.")
