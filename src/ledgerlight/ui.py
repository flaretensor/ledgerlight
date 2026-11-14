import os
import re

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
    "EPD": "Enterprise Products Partners",
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
    "Ledgerlight searches 5,230 passages from annual (10-K) and quarterly (10-Q) reports "
    "that 22 U.S. energy, healthcare, and automotive companies filed with the SEC in "
    "2022–2025. The passages come from the FinRank dataset.\n\n"
    "These are excerpts, not complete reports, so some questions can't be answered here. "
    "The dataset has no links to the original filings; each source shows the company, "
    "report type, filing year, and page instead.\n\n"
    "A small AI model writes the answers and can make mistakes. Check the sources "
    "before relying on an answer."
)
SOURCE_NAVIGATION = """<script>
if (!window.ledgerlightSourceNavigation) {
  window.ledgerlightSourceNavigation = true;
  document.addEventListener("click", event => {
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
    if (details && !details.open) details.querySelector("summary")?.click();
    requestAnimationFrame(() => {
      location.hash = url.hash;
      source.scrollIntoView({block: "start"});
    });
  });
}
</script>"""
CANDIDATE_COLUMNS = [
    ("passage_id", "Passage", "Passage ID in the dataset."),
    ("ltr_rank", "Final rank", "Position after all steps. 1 is best."),
    ("sent_to_model", "Used", "The AI model read this passage."),
    ("cited", "Cited", "The answer cites this passage."),
    (
        "ce_rank",
        "Reranker rank",
        "Position according to the reranker, a model that reads the question and the passage together.",
    ),
    ("rrf_rank", "Combined rank", "Position after merging keyword and semantic search."),
    ("bm25_rank", "Keyword rank", "Position in keyword search (BM25)."),
    ("dense_rank", "Semantic rank", "Position in search by meaning (text embeddings)."),
    ("ltr", "Final score", "Score from the final ranking model (CatBoost). Higher is better."),
    ("ce", "Reranker score", "Score from the reranker (cross-encoder). Higher is better."),
    (
        "rrf",
        "Combined score",
        "Reciprocal rank fusion of the keyword and semantic ranks. Higher is better.",
    ),
    ("bm25", "Keyword score", "BM25 score. Higher is better."),
    (
        "dense",
        "Semantic score",
        "Similarity between the question and the passage. Higher is better.",
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


def source_label(sources):
    labels = []
    for source in sources:
        page = re.sub(r"(?<=\d)\s*[-–—]\s*(?=\d)", "–", str(source["page"]))
        labels.append(
            f"{company_name(source['ticker'])} · "
            f"{REPORT_TYPES.get(source['form'], source['form'])} · "
            f"filed {source['year']} · p. {page}"
        )
    return " / ".join(labels)


def linked_answer(answer, citations):
    labels = {str(c["number"]): f"[{c['number']}] " + source_label(c["sources"]) for c in citations}

    def link(match):
        number = match.group(1)
        if number not in labels:
            return match.group(0)
        title = labels[number].replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        return f'[[{number}]](#source-{number} "{title}")'

    return re.sub(r"\[(\d+)\]", link, answer)


def duration(seconds):
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"


st.set_page_config(page_title="Ledgerlight", page_icon="📑", layout="wide")
st.html(
    "<style>"
    '[data-testid="stMainBlockContainer"]{padding-top:2rem;padding-bottom:2rem}'
    '[data-testid="stCaptionContainer"] p,[role="tooltip"] p{color:#5f6368!important}'
    '[data-testid="stTextArea"] textarea::placeholder{color:#80868b;opacity:1}'
    "</style>"
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
    st.info("Ledgerlight is still starting. Refresh this page in a minute.")
    st.stop()

with st.sidebar:
    st.subheader("Filters")
    filters = {}
    tickers = sorted(
        {"ABBV" if ticker.upper() == "ABBV" else ticker for ticker in health["filters"]["ticker"]},
        key=lambda ticker: company_name(ticker).casefold(),
    )
    ticker = st.selectbox(
        "Company",
        [None, *tickers],
        format_func=company_option,
        help="Search one company's reports only. Useful when your question doesn't name the company.",
    )
    if ticker is not None:
        filters["ticker"] = ticker
    form = st.selectbox(
        "Report type",
        [None, *REPORT_TYPES],
        format_func=lambda value: "All types" if value is None else REPORT_TYPES[value],
        help=(
            "U.S. public companies file a 10-K once a year and a shorter 10-Q after each "
            "of the other three quarters. Annual reports have the most detail."
        ),
    )
    if form is not None:
        filters["form"] = form
    year = st.selectbox(
        "Filing year",
        [None, *health["filters"]["year"]],
        format_func=lambda value: "All years" if value is None else value,
        help=(
            "The year the report was filed, not the year it covers. An annual report for "
            "2024 is usually filed in early 2025, so choose 2025 to find it. The dataset "
            "estimates the year for a few reports."
        ),
    )
    if year is not None:
        filters["year"] = year

ask_tab, rank_tab = st.tabs(["Ask", "Search details"])
with ask_tab:
    with st.form("ask_form"):
        question = st.text_area(
            "Question",
            label_visibility="collapsed",
            height=100,
            max_chars=2000,
            placeholder=(
                "Ask in English, e.g., What factors influence the incentives GM uses to "
                "promote its products?"
            ),
        )
        submitted = st.form_submit_button("Ask", type="primary")
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
        if filters != st.session_state["filters"]:
            st.info("Filters changed. Click Ask to update the answer.")
        if result["abstained"]:
            st.info(
                "No answer found in these reports. Try rephrasing the question or removing filters."
            )
        else:
            st.markdown(
                linked_answer(result["answer"], result["citations"]), unsafe_allow_html=False
            )
        seconds = result["latencies"]["e2e"]
        if result["abstained"]:
            st.caption(f"{seconds:.1f} s")
        else:
            count = len(result["citations"])
            st.caption(f"{count} {'source' if count == 1 else 'sources'} · {seconds:.1f} s")
        for citation in result["citations"]:
            number = int(citation["number"])
            with st.expander(f"[{number}] " + source_label(citation["sources"])):
                st.html(
                    f'<span id="source-{number}"></span>'
                    + (SOURCE_NAVIGATION if number == 1 else ""),
                    unsafe_allow_javascript=number == 1,
                )
                st.text(citation["text"])

with rank_tab:
    result = st.session_state.get("result")
    if result:
        st.caption(
            "How the answer to your last question was found.",
            help=(
                "Ledgerlight first picks 100 candidate passages using keyword search and "
                "semantic search (search by meaning). Two ranking models then reorder them. "
                "The AI model reads the top 5 and writes the answer."
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
        columns = [key for key, _, _ in CANDIDATE_COLUMNS]
        candidates = candidates.reindex(columns=columns)
        st.subheader(
            "Passages used for the answer",
            help=(
                "The AI model read one excerpt from each of these passages. "
                "Open a passage to see its full text."
            ),
        )
        for rank, passage in enumerate(ranking.get("results", []), 1):
            if passage.get("text"):
                label = f"{rank}. {source_label(passage['sources'])} · {passage['passage_id']}"
                with st.expander(label):
                    st.text(passage["text"])
        sent_count = len(ranking.get("results", []))
        st.subheader(
            "Candidates",
            help=f"All {len(candidates)} passages the search considered, best first. "
            f"The top {sent_count} were used for the answer.",
        )
        column_config = {}
        for key, label, help_text in CANDIDATE_COLUMNS:
            if key == "passage_id":
                column = st.column_config.TextColumn(label, help=help_text)
            elif key in {"sent_to_model", "cited"}:
                column = st.column_config.CheckboxColumn(label, help=help_text)
            else:
                column = st.column_config.NumberColumn(
                    label, help=help_text, format="%d" if key.endswith("_rank") else "%.3f"
                )
            column_config[key] = column
        st.dataframe(
            candidates,
            hide_index=True,
            width="stretch",
            height=400,
            column_order=columns,
            column_config=column_config,
        )
        st.subheader("Timing")
        timings = ranking["latencies"]
        steps = [
            ("Keyword search", timings.get("bm25")),
            ("Question embedding", timings.get("embed")),
            ("Semantic search", timings.get("qdrant")),
            ("Reranker", timings.get("ce")),
            ("Final ranking", timings.get("ltr")),
            ("Search total", timings.get("e2e")),
            ("Writing the answer", result["latencies"].get("generation")),
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
            width="stretch",
            height="content",
        )
        st.subheader(
            "Versions", help="The model, index, and runtime versions that produced this answer."
        )
        st.json(result["ranking"]["versions"], expanded=False)
    else:
        st.info("Ask a question first. This tab shows how its answer was found.")
