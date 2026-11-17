import json
from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src/ledgerlight/ui.py"


@pytest.fixture
def ui(monkeypatch):
    st.cache_data.clear()
    health = {
        "filters": {
            "ticker": ["GM", "ABBV", "ABbV", "JNJ", "EPD"],
            "year": ["2024", "2025"],
            "form": ["10-K", "10-Q"],
        }
    }
    source = {"ticker": "GM", "form": "10-K", "year": "2025", "page": "4-5"}
    passage = {
        "passage_id": "p0",
        "sources": [source],
        "text": "Synthetic evidence. Full passage tail.",
        "windows": ["Synthetic evidence."],
    }
    state = {
        "fail": False,
        "requests": [],
        "result": {
            "answer": "Revenue rose. [1]",
            "abstained": False,
            "citations": [{"number": 1, **passage}],
            "latencies": {"e2e": 9.74, "generation": 8.0},
            "generation": {"context_clipped": False},
            "ranking": {
                "results": [passage],
                "trace": [
                    {"passage_id": "p1", "ltr_rank": 2, "ltr": 0.45678},
                    {"passage_id": "p0", "ltr_rank": 1, "ltr": 0.98765},
                ],
                "latencies": {
                    "bm25": 0.0004,
                    "embed": 0.745,
                    "qdrant": 0.2,
                    "ce": 0.5,
                    "ltr": 0.001,
                    "e2e": 1.5,
                    "queue": 0.1,
                    "registry": 0.1,
                    "total": 1.0,
                },
                "versions": {
                    "pipeline": "3",
                    "method": "CatBoost",
                    "components": {"encoder": "1", "cross_encoder": "2", "ranker": "3"},
                    "collection": "synthetic-collection",
                    "index": "trained",
                    "backend": "ort-cuda",
                    "llm": {
                        "revision": "complete-revision",
                        "sha256": "sha256:" + "a" * 64,
                        "runtime_build": "b12284",
                    },
                    "extra": {"nested": "complete-extra-value"},
                },
            },
        },
    }

    def get(url, **kwargs):
        return httpx.Response(200, json=health, request=httpx.Request("GET", url))

    def post(url, **kwargs):
        state["requests"].append(kwargs["json"])
        return httpx.Response(
            503 if state["fail"] else 200,
            json=state["result"],
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "post", post)
    from ledgerlight import ui_transport

    def stream(url, payload, headers, connected):
        response = post(url, json=payload)
        response.raise_for_status()
        yield "final", response.json()

    monkeypatch.setattr(ui_transport, "events", stream)
    yield AppTest.from_file(str(APP)).run(), state
    st.cache_data.clear()


def submit(app, question):
    app.text_area[0].set_value(question)
    return app.button[0].click().run()


def test_ui_copy_sources_and_detail_tables(ui):
    app, state = ui
    assert not app.exception
    assert [x.label for x in app.selectbox] == ["Company", "Report type", "Filing year"]
    assert app.selectbox[0].options == [
        "All companies",
        "AbbVie (ABBV)",
        "Enterprise Products (EPD)",
        "General Motors (GM)",
        "Johnson & Johnson (JNJ)",
    ]
    assert app.text_area[0].label == "Question"
    assert len(app.caption[0].proto.help.split("\n\n")) == 3
    assert app.caption[0].proto.help.split("\n\n")[1] == (
        "The passages come from FinRank, a public retrieval dataset. They are excerpts of the "
        "reports, so some questions can't be answered here. Each source shows the company, "
        "report type, filing year and page. Text appears as stored, including leftover markup."
    )
    assert app.selectbox[2].proto.help == (
        "The year the report was filed. A 2024 annual report is usually filed in early 2025, "
        "so pick 2025 to find it. The dataset estimates the year for a few reports."
    )
    assert [t.label for t in app.tabs] == ["Ask", "Search details"]
    assert not app.sidebar.selectbox
    assert len(app.get("form")[0].get("selectbox")) == 3
    assert not app.get("form")[0].proto.form.border
    assert "Ask a question first. This tab shows how its answer was found." in [
        x.value for x in app.markdown
    ]
    assert not app.info
    app.selectbox[0].select("ABBV")
    submit(app, "Synthetic financial question")
    assert not app.exception
    assert state["requests"][-1]["filters"] == {"ticker": "ABBV"}
    captions = [x.value for x in app.caption]
    assert "1 source · 9.7 s" in captions
    assert "How the answer was found." in captions
    assert not any(x.startswith(("Question:", "Previous question:")) for x in captions)
    label = "[1] General Motors · Annual report (10-K) · filed 2025 · p. 4–5"
    assert app.expander[0].label == (
        ":primary-background[**1**] **General Motors** · Annual report (10-K) · filed 2025 · p. 4–5"
    )
    assert (
        app.expander[2].label
        == "**1.** **General Motors** · Annual report (10-K) · filed 2025 · p. 4–5 · p0"
    )
    assert app.markdown[0].value == f'Revenue rose.\u00a0[1](#source-1 "{label}")'
    assert not app.markdown[0].proto.allow_html
    assert any(x.value.startswith('<span id="source-1"></span>') for x in app.html)
    candidates, timing = [x.value for x in app.dataframe]
    assert candidates.passage_id.tolist() == ["p0", "p1"]
    assert candidates.sent_to_model.tolist() == ["✓", ""]
    assert candidates.cited.tolist() == ["✓", ""]
    assert len(candidates.columns) == 8
    config = app.dataframe[0].proto.columns
    assert '"label": "CE rank"' in config
    assert '"label": "CE score"' in config
    numeric = json.loads(config)
    assert all(
        numeric[key]["alignment"] == "center"
        for key in [
            "ltr_rank",
            "ce_rank",
            "rrf_rank",
            "bm25_rank",
            "dense_rank",
            "ltr",
            "ce",
            "rrf",
            "bm25",
            "dense",
        ]
    )
    assert not app.toggle[0].value
    assert app.toggle[0].label == "Show scores"
    assert "color: #1f1f1f" in app.dataframe[0].proto.arrow_data.styler.styles
    assert not {"ticker", "year", "form"} & set(candidates.columns)
    assert timing.Step.tolist() == [
        "BM25",
        "Query embedding",
        "Dense retrieval",
        "Cross-encoder",
        "CatBoost ranker",
        "Retrieval total",
        "LLM generation",
        "Total",
    ]
    assert timing.Time.tolist() == [
        "<1 ms",
        "745 ms",
        "200 ms",
        "500 ms",
        "1 ms",
        "1.5 s",
        "8.0 s",
        "9.7 s",
    ]
    assert (
        "BM25 + dense retrieval → RRF: 2 candidates → cross-encoder → CatBoost ranker → top 1 → LLM"
    ) in [x.value for x in app.markdown]
    assert [x.value for x in app.subheader] == ["LLM context", "Candidates", "Timing"]
    assert not any(x.label == "All versions" for x in app.expander)
    assert (
        app.session_state["result"]["ranking"]["versions"] == state["result"]["ranking"]["versions"]
    )
    assert not app.get("json")
    assert [x.value for x in app.text] == [
        "Synthetic evidence.",
        "Synthetic evidence. Full passage tail.",
        "Synthetic evidence.",
        "Synthetic evidence. Full passage tail.",
    ]
    assert [x.value for x in app.markdown].count("**Excerpt sent to the LLM**") == 2
    app.toggle[0].set_value(True).run()
    assert len(app.dataframe[0].value.columns) == 13
    assert len(state["requests"]) == 1


def test_previous_question_survives_failed_and_empty_submit(ui):
    app, state = ui
    submit(app, "First question")
    state["fail"] = True
    submit(app, "Failed question")
    assert app.error[0].value == "Something went wrong. Try again in a moment."
    assert "Previous question: First question" in [x.value for x in app.caption]
    app.selectbox[0].select("GM")
    assert len(state["requests"]) == 2
    assert not app.info
    assert app.session_state["filters"] == {}
    submit(app, "")
    assert app.warning[0].value == "Enter a question first."
    assert "Previous question: First question" in [x.value for x in app.caption]
    state["fail"] = False
    submit(app, "New question")
    assert state["requests"][-1]["filters"] == {"ticker": "GM"}
    assert app.session_state["filters"] == {"ticker": "GM"}
    assert not any(x.value.startswith("Previous question:") for x in app.caption)
    assert not app.exception


def test_abstention_has_one_message_and_startup_failure_has_exact_copy(ui, monkeypatch):
    app, state = ui
    state["result"].update(answer="not found in sources", abstained=True, citations=[])
    submit(app, "Unanswered synthetic question")
    assert [x.value for x in app.info] == [
        "No answer found in these reports. Try rephrasing the question or removing filters."
    ]
    assert "9.7 s" in [x.value for x in app.caption]
    st.cache_data.clear()

    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("Synthetic unavailable service")

    monkeypatch.setattr(httpx, "get", unavailable)
    app = AppTest.from_file(str(APP)).run()
    assert app.info[0].value == "The app is still starting. Refresh this page in a minute."
    assert not app.text_area
    assert not app.exception


@pytest.mark.parametrize("clipped,missing", [(True, False), (False, True)])
def test_passage_falls_back_when_exact_window_is_unavailable(ui, clipped, missing):
    app, state = ui
    state["result"]["generation"]["context_clipped"] = clipped
    if missing:
        for item in [*state["result"]["citations"], *state["result"]["ranking"]["results"]]:
            item.pop("windows", None)
    submit(app, "Synthetic question")
    assert not app.exception
    assert [x.value for x in app.text] == ["Synthetic evidence. Full passage tail."] * 2
    assert not any("Excerpt sent to the LLM" in x.value for x in app.markdown)


def test_citation_spacing_preserves_unknown_citations(ui):
    app, state = ui
    state["result"]["answer"] = "Known [1]. Unknown  [9]. Tab\t[1]. Adjacent[1]."
    submit(app, "Synthetic question")
    answer = app.markdown[0].value
    assert "Known\u00a0[1](#source-1" in answer
    assert "Unknown  [9]." in answer
    assert "Tab\u00a0[1](#source-1" in answer
    assert "Adjacent[1](#source-1" in answer


@pytest.mark.parametrize("punctuation", [".", ",", ";", ":"])
def test_citation_punctuation_and_unknown_markup_are_preserved(ui, punctuation):
    app, state = ui
    state["result"]["answer"] = f"Known [1]{punctuation} Unknown  [9] {punctuation}"
    submit(app, "Synthetic question")
    assert not app.exception
    answer = app.markdown[0].value
    assert "Known\u00a0[1](#source-1" in answer
    assert f'"){punctuation} Unknown  [9] {punctuation}' in answer


def test_source_labels_bold_all_companies_and_keep_model_markup_disabled(ui):
    app, state = ui
    state["result"]["citations"][0]["sources"].append(
        {"ticker": "JNJ", "form": "10-Q", "year": "2024", "page": "9"}
    )
    state["result"]["answer"] = "<script>invalid model markup</script> [1]"
    submit(app, "Synthetic question")
    assert "**General Motors**" in app.expander[0].label
    assert "**Johnson & Johnson**" in app.expander[0].label
    assert not app.markdown[0].proto.allow_html
    copy = " ".join(
        [x.value for x in app.caption]
        + [x.proto.help for x in app.caption]
        + [x.proto.help for x in app.subheader]
    )
    assert "AI model" not in copy
    assert "Ledgerlight" not in copy


def test_examples_submit_and_clear_filters_then_disappear(ui, monkeypatch, tmp_path):
    _, state = ui
    examples = tmp_path / "examples.json"
    examples.write_text(json.dumps([{"question": "Synthetic example question"}]))
    monkeypatch.setenv("LEDGERLIGHT_EXAMPLES_FILE", str(examples))
    app = AppTest.from_file(str(APP)).run()
    app.selectbox[0].select("GM")
    app.selectbox[1].select("10-K")
    app.selectbox[2].select("2025")
    next(
        button for button in app.button if button.label == "Synthetic example question"
    ).click().run()
    assert not app.exception
    assert state["requests"][-1] == {"question": "Synthetic example question", "filters": {}}
    assert [button.label for button in app.button] == ["Ask"]
