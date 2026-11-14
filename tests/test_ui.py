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
            "ticker": ["GM", "ABBV", "ABbV", "JNJ"],
            "year": ["2024", "2025"],
            "form": ["10-K", "10-Q"],
        }
    }
    source = {"ticker": "GM", "form": "10-K", "year": "2025", "page": "4-5"}
    passage = {"passage_id": "p0", "sources": [source], "text": "Synthetic evidence."}
    state = {
        "fail": False,
        "requests": [],
        "result": {
            "answer": "Revenue rose. [1]",
            "abstained": False,
            "citations": [{"number": 1, **passage}],
            "latencies": {"e2e": 9.74, "generation": 8.0},
            "ranking": {
                "results": [passage],
                "trace": [
                    {"passage_id": "p1", "ltr_rank": 2, "ltr": 0.45678},
                    {"passage_id": "p0", "ltr_rank": 1, "ltr": 0.98765},
                ],
                "latencies": {
                    "bm25": 0.004,
                    "embed": 0.745,
                    "qdrant": 0.2,
                    "ce": 0.5,
                    "ltr": 0.001,
                    "e2e": 1.5,
                    "queue": 0.1,
                    "registry": 0.1,
                    "total": 1.0,
                },
                "versions": {"pipeline": "synthetic"},
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
        "General Motors (GM)",
        "Johnson & Johnson (JNJ)",
    ]
    assert app.text_area[0].label == "Question"
    assert len(app.caption[0].proto.help.split("\n\n")) == 3
    assert [t.label for t in app.tabs] == ["Ask", "Search details"]
    assert app.info[0].value == "Ask a question first. This tab shows how its answer was found."
    app.selectbox[0].select("ABBV").run()
    submit(app, "Synthetic financial question")
    assert not app.exception
    assert state["requests"][-1]["filters"] == {"ticker": "ABBV"}
    captions = [x.value for x in app.caption]
    assert "1 source · 9.7 s" in captions
    assert not any(x.startswith(("Question:", "Previous question:")) for x in captions)
    label = "[1] General Motors · Annual report (10-K) · filed 2025 · p. 4–5"
    assert app.expander[0].label == label
    assert (
        app.expander[1].label
        == "1. General Motors · Annual report (10-K) · filed 2025 · p. 4–5 · p0"
    )
    assert f'[[1]](#source-1 "{label}")' in app.markdown[0].value
    assert not app.markdown[0].proto.allow_html
    assert any(x.value.startswith('<span id="source-1"></span>') for x in app.html)
    candidates, timing = [x.value for x in app.dataframe]
    assert candidates.passage_id.tolist() == ["p0", "p1"]
    assert candidates.sent_to_model.tolist() == [True, False]
    assert candidates.cited.tolist() == [True, False]
    assert not {"ticker", "year", "form"} & set(candidates.columns)
    assert timing.Step.tolist() == [
        "Keyword search",
        "Question embedding",
        "Semantic search",
        "Reranker",
        "Final ranking",
        "Search total",
        "Writing the answer",
        "Total",
    ]
    assert timing.Time.tolist() == [
        "4 ms",
        "745 ms",
        "200 ms",
        "500 ms",
        "1 ms",
        "1.5 s",
        "8.0 s",
        "9.7 s",
    ]


def test_previous_question_survives_failed_and_empty_submit(ui):
    app, state = ui
    submit(app, "First question")
    state["fail"] = True
    submit(app, "Failed question")
    assert app.error[0].value == "Something went wrong. Try again in a moment."
    assert "Previous question: First question" in [x.value for x in app.caption]
    app.selectbox[0].select("GM").run()
    assert len(state["requests"]) == 2
    assert "Filters changed. Click Ask to update the answer." in [x.value for x in app.info]
    submit(app, "")
    assert app.warning[0].value == "Enter a question first."
    assert "Previous question: First question" in [x.value for x in app.caption]
    state["fail"] = False
    submit(app, "New question")
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
    assert app.info[0].value == "Ledgerlight is still starting. Refresh this page in a minute."
    assert not app.text_area
    assert not app.exception
