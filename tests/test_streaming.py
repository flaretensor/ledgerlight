import asyncio

import pytest

from ledgerlight.answer_stream import AnswerPreview
from ledgerlight.citations import ABSTENTION, issue, render


def sources():
    return issue(
        [
            {"passage_id": "p0", "text": "Ten.", "windows": ["Ten."], "sources": []},
            {"passage_id": "p1", "text": "Twenty.", "windows": ["Twenty."], "sources": []},
        ]
    )


def test_stream_marker_splits_and_final_numbering():
    issued = sources()
    first, second = issued
    raw = "Twenty " + second + ", then ten " + first + "."
    for split in range(len(raw) + 1):
        preview = AnswerPreview(issued)
        text = preview.feed(raw[:split]) + preview.feed(raw[split:])
        assert text == render(raw, issued)[0]
        assert "[[" not in text and "SRC_" not in text
    preview = AnswerPreview(issued)
    assert "".join(preview.feed(c) for c in raw) == "Twenty [1], then ten [2]."


def test_stream_holds_abstention_and_hides_reasoning_and_invalid_marker():
    issued = sources()
    for raw in [ABSTENTION, "  NOT FOUND IN SOURCES.", "<think>secret</think>" + ABSTENTION]:
        preview = AnswerPreview(issued)
        assert "".join(preview.feed(c) for c in raw) == ""
    preview = AnswerPreview(issued)
    text = "".join(preview.feed(c) for c in "<think>secret</think>Ten [[SRC_unknown]]")
    assert "secret" not in text and "SRC" not in text
    assert preview.invalid


class Connected:
    async def is_disconnected(self):
        return False


def test_stream_retry_reset_and_final_share_nonstream_parser(monkeypatch):
    from ledgerlight import gateway

    issued = sources()
    raw = "Ten " + next(iter(issued)) + "\nNot found in sources."

    async def fake(client, question, issued, thinking=True):
        yield "content", "Unfinished" if thinking else raw
        yield (
            "complete",
            {
                "output": "Unfinished" if thinking else raw,
                "generation": {"finish_reason": "length" if thinking else "stop"},
            },
        )

    monkeypatch.setattr(gateway, "stream_completion", fake)

    async def run():
        return [event async for event in gateway.answer_events(Connected(), "q", issued, None)]

    events = asyncio.run(run())
    assert [kind for kind, _ in events] == ["delta", "reset", "delta", "parsed"]
    final = events[-1][1]
    assert (final["answer"], final["citations"]) == render(raw, issued)
    assert final["generation"]["retried_without_thinking"]


def test_stream_disconnect_cancels_upstream(monkeypatch):
    from ledgerlight import gateway

    closed = []

    class Disconnected:
        async def is_disconnected(self):
            return True

    async def fake(*args):
        try:
            yield "content", "Ten"
            await asyncio.sleep(60)
        finally:
            closed.append(True)

    monkeypatch.setattr(gateway, "stream_completion", fake)

    async def run():
        with pytest.raises(asyncio.CancelledError):
            async for _ in gateway.answer_events(Disconnected(), "q", sources(), None):
                pass

    asyncio.run(run())
    assert closed == [True]
