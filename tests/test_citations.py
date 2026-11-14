import pytest

from ledgerlight.citations import ABSTENTION, issue, render


@pytest.fixture
def issued():
    return issue(
        [
            {"passage_id": name, "text": "Evidence", "windows": ["Evidence"], "sources": []}
            for name in ["p0", "p1"]
        ]
    )


@pytest.mark.parametrize(
    "tail",
    [
        "Citations: * {a} * {b}",
        "Citations:\n* {a}\n- {b}\n• {a}",
        "sources: {a}, {b}",
        "REFERENCES {a}\n{b}",
        "Sources:\n* [1]\n- [2]",
    ],
)
def test_final_citation_block_is_removed_and_unused_sources_are_dropped(issued, tail):
    first, other = issued
    answer, citations = render(
        "Revenue rose. " + first + "\n\n" + tail.format(a=first, b=other), issued
    )
    assert answer == "Revenue rose. [1]"
    assert [item["passage_id"] for item in citations] == ["p0"]


def test_citation_prose_and_nonfinal_labels_are_preserved(issued):
    first, other = issued
    for prose in [
        "Citations of competitors increased this year. " + other,
        "Sources: revenue increased this year. " + other,
        "References " + other + "\nThis is subsequent prose. " + first,
    ]:
        text = "Revenue rose. " + first + "\n" + prose
        answer, citations = render(text, issued)
        assert answer == text.replace(first, "[1]").replace(other, "[2]")
        assert [item["passage_id"] for item in citations] == ["p0", "p1"]


@pytest.mark.parametrize(
    "body",
    ["not found in sources", "  NOT FOUND IN SOURCES.!?  ", "Uncited financial prose."],
)
def test_stripping_cannot_turn_abstention_or_uncited_prose_into_a_cited_answer(issued, body):
    marker = next(iter(issued))
    assert render(body, issued) == (ABSTENTION, [])
    assert render(body + "\nCitations: * " + marker, issued) == (ABSTENTION, [])


@pytest.mark.parametrize("abstention_first", [True, False])
def test_final_abstention_rule_still_applies_after_citation_block_removal(issued, abstention_first):
    first, other = issued
    tails = ["Not found in sources.", "References: " + other]
    if not abstention_first:
        tails.reverse()
    answer, citations = render("Revenue rose. " + first + "\n" + "\n".join(tails), issued)
    assert answer == "Revenue rose. [1]"
    assert [item["passage_id"] for item in citations] == ["p0"]


@pytest.mark.parametrize(
    "suffix",
    [
        "Citations: [[SRC_unknown]]",
        "Sources: [[SRC_broken",
        "References: https://example.com",
        "Citations: <b>source</b>",
    ],
)
def test_invalid_tail_content_is_still_rejected(issued, suffix):
    marker = next(iter(issued))
    assert render("Revenue rose. " + marker + "\n" + suffix, issued) == (ABSTENTION, [])
    assert render("Revenue rose. [1] " + marker, issued) == (ABSTENTION, [])
