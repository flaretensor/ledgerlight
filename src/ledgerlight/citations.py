import re
import secrets
import unicodedata

ABSTENTION = "not found in sources"
MARKER = re.compile(r"\[\[SRC_[A-Za-z0-9_]+\]\]")
CITATION_TAIL = re.compile(
    r"(?:^|\n)[ \t]*(?i:Citations|Sources|References)\b[ \t]*:?"
    r"(?:" + MARKER.pattern + r"|\[\d+\]|[*\-•,\s])*\Z"
)
URL_OR_HTML = re.compile(
    r"[A-Za-z][A-Za-z0-9+.-]*://|(?:mailto|javascript|data|tel):|www\.|\[[^\]]+\]\([^)]*\)|<\s*/?\s*[A-Za-z][^>]*>",
    re.I,
)


def issue(results):
    issued = {}
    all_text = "\n".join(w for item in results for w in item["windows"])
    for item in results:
        while True:
            marker = "[[SRC_" + secrets.token_hex(12) + "]]"
            if marker not in all_text and marker not in issued:
                break
        issued[marker] = item
    return issued


def is_abstention(text):
    text = text.strip()
    while text and (text[-1].isspace() or unicodedata.category(text[-1]).startswith("P")):
        text = text[:-1]
    return text.casefold() == ABSTENTION


def render(answer, issued):
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S | re.I).strip()
    if "<think>" in answer.casefold() or "</think>" in answer.casefold():
        return ABSTENTION, []
    if URL_OR_HTML.search(answer):
        return ABSTENTION, []
    markers = MARKER.findall(answer)
    if not markers or any(marker not in issued for marker in markers):
        return ABSTENTION, []
    if "[[SRC_" in MARKER.sub("", answer):
        return ABSTENTION, []
    while True:
        previous = answer
        answer = CITATION_TAIL.sub("", answer).rstrip()
        lines = answer.splitlines()
        if lines and is_abstention(lines[-1]) and MARKER.search("\n".join(lines[:-1])):
            lines.pop()
        answer = "\n".join(lines).strip()
        if answer == previous:
            break
    if re.search(r"\[\d+\]", answer):
        return ABSTENTION, []
    markers = MARKER.findall(answer)
    if not markers:
        return ABSTENTION, []
    answer = "\n".join(
        line
        for line in answer.splitlines()
        if not (MARKER.search(line) and not MARKER.sub("", line).strip())
    ).strip()
    if not answer or is_abstention(answer):
        return ABSTENTION, []
    citations, mapping = [], {}
    for marker in markers:
        if marker not in mapping:
            number = len(citations) + 1
            mapping[marker] = number
            item = issued[marker]
            citations.append(
                {
                    "number": number,
                    "passage_id": item["passage_id"],
                    "sources": item["sources"],
                    "windows": item["windows"],
                    "text": item["text"],
                }
            )
    answer = MARKER.sub(lambda m: f"[{mapping[m.group(0)]}]", answer)
    return answer, citations
