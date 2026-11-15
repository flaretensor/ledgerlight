import re

NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:[, ]\d{3})*(?:\s*\.\s*\d+)?")


def numeric_check(answer, citations):
    answer = re.sub(r"\[\d+\]", "", answer)

    def values(text):
        return {re.sub(r"[\s,]", "", m.group()) for m in NUMBER.finditer(text)}

    expected = values(answer)
    evidence = values("\n".join(w for c in citations for w in c.get("windows", [])))
    return {
        "passed": expected <= evidence,
        "numbers": sorted(expected),
        "missing": sorted(expected - evidence),
    }
