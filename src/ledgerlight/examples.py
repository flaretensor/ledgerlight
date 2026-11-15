import re

NUMBER = re.compile(r"(?<!\d)[+-]?\d+(?:(?:,\s*|\s+)\d{3}(?!\d))*(?:\s*\.\s*\d+)?")


def numeric_check(answer, citations):
    answer = re.sub(r"\[\d+\]", "", answer)

    def values(text):
        return {re.sub(r"[\s,]", "", m.group()) for m in NUMBER.finditer(text.replace("−", "-"))}

    expected = values(answer)
    evidence = values("\n".join(w for c in citations for w in c.get("windows", [])))
    return {
        "passed": expected <= evidence,
        "numbers": sorted(expected),
        "missing": sorted(expected - evidence),
    }
