from ledgerlight.examples import numeric_check


def test_numeric_proxy_handles_grouping_years_and_citation_numbers():
    citations = [{"windows": ["FY 2024 and 2023: $1, 000 and $31. 9. Loss −2."]}]
    check = numeric_check("FY2024 and 2023: $1,000; $31.9; -2. [1]", citations)
    assert check["passed"]
    assert set(check["numbers"]) == {"2024", "2023", "1000", "31.9", "-2"}
    assert numeric_check("Revenue $1,500 [1]", citations)["missing"] == ["1500"]
