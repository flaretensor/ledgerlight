import json
import time
import uuid

import httpx
from playwright.sync_api import sync_playwright

from ledgerlight.common import path, read_json, read_jsonl, write_json

queries = {q["id"]: q["question"] for q in read_jsonl("data/processed/queries.jsonl")}
dev = read_json("data/processed/splits.json")["dev"]
console_errors = []
deadline = time.monotonic() + 60
while True:
    try:
        ready = httpx.get("http://127.0.0.1:8501/_stcore/health", timeout=2)
        ready.raise_for_status()
        break
    except httpx.HTTPError:
        if time.monotonic() >= deadline:
            raise RuntimeError("UI did not become ready") from None
        time.sleep(1)
with sync_playwright() as playwright:
    profile = path(".tmp/browser-" + uuid.uuid4().hex)
    context = playwright.chromium.launch_persistent_context(
        str(profile),
        headless=True,
        args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"],
        viewport={"width": 1440, "height": 1000},
    )
    page = context.pages[0]
    page.on("pageerror", lambda error: console_errors.append(str(error)))
    page.goto("http://127.0.0.1:8501", wait_until="networkidle", timeout=60000)
    page.get_by_label("Ask about the filings", exact=True).wait_for(timeout=60000)
    attempts = []
    source_count = 0
    for qid in dev[:3]:
        question = queries[qid]
        page.get_by_label("Ask about the filings", exact=True).fill(question)
        page.get_by_role("button", name="Find an answer", exact=True).click()
        page.get_by_role("heading", name=question, exact=True).wait_for(timeout=240000)
        expanders = page.locator('[data-testid="stExpander"]')
        source_count = expanders.count()
        markdown = page.locator('[data-testid="stMarkdownContainer"]').all_text_contents()
        attempts.append(
            {
                "qid": qid,
                "question": question,
                "source_expanders": source_count,
                "markdown": markdown,
            }
        )
        if source_count:
            expanders.first.locator("summary").click()
            assert expanders.first.inner_text().strip()
            break
    if not source_count:
        raise RuntimeError("Three fixed DEV probes abstained; live Sources expansion not verified")
    page.screenshot(path=str(path("logs/ui-dev.png")), full_page=True)
    page.get_by_role("tab", name="How it was ranked", exact=True).click()
    page.locator('[data-testid="stDataFrame"]').first.wait_for(timeout=15000)
    ranking_tables = page.locator('[data-testid="stDataFrame"]').count()
    page.screenshot(path=str(path("logs/ui-ranking-dev.png")), full_page=True)
    page.get_by_role("tab", name="Ask", exact=True).click()
    selected = read_jsonl("data/processed/corpus.jsonl")[0]["sources"][0]
    selected = {key: selected[key] for key in ["ticker", "year", "form"]}
    for key, label in [
        ("ticker", "Company ticker"),
        ("year", "Filing year"),
        ("form", "Filing form"),
    ]:
        page.get_by_role("combobox", name=label, exact=True).click()
        page.get_by_role("option", name=selected[key], exact=True).click()
    page.get_by_text(
        "The result below uses the filters from the last submitted question.", exact=True
    ).wait_for()
    filtered_question = queries[dev[3]]
    page.get_by_label("Ask about the filings", exact=True).fill(filtered_question)
    page.get_by_role("button", name="Find an answer", exact=True).click()
    page.get_by_role("heading", name=filtered_question, exact=True).wait_for(timeout=240000)
    with httpx.Client(timeout=180) as client:
        response = client.post(
            "http://127.0.0.1:8001/search",
            json={"question": filtered_question, "filters": selected},
        )
        response.raise_for_status()
        filtered = response.json()
        assert filtered["results"]
        assert all(
            all(source[key] == value for key, value in selected.items())
            for item in filtered["results"]
            for source in item["sources"]
        )
    result = {
        "browser": "chromium",
        "question_id": qid,
        "question": question,
        "page_title": page.title(),
        "form_submitted": True,
        "source_expanders": source_count,
        "ranking_tables": ranking_tables,
        "attempts": attempts,
        "filters_submitted": selected,
        "filtered_question": filtered_question,
        "filtered_result_count": len(filtered["results"]),
        "filtered_sources_match": True,
        "page_errors": console_errors,
        "screenshots": ["logs/ui-dev.png", "logs/ui-ranking-dev.png"],
    }
    write_json("data/serving/browser.json", result)
    print(json.dumps(result, indent=2))
    context.close()
if console_errors:
    raise SystemExit("Browser reported JavaScript errors")
