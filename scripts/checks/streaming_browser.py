import argparse
import asyncio
import json
import re
import shutil
import threading
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright
from runtime_helpers import OUT, save, summary


def main():
    examples = json.loads((OUT / "examples/candidates.json").read_text())
    example = max(examples, key=lambda item: len(item["answer"]))
    errors = []
    result = {}
    directory = OUT / "ui"
    directory.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(directory / "profile"),
            executable_path=shutil.which("google-chrome"),
            headless=True,
            viewport={"width": 1440, "height": 1000},
            args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = context.pages[0]
        result["browser_executable"] = shutil.which("google-chrome")
        result["browser_version"] = context.browser.version
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "console",
            lambda message: errors.append(message.text) if message.type == "error" else None,
        )
        page.goto("http://127.0.0.1:8501", wait_until="networkidle", timeout=60000)
        page.get_by_label("Question", exact=True).wait_for(timeout=60000)
        result["title"] = page.title()
        assert result["title"] == "Ledgerlight"
        example_button = page.get_by_role("button", name=example["question"], exact=True)
        assert example_button.is_visible()
        result["examples_visible"] = True
        page.get_by_role("combobox", name="Company", exact=True).click()
        page.get_by_role("combobox", name="Company", exact=True).fill("Ford")
        page.get_by_role("option", name="Ford (F)", exact=True).click()
        example_button.click()
        lengths = []
        clock = time.monotonic()
        deadline = clock + 300
        while time.monotonic() < deadline:
            if page.locator(".st-key-answer_text").count():
                break
            preview = page.locator(".st-key-answer_preview")
            if preview.count():
                text = preview.inner_text()
                lengths.append({"seconds": time.monotonic() - clock, "characters": len(text)})
                assert "[[SRC_" not in text
            time.sleep(0.1)
        page.locator(".st-key-answer_text").wait_for(timeout=5000)
        positive = [r["characters"] for r in lengths if r["characters"] > 0]
        assert len(set(positive)) >= 2, "No progressive visible answer growth"
        result["growth_samples"] = lengths
        result["distinct_positive_lengths"] = len(set(positive))
        save(directory / "partial.json", result)
        page.wait_for_function(
            'document.querySelectorAll(\'[role="combobox"][aria-label="Company"]\').length === 1'
        )
        result["filter_widgets"] = {}
        for label, expected in [
            ("Company", "All companies"),
            ("Form type", "All types"),
            ("Filing year", "All years"),
        ]:
            widget = page.get_by_role("combobox", name=label, exact=True)
            parent_text = page.get_by_test_id("stSelectbox").filter(has=widget).inner_text()
            result["filter_widgets"][label] = {
                "input": widget.input_value(),
                "text": parent_text,
                "expected": expected,
                "reset": widget.input_value() in {"", expected} or expected in parent_text,
            }
        result["filters_reset"] = all(v["reset"] for v in result["filter_widgets"].values())
        save(directory / "partial.json", result)
        example_button.wait_for(state="detached")
        page.screenshot(path=str(directory / "answer.png"), full_page=False)
        page.locator('a[href="#source-1"]').first.click()
        page.wait_for_function('document.querySelector("#source-1")?.closest("details")?.open')
        result["citation_1_opened"] = True
        page.screenshot(path=str(directory / "citation.png"), full_page=False)
        page.get_by_text("Which companies are covered?", exact=True).click()
        body = page.locator('[data-testid="stExpander"]').filter(
            has_text="Which companies are covered?"
        )
        assert all(word in body.inner_text() for word in ["Energy:", "Healthcare:", "Automotive:"])
        result["company_expander_opened"] = True
        state = {}
        ready = threading.Event()

        def hold():
            async def requests():
                async with httpx.AsyncClient(timeout=300) as client:

                    async def request(number):
                        async with client.stream(
                            "POST",
                            "http://127.0.0.1:8000/ask/stream",
                            json={"question": example["question"]},
                            headers={"CF-Connecting-IP": f"198.19.20.{number}"},
                        ) as response:
                            async for _ in response.aiter_lines():
                                pass

                    tasks = [asyncio.create_task(request(number)) for number in range(1, 5)]
                    state["loop"] = asyncio.get_running_loop()
                    state["tasks"] = tasks
                    ready.set()
                    await asyncio.gather(*tasks, return_exceptions=True)

            asyncio.run(requests())

        holder = threading.Thread(target=hold)
        holder.start()
        ready.wait(5)
        try:
            for _ in range(100):
                capacity = httpx.get("http://127.0.0.1:8000/capacity", timeout=3).json()
                if capacity["waiting"] == 3:
                    break
                time.sleep(0.05)
            assert capacity["waiting"] == 3, "Could not fill the bounded queue"
            page.get_by_label("Question", exact=True).fill(example["question"])
            page.get_by_role("button", name="Ask", exact=True).click()
            message = "Another question is being answered right now. This demo runs on a single laptop, so please try again in a few seconds."
            page.get_by_text(message, exact=True).wait_for(timeout=10000)
            result["busy_message_visible"] = True
            page.screenshot(path=str(directory / "busy.png"), full_page=False)
        finally:
            for task in state.get("tasks", []):
                state["loop"].call_soon_threadsafe(task.cancel)
            holder.join(timeout=15)
        result["console_errors"] = errors
        assert not errors
        html = httpx.get("http://127.0.0.1:8501/").text
        result["raw_html_title"] = (
            "<title>Ledgerlight</title>" in html and "<title>Streamlit</title>" not in html
        )
        assert result["raw_html_title"]
        icon = re.search(r'<link[^>]+rel="(?:shortcut )?icon"[^>]+href="([^"]+)"', html)
        assert icon is not None
        icon_url = str(httpx.URL("http://127.0.0.1:8501/").join(icon[1]))
        icon_response = httpx.get(icon_url)
        result["raw_html_favicon"] = {
            "href": icon[1],
            "http_status": icon_response.status_code,
            "project_bytes_match": icon_response.content
            == Path("src/ledgerlight/assets/favicon.png").read_bytes(),
        }
        assert (
            icon_response.status_code == 200 and result["raw_html_favicon"]["project_bytes_match"]
        )
        context.close()
    result["passed"] = result["filters_reset"]
    save(directory / "result.json", result)
    summary("browser", "checks", {k: v for k, v in result.items() if k != "growth_samples"})
    print(json.dumps({k: v for k, v in result.items() if k != "growth_samples"}, indent=2))
    if not result["passed"]:
        raise RuntimeError("Browser filter reset check failed; inspect the captured widget state")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
