import asyncio
import json
import threading
import time

import httpx
from bench_runtime import OUT, mark_step, save, summary
from playwright.sync_api import sync_playwright


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
            headless=True,
            viewport={"width": 1440, "height": 1000},
            args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = context.pages[0]
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
        result["filters_reset"] = all(
            page.get_by_role("combobox", name=label, exact=True).inner_text() == expected
            for label, expected in [
                ("Company", "All companies"),
                ("Report type", "All types"),
                ("Filing year", "All years"),
            ]
        )
        assert result["filters_reset"]
        assert not page.get_by_role("button", name=example["question"], exact=True).count()
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
        context.close()
    save(directory / "result.json", result)
    summary("browser", "checks", {k: v for k, v in result.items() if k != "growth_samples"})
    mark_step(8, "8.7", "DONE", artifacts=["data/analysis/stage2/ui/result.json"])
    print(json.dumps({k: v for k, v in result.items() if k != "growth_samples"}, indent=2))


if __name__ == "__main__":
    main()
