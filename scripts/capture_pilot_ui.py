import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/pilot/ui-audit"
IDS = {
    "answered": "02ecc8d1-7e0d-4126-bef1-246f49681afe",
    "abstained": "06fc84ba-d32f-407b-96ac-bd661e20ede9",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "open",
            "close",
            "start",
            *IDS,
            "filtered",
            "ranking",
            "versions",
            "source",
            "error",
        ],
    )
    parser.add_argument("--session", default="ledgerlight-validation")
    parser.add_argument("--capture-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    def cli(*command):
        result = subprocess.run(
            ["playwright-cli", "-s=" + args.session, "--raw", *command],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        with (OUTPUT / "capture.log").open("a") as stream:
            stream.write(json.dumps(list(command)) + "\n" + result.stdout + result.stderr + "\n")
        if result.returncode or "### Error" in result.stdout:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout.strip()

    def js(code):
        return cli("run-code", "async page => { " + code + " }")

    def capture(state, segments=True):
        for width, height in [(1440, 1000), (390, 844)]:
            prefix = "data/pilot/ui-audit/" + state + "-" + str(width)
            metrics = js(
                f"await page.setViewportSize({{width:{width},height:{height}}});"
                "await page.waitForTimeout(150);"
                "const main=page.getByRole('main');"
                "const size=await main.evaluate(e=>({height:e.scrollHeight,view:e.clientHeight,"
                "width:e.clientWidth,scrollWidth:e.scrollWidth}));"
                f"const count={str(segments).lower()}?1+Math.ceil(Math.max(0,size.height-size.view)/Math.max(1,size.view-70)):1;"
                "for(let i=0;i<count;i++){"
                "await main.evaluate((e,y)=>{e.scrollTop=y;},i*Math.max(1,size.view-70));"
                "await page.waitForTimeout(80);"
                f"await page.screenshot({{path:{json.dumps(prefix)}+'-'+String(i+1).padStart(2,'0')+'.png',scale:'css'}});"
                "}"
                "await main.evaluate(e=>{e.scrollTop=0;});"
                "return {...size,count,bodyWidth:await page.evaluate(()=>document.documentElement.scrollWidth)};"
            )
            cli("snapshot", "--filename=" + prefix + ".yaml")
            print(json.dumps({"state": state, "width": width, "geometry": json.loads(metrics)}))

    if args.action in {"open", "close"}:
        cli(args.action, *(["about:blank"] if args.action == "open" else []))
        return
    if args.capture_only:
        print(js("return await page.evaluate(()=>window.__llAudit || null);"))
        capture(args.action)
        return
    if args.action == "start":
        cli("goto", "http://127.0.0.1:8501")
        js("await page.getByRole('button',{name:'Ask',exact:true}).waitFor();")
        capture("empty")
        return
    if args.action == "ranking":
        js("await page.getByRole('tab',{name:'Search details',exact:true}).click();")
        capture("ranking")
        return
    if args.action == "versions":
        js(
            "await page.getByRole('tab',{name:'Search details',exact:true}).click();"
            "await page.getByText('Versions',{exact:true}).scrollIntoViewIfNeeded();"
            "const tree=page.locator('[data-testid=stJson]');"
            "while(await tree.locator('.collapsed-icon').count())"
            "await tree.locator('.collapsed-icon').first().click();"
        )
        capture("versions")
        return

    if args.action == "source":
        js(
            "await page.getByRole('tab',{name:'Search details',exact:true}).click();"
            "await page.locator('[data-testid=stExpander] summary').first().click();"
        )
        capture("source")
        return

    qid = IDS[
        "abstained"
        if args.action == "error"
        else "answered"
        if args.action == "filtered"
        else args.action
    ]
    splits = json.loads((ROOT / "data/processed/splits.json").read_text())
    if qid not in splits["dev"]:
        raise ValueError("Capture question must belong to DEV")
    question = None
    for line in (ROOT / "data/processed/queries.jsonl").open():
        row = json.loads(line)
        if row["id"] == qid:
            question = row["question"]
            break
    if question is None:
        raise ValueError("Missing DEV question")
    js("await page.setViewportSize({width:1440,height:1000});")
    js("await page.getByRole('tab',{name:'Ask',exact:true}).click();")
    if args.action == "filtered":
        js(
            "await page.getByRole('combobox',{name:'Company',exact:true}).click();"
            "await page.getByRole('option',{name:'General Motors (GM)',exact:true}).click();"
        )
        capture("filter-pending")
    js(
        "await page.getByRole('textbox',{name:'Question',exact:true}).fill("
        + json.dumps(question)
        + ");await page.evaluate(()=>{window.__llAudit={start:performance.now(),seen:false,end:null};"
        "const observer=new MutationObserver(()=>{const a=window.__llAudit;"
        "const loading=document.body.innerText.includes('Searching the reports…');"
        "const failed=document.body.innerText.includes('Something went wrong. Try again in a moment.');"
        "if(loading)a.seen=true;if(((a.seen&&!loading)||failed)&&a.end===null){a.end=performance.now();observer.disconnect();}});"
        "observer.observe(document.body,{subtree:true,childList:true,characterData:true});});"
        "await page.getByRole('button',{name:'Ask',exact:true}).click();"
    )
    if args.action == "answered":
        js("await page.getByText('Searching the reports…',{exact:true}).waitFor();")
        capture("loading", segments=False)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        timing = json.loads(js("return await page.evaluate(()=>window.__llAudit);"))
        if timing["end"] is not None:
            break
        time.sleep(0.2)
    else:
        raise TimeoutError("No rendered response within 180 seconds")
    print(
        json.dumps(
            {"action": args.action, "rendered_seconds": (timing["end"] - timing["start"]) / 1000}
        )
    )
    capture(args.action)


if __name__ == "__main__":
    main()
