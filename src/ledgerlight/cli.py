import typer

from ledgerlight import __version__

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.callback()
def main():
    pass


@app.command()
def version():
    typer.echo(__version__)


@app.command()
def promote(version: int):
    from ledgerlight.registry import promote as promote_model

    promote_model(version)


@app.command()
def rollback():
    from ledgerlight.registry import rollback as rollback_model

    rollback_model()


@app.command()
def start():
    import subprocess
    import sys

    from ledgerlight.common import ROOT

    subprocess.run([sys.executable, "scripts/compose.py", "start"], cwd=ROOT, check=True)


@app.command()
def stop():
    import subprocess
    import sys

    from ledgerlight.common import ROOT

    subprocess.run([sys.executable, "scripts/compose.py", "stop"], cwd=ROOT, check=True)


@app.command()
def health():
    import json

    import httpx

    response = httpx.get("http://127.0.0.1:8000/health", timeout=180)
    response.raise_for_status()
    typer.echo(json.dumps(response.json(), indent=2))


@app.command()
def ask(question: str, ticker: str | None = None, year: str | None = None, form: str | None = None):
    import json

    import httpx

    filters = {k: v for k, v in {"ticker": ticker, "year": year, "form": form}.items() if v}
    response = httpx.post(
        "http://127.0.0.1:8000/ask", json={"question": question, "filters": filters}, timeout=300
    )
    response.raise_for_status()
    typer.echo(json.dumps(response.json(), indent=2))
