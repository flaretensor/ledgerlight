import typer

from ledgerlight import __version__

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main():
    pass


@app.command()
def version():
    typer.echo(__version__)
