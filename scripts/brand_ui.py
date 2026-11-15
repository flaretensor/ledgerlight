import re
import shutil
from pathlib import Path

import streamlit


def main():
    root = Path(streamlit.__file__).parent / "static"
    index = root / "index.html"
    html = index.read_text()
    if len(re.findall(r"<title>.*?</title>", html)) != 1:
        raise RuntimeError("Streamlit HTML title layout changed")
    if not re.search(r'<link[^>]+href="\./favicon.png"', html):
        raise RuntimeError("Streamlit HTML favicon layout changed")
    html = re.sub(r"<title>.*?</title>", "<title>Ledgerlight</title>", html)
    index.write_text(html)
    shutil.copyfile(Path("src/ledgerlight/assets/favicon.png"), root / "favicon.png")


if __name__ == "__main__":
    main()
