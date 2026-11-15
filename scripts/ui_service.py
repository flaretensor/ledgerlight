import os
import socket

address = socket.gethostbyname("ui-ingress")
args = [
    "python",
    "-m",
    "streamlit",
    "run",
    "src/ledgerlight/ui.py",
    "--server.address",
    address,
    "--server.port",
    "8501",
    "--browser.gatherUsageStats",
    "false",
]
os.execvpe(args[0], args, os.environ)
