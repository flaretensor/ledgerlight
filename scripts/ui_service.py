import argparse
import asyncio
import os
import signal
import socket

from forward_ui import handle


async def main():
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

    process = await asyncio.create_subprocess_exec(*args, env=os.environ)
    loop = asyncio.get_running_loop()
    for signum in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(signum, lambda s=signum: process.send_signal(s))
    server = await asyncio.start_server(handle, socket.gethostbyname("ui-public"), 8501)
    try:
        async with server:
            return await process.wait()
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    raise SystemExit(asyncio.run(main()))
