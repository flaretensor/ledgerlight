import json
import threading

import httpx


def events(url, payload, headers, connected=lambda: True):
    stopped = threading.Event()
    with httpx.stream("POST", url, json=payload, headers=headers, timeout=320) as response:
        response.raise_for_status()

        def watch():
            while not stopped.wait(0.25):
                if not connected():
                    response.close()
                    break

        monitor = threading.Thread(target=watch, daemon=True)
        monitor.start()
        kind = None
        try:
            for line in response.iter_lines():
                if line.startswith("event:"):
                    kind = line[6:].strip()
                elif line.startswith("data:") and kind:
                    yield kind, json.loads(line[5:])
        finally:
            stopped.set()
            monitor.join(timeout=1)
