import asyncio
import json
import queue
import threading
import time

import httpx


def events(url, payload, headers, connected=lambda: True):
    inbox = queue.SimpleQueue()
    headers_received = threading.Event()
    stopping = threading.Event()
    state = {}

    async def receive():
        state.update(loop=asyncio.get_running_loop(), task=asyncio.current_task())
        if stopping.is_set():
            return
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(320, connect=5)) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    headers_received.set()
                    if response.is_error:
                        await response.aread()
                    response.raise_for_status()
                    # Local UI phase transition; not an additional server SSE event.
                    inbox.put(("event", ("admitted", {})))
                    kind = None
                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            kind = line[6:].strip()
                        elif line.startswith("data:") and kind:
                            inbox.put(("event", (kind, json.loads(line[5:]))))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            inbox.put(("exception", exc))
        finally:
            inbox.put(("closed", None))

    def worker():
        asyncio.run(receive())

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    started, announced = time.monotonic(), False
    try:
        while connected():
            try:
                kind, value = inbox.get(timeout=0.1)
            except queue.Empty:
                if (
                    not headers_received.is_set()
                    and not announced
                    and time.monotonic() - started > 0.25
                ):
                    announced = True
                    yield "queued", {}
                continue
            if kind == "closed":
                return
            if kind == "exception":
                raise value
            yield value
    finally:
        stopping.set()
        loop, task = state.get("loop"), state.get("task")
        if loop is not None and not loop.is_closed() and task is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass
        thread.join(timeout=2)
