import argparse
import asyncio
import os
import socket


async def relay(reader, writer):
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    finally:
        writer.close()


async def handle(reader, writer):
    upstream = None
    try:
        target, upstream = await asyncio.open_connection("ui-ingress", 8501)
        forward = asyncio.create_task(relay(reader, upstream))
        backward = asyncio.create_task(relay(target, writer))
        done, pending = await asyncio.wait([forward, backward], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*done, *pending, return_exceptions=True)
    except OSError:
        pass
    finally:
        writer.close()
        if upstream:
            upstream.close()


async def main():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))
        address = probe.getsockname()[0]
    server = await asyncio.start_server(handle, address, int(os.environ.get("PORT", "8501")))
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    asyncio.run(main())
