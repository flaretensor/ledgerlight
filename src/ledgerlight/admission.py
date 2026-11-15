import asyncio
import ipaddress
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass


class Rejected(Exception):
    def __init__(self, code):
        self.code = code


@dataclass(eq=False)
class Ticket:
    started: float
    granted: bool = False
    released: bool = False


class Admission:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.condition = threading.Condition()
        self.active = None
        self.queue = deque()
        self.clients = {}

    def _expire(self, now):
        for key in list(self.clients):
            timestamps = self.clients[key]
            while timestamps and timestamps[0] <= now - 60:
                timestamps.popleft()
            if not timestamps:
                del self.clients[key]

    def expire(self):
        with self.condition:
            self._expire(self.clock())

    def submit(self, address):
        with self.condition:
            now = self.clock()
            self._expire(now)
            timestamps = self.clients.setdefault(address, deque())
            if len(timestamps) >= 6:
                raise Rejected("rate")
            timestamps.append(now)
            if self.active is not None and len(self.queue) >= 3:
                raise Rejected("busy")
            ticket = Ticket(now)
            if self.active is None:
                self.active = ticket
                ticket.granted = True
            else:
                self.queue.append(ticket)
            return ticket

    def finish(self, ticket):
        with self.condition:
            if ticket.released:
                return
            ticket.released = True
            if self.active is ticket:
                self.active = None
                while self.queue:
                    following = self.queue.popleft()
                    if not following.released:
                        self.active = following
                        following.granted = True
                        break
            elif ticket in self.queue:
                self.queue.remove(ticket)
            self.condition.notify_all()

    def wait(self, ticket, timeout=60):
        try:
            with self.condition:
                while not ticket.granted:
                    remaining = timeout - (self.clock() - ticket.started)
                    if remaining <= 0:
                        raise Rejected("busy")
                    self.condition.wait(min(remaining, 0.1))
        except BaseException:
            self.finish(ticket)
            raise

    async def wait_async(self, ticket, disconnected, timeout=60):
        try:
            while not ticket.granted:
                if await disconnected():
                    raise asyncio.CancelledError()
                if self.clock() - ticket.started >= timeout:
                    raise Rejected("busy")
                await asyncio.sleep(0.025)
        except BaseException:
            self.finish(ticket)
            raise

    def snapshot(self):
        with self.condition:
            return {"active": self.active is not None, "waiting": len(self.queue)}


def client_address(request):
    peer = request.client.host if request.client else "unknown"
    trusted = {"127.0.0.1", "::1", "testclient"}
    try:
        trusted.update(result[4][0] for result in socket.getaddrinfo("ui", None))
    except socket.gaierror:
        pass
    try:
        with open("/proc/net/route") as stream:
            for line in stream.read().splitlines()[1:]:
                fields = line.split()
                if fields[1] == "00000000":
                    trusted.add(socket.inet_ntoa(bytes.fromhex(fields[2])[::-1]))
    except OSError:
        pass
    if peer in trusted:
        candidate = request.headers.get("CF-Connecting-IP")
        if not candidate:
            candidate = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        try:
            return str(ipaddress.ip_address(candidate))
        except ValueError:
            pass
    return peer
