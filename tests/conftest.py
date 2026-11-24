import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


def drain(sub) -> list[dict]:
    """All events queued for a hub subscriber so far."""
    events = []
    while not sub.queue.empty():
        events.append(sub.queue.get_nowait())
    return events
