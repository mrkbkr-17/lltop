"""The streaming reverse proxy.

Every request is forwarded to Ollama unchanged (method, path, query, headers, body).
The response goes back to the caller chunk by chunk, as it arrives: nothing is
buffered. For the inspected endpoints (tap.ENDPOINTS, POST only) a Tap gets a copy of
the request body and of each chunk *after* the caller has been sent it.

Fail open: if creating or feeding the tap fails, the request is unaffected (Tap
methods never raise). Only Ollama itself being unreachable makes the proxy answer
with its own error (502), the same thing a caller would see without the proxy.
"""

import asyncio
import logging

import anyio
import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .tap import ENDPOINTS, Tap

log = logging.getLogger("lltop")

# Headers that describe one connection, not the message: never forwarded (RFC 9110 7.6.1).
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailer", "transfer-encoding", "upgrade"}
# httpx sets these itself for the upstream request.
REQUEST_SKIP = HOP_BY_HOP | {"host", "content-length"}
# The response is re-chunked on the way out, so its length header no longer applies.
RESPONSE_SKIP = HOP_BY_HOP | {"content-length"}


def quietly(tap: Tap | None, method: str, *args) -> None:
    """Call a tap method, ignoring any error. Tap methods already catch their own
    errors; this second net keeps the request safe even from a bug in that."""
    if tap is None:
        return
    try:
        getattr(tap, method)(*args)
    except Exception:
        log.exception("tap.%s failed; the request continues", method)


def make_tap(hub, request: Request) -> Tap | None:
    if request.method != "POST" or request.url.path not in ENDPOINTS:
        return None
    try:
        client = f"{request.client.host}:{request.client.port}" if request.client else None
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client = f"{forwarded} (via {client})"
        return Tap(hub, request.url.path, client)
    except Exception:
        log.exception("could not create a tap; forwarding without inspection")
        return None


async def relay(upstream: httpx.Response, tap: Tap | None):
    """Yield Ollama's response bytes to the caller as they arrive, copying each chunk
    to the tap afterwards. Runs until Ollama finishes, the connection to Ollama breaks,
    or the caller disconnects (then the generator is cancelled or closed)."""
    finished = False
    try:
        async for chunk in upstream.aiter_raw():
            yield chunk
            quietly(tap, "feed", chunk)
        finished = True
        quietly(tap, "finish")
    except httpx.HTTPError as e:
        quietly(tap, "fail", f"connection to Ollama broke mid-response: {type(e).__name__}: {e}")
        raise
    finally:
        if not finished:
            quietly(tap, "cancel")   # no-op if fail() already ended it
        # Closing the upstream response closes the connection, which is how Ollama
        # learns to stop generating. Shielded: we may be here because of cancellation.
        with anyio.CancelScope(shield=True):
            await upstream.aclose()


async def wait_for_disconnect(request: Request) -> None:
    """Returns when the caller hangs up. Only used after the body was read in full,
    so the next ASGI message can only be http.disconnect."""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


async def send_unless_caller_leaves(client: httpx.AsyncClient, upstream_request, request):
    """client.send(), but give up if the caller disconnects while we wait for Ollama's
    first byte (a model load, a long prompt, or a whole non-streaming answer can take
    minutes). Cancelling closes the connection, so Ollama stops too. None = caller left."""
    send = asyncio.ensure_future(client.send(upstream_request, stream=True))
    gone = asyncio.ensure_future(wait_for_disconnect(request))
    try:
        await asyncio.wait({send, gone}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:             # we were cancelled ourselves: don't orphan the send
        send.cancel()
        raise
    finally:
        gone.cancel()
    if send.done():
        return send.result()          # raises httpx errors to the caller of this
    send.cancel()
    try:
        await send
    except BaseException:
        pass
    return None


async def forward(request: Request, client: httpx.AsyncClient, hub):
    tap = make_tap(hub, request)
    headers = [(k, v) for k, v in request.headers.items() if k.lower() not in REQUEST_SKIP]
    if tap:
        body = await request.body()      # small JSON; the tap needs to read it
        quietly(tap, "request", body)
        content = body
    else:
        content = request.stream()       # e.g. model uploads: stream straight through
    url = httpx.URL(path=request.url.path, query=request.url.query.encode("latin-1"))
    upstream_request = client.build_request(request.method, url, headers=headers,
                                            content=content)
    try:
        if tap:
            upstream = await send_unless_caller_leaves(client, upstream_request, request)
            if upstream is None:
                quietly(tap, "cancel")
                return Response(status_code=499)   # nobody is listening any more
        else:
            upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as e:
        message = f"lltop: could not reach Ollama at {client.base_url}: {type(e).__name__}: {e}"
        quietly(tap, "fail", message)
        log.warning(message)
        return JSONResponse({"error": message}, status_code=502)
    quietly(tap, "response", upstream.status_code, upstream.headers)
    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in RESPONSE_SKIP}
    return StreamingResponse(relay(upstream, tap), status_code=upstream.status_code,
                             headers=headers)
