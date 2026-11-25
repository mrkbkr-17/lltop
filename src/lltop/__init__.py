"""lltop: a streaming proxy in front of Ollama with a live dashboard."""

import argparse
import logging
import os

__version__ = "0.1.0"


def main() -> None:
    p = argparse.ArgumentParser(
        prog="lltop",
        description="Streaming proxy in front of Ollama with a live dashboard at /lltop/.")
    p.add_argument("--host", default=os.environ.get("LLTOP_HOST", "127.0.0.1"),
                   help="address to listen on (env LLTOP_HOST, default 127.0.0.1; "
                        "0.0.0.0 = the whole LAN, which then sees every prompt)")
    p.add_argument("--port", type=int, default=int(os.environ.get("LLTOP_PORT", "11435")),
                   help="port to listen on (env LLTOP_PORT, default 11435)")
    p.add_argument("--upstream", default=os.environ.get("LLTOP_UPSTREAM", "http://127.0.0.1:11434"),
                   help="Ollama's URL (env LLTOP_UPSTREAM, default http://127.0.0.1:11434)")
    p.add_argument("--poll", type=float, default=float(os.environ.get("LLTOP_POLL", "3")),
                   help="seconds between /api/ps polls (env LLTOP_POLL, default 3)")
    p.add_argument("--log-level", default=os.environ.get("LLTOP_LOG_LEVEL", "info"))
    args = p.parse_args()

    import uvicorn

    from .app import create_app

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(args.upstream.rstrip("/"), poll_interval=args.poll)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
