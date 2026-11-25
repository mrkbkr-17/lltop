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
    p.add_argument("--history", default=os.environ.get("LLTOP_HISTORY"),
                   help="file that keeps the last 50 requests across restarts (env LLTOP_HISTORY, "
                        "default ~/.local/state/lltop/history.json; holds prompts, mode 0600)")
    p.add_argument("--no-history", action="store_true",
                   help="keep history in memory only")
    p.add_argument("--log-level", default=os.environ.get("LLTOP_LOG_LEVEL", "info"))
    args = p.parse_args()

    import uvicorn

    from .app import create_app
    from .history import default_path

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs every request at INFO, including the /api/ps poll every few seconds.
    logging.getLogger("httpx").setLevel(max(logging.WARNING, logging.getLogger().level))
    history = None if args.no_history else (args.history or default_path())
    app = create_app(args.upstream.rstrip("/"), poll_interval=args.poll, history_path=history)
    # Don't wait for long generations to finish when stopping: after 5 s the connections
    # are closed, running requests are saved as 'interrupted', and lltop exits.
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower(),
                timeout_graceful_shutdown=5)
