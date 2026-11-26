"""lltop: a streaming proxy in front of Ollama with a live dashboard."""

import argparse
import logging
import os

__version__ = "0.2.0"


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
    p.add_argument("--pg-dsn", default=os.environ.get("LLTOP_PG_DSN"),
                   help="Postgres connection string for the full, searchable history (env "
                        "LLTOP_PG_DSN, e.g. postgresql://lltop@localhost:5432/lltop; put the "
                        "password in ~/.pgpass, not here). Off when unset.")
    p.add_argument("--no-machine", action="store_true",
                   default=os.environ.get("LLTOP_MACHINE", "").lower() in ("off", "0", "no", "false"),
                   help="leave out the machine card (CPU, memory, disk, temperature, GPU of the "
                        "machine lltop runs on; env LLTOP_MACHINE=off). Use it when Ollama "
                        "runs on another machine.")
    p.add_argument("--machines", default=os.environ.get("LLTOP_MACHINES"),
                   help="other machines to show beside this one, comma-separated, each "
                        "NAME=URL or just URL of an lltop there (env LLTOP_MACHINES, e.g. "
                        "gpu-box=http://192.168.1.20:11435). Run `lltop --meters-only` on them.")
    p.add_argument("--meters-only", action="store_true",
                   default=os.environ.get("LLTOP_METERS_ONLY", "").lower() in ("on", "1", "yes", "true"),
                   help="serve only this machine's meters at /lltop/api/meters, for another "
                        "lltop's --machines: no proxy, no dashboard, nothing stored "
                        "(env LLTOP_METERS_ONLY=on)")
    p.add_argument("--app-headers", default=os.environ.get("LLTOP_APP_HEADERS"),
                   help="comma-separated request headers that name the calling app, first found "
                        "wins (env LLTOP_APP_HEADERS; default: x-lltop-app and LiteLLM's "
                        "x-litellm-user_api_key_alias, ..._team_alias, ..._end_user_id)")
    p.add_argument("--log-level", default=os.environ.get("LLTOP_LOG_LEVEL", "info"))
    args = p.parse_args()

    import uvicorn

    from .app import create_app, create_meters_app
    from .history import default_path
    from .machine import parse_machines

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs every request at INFO, including the /api/ps poll every few seconds.
    logging.getLogger("httpx").setLevel(max(logging.WARNING, logging.getLogger().level))
    if args.meters_only:
        uvicorn.run(create_meters_app(poll_interval=args.poll), host=args.host, port=args.port,
                    log_level=args.log_level.lower())
        return
    try:
        import socket
        machines = parse_machines(args.machines, socket.gethostname())
    except ValueError as e:
        p.error(str(e))
    history = None if args.no_history else (args.history or default_path())
    app = create_app(args.upstream.rstrip("/"), poll_interval=args.poll, history_path=history,
                     pg_dsn=args.pg_dsn or None, machine=not args.no_machine, machines=machines,
                     app_headers=tuple(h.strip().lower() for h in args.app_headers.split(",") if h.strip())
                     if args.app_headers else None)
    # Don't wait for long generations to finish when stopping: after 5 s the connections
    # are closed, running requests are saved as 'interrupted', and lltop exits.
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower(),
                timeout_graceful_shutdown=5)
