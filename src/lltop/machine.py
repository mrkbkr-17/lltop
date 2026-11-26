"""Meters for the machine lltop runs on: CPU, memory, disk, temperature, NVIDIA GPUs.

With PostgreSQL configured, a row per minute is kept (MinuteLog) for the Machine tab.

Other machines: each runs `lltop --meters-only` (or a whole lltop), which answers
/lltop/api/meters with its last reading; the main lltop is started with --machines and
fetches those (Remote, collect_forever), so one dashboard shows them all.

Read straight from Linux's /proc and /sys (no extra package), GPUs from `nvidia-smi`
when it is installed. Every few seconds a 'machine' event goes to the hub. Anything
that can't be read is left out (None); on a system without /proc (macOS, Windows) the
first reading fails and the poller stops, so the dashboard just shows no machine card.

This is lltop's machine. It is also Ollama's when both run on the same host, which is
the usual setup; with a remote --upstream, start lltop with --no-machine.
"""

import asyncio
import logging
import math
import os
import shutil
import socket
from pathlib import Path
from urllib.parse import urlsplit

log = logging.getLogger("lltop")

CPU_CHIPS = ("coretemp", "k10temp", "zenpower", "cpu_thermal", "soc_thermal")
GPU_QUERY = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
MIB = 1024 * 1024
METERS_PATH = "/lltop/api/meters"
STALE_S = 30                # a reading older than this is not passed on
MAX_GPUS = 16


def cpu_times(root: Path) -> tuple[int, int]:
    """(idle, total) jiffies since boot, from the first line of /proc/stat."""
    fields = (root / "proc/stat").read_text().split("\n", 1)[0].split()
    if not fields or fields[0] != "cpu":
        raise ValueError("unexpected /proc/stat")
    values = [int(v) for v in fields[1:9]]     # user nice system idle iowait irq softirq steal
    return values[3] + values[4], sum(values)


def memory(root: Path) -> tuple[int | None, int | None]:
    """(used, total) bytes; used = total - MemAvailable, what `free` calls used + cache kept."""
    info = {}
    for line in (root / "proc/meminfo").read_text().splitlines():
        name, _, rest = line.partition(":")
        if name in ("MemTotal", "MemAvailable"):
            info[name] = int(rest.split()[0]) * 1024
    total, free = info.get("MemTotal"), info.get("MemAvailable")
    if not total or free is None:
        return None, total
    return total - free, total


def load1(root: Path) -> float | None:
    try:
        return float((root / "proc/loadavg").read_text().split()[0])
    except Exception:
        return None


def temperature(root: Path) -> float | None:
    """The hottest CPU sensor in °C; if no known CPU chip is found, the hottest sensor."""
    cpu, other = [], []
    for chip in sorted((root / "sys/class/hwmon").glob("hwmon*")):
        try:
            name = (chip / "name").read_text().strip()
        except Exception:
            name = ""
        for f in chip.glob("temp*_input"):
            try:
                c = int(f.read_text().strip()) / 1000
            except Exception:
                continue
            if 0 < c < 150:
                (cpu if name in CPU_CHIPS else other).append(c)
    found = cpu or other
    return round(max(found), 1) if found else None


def parse_nvidia(text: str) -> list[dict]:
    """The answer of `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`."""
    def num(s: str) -> float | None:
        try:
            return float(s)
        except ValueError:                      # "[N/A]", "[Not Supported]"
            return None

    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        util, used, total, temp, watts = (num(p) for p in parts[1:])
        gpus.append({"name": parts[0], "util_pct": util,
                     "mem_used": int(used * MIB) if used is not None else None,
                     "mem_total": int(total * MIB) if total is not None else None,
                     "temp_c": temp, "watts": watts})
    return gpus


async def nvidia(binary: str) -> list[dict]:
    proc = await asyncio.create_subprocess_exec(
        binary, f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        async with asyncio.timeout(4):
            out, _ = await proc.communicate()
    except BaseException:
        proc.kill()
        raise
    return parse_nvidia(out.decode(errors="replace")) if proc.returncode == 0 else []


class Machine:
    """Reads the meters. CPU use is the share of time not idle since the previous reading."""

    def __init__(self, root: str | Path = "/", nvidia_smi: str | None = None, disk: str = "/"):
        self.root = Path(root)
        self.disk = disk
        self.nvidia_smi = nvidia_smi if nvidia_smi is not None else (shutil.which("nvidia-smi") or "")
        self.name = socket.gethostname()
        self.cores = os.cpu_count()
        self.last: tuple[int, int] | None = None

    async def read(self) -> dict:
        idle, total = cpu_times(self.root)         # raises where there is no /proc: not supported
        cpu_pct = None
        if self.last and total > self.last[1]:
            d_idle, d_total = idle - self.last[0], total - self.last[1]
            cpu_pct = round(100 * max(0.0, min(1.0, 1 - d_idle / d_total)), 1)
        self.last = (idle, total)
        ev = {"type": "machine", "name": self.name, "cores": self.cores, "cpu_pct": cpu_pct,
              "load1": load1(self.root), "mem_used": None, "mem_total": None,
              "disk_used": None, "disk_total": None, "temp_c": None, "gpus": []}
        try:
            ev["mem_used"], ev["mem_total"] = memory(self.root)
        except Exception:
            log.debug("machine: no memory reading", exc_info=True)
        try:
            usage = shutil.disk_usage(self.disk)
            ev["disk_used"], ev["disk_total"] = usage.used, usage.total
        except Exception:
            log.debug("machine: no disk reading", exc_info=True)
        try:
            ev["temp_c"] = temperature(self.root)
        except Exception:
            log.debug("machine: no temperature reading", exc_info=True)
        if self.nvidia_smi:
            try:
                ev["gpus"] = await nvidia(self.nvidia_smi)
            except Exception:
                log.debug("machine: nvidia-smi failed", exc_info=True)
        return ev


def _avg(values) -> float | None:
    v = [x for x in values if x is not None]
    return round(sum(v) / len(v), 2) if v else None


def _max(values) -> float | None:
    v = [x for x in values if x is not None]
    return max(v) if v else None


def _pct(used, total) -> float | None:
    return 100 * used / total if used is not None and total else None


class MinuteLog:
    """Turns one machine's readings into one row per minute for the history (store.py
    table machine_minutes): averages, and the peak for CPU, GPU use and temperatures."""

    def __init__(self, machine: str = ""):
        self.machine = machine
        self.minute: int | None = None
        self.readings: list[dict] = []

    def add(self, ev: dict) -> dict | None:
        """Add a reading; returns the finished row when a new minute begins."""
        minute = int(ev["ts"] // 60) * 60
        done = self.flush() if self.minute is not None and minute != self.minute else None
        self.minute = minute
        self.readings.append(ev)
        return done

    def flush(self) -> dict | None:
        """The row of the minute so far (None if there are no readings); starts afresh."""
        r, minute = self.readings, self.minute
        self.readings, self.minute = [], None
        if not r:
            return None
        gpus = []
        for i in range(max(len(e.get("gpus") or []) for e in r)):
            g = [e["gpus"][i] for e in r if len(e.get("gpus") or []) > i]
            gpus.append({"name": g[-1].get("name"),
                         "util": _avg(x.get("util_pct") for x in g),
                         "util_max": _max(x.get("util_pct") for x in g),
                         "mem": _avg(_pct(x.get("mem_used"), x.get("mem_total")) for x in g),
                         "temp": _max(x.get("temp_c") for x in g),
                         "watts": _avg(x.get("watts") for x in g)})
        mem_used = _avg(e.get("mem_used") for e in r)
        return {"machine": self.machine, "minute": minute, "samples": len(r),
                "cpu": _avg(e.get("cpu_pct") for e in r), "cpu_max": _max(e.get("cpu_pct") for e in r),
                "mem": _avg(_pct(e.get("mem_used"), e.get("mem_total")) for e in r),
                "mem_used": None if mem_used is None else int(mem_used),
                "mem_total": r[-1].get("mem_total"), "load1": _avg(e.get("load1") for e in r),
                "temp": _max(e.get("temp_c") for e in r),
                "disk": _pct(r[-1].get("disk_used"), r[-1].get("disk_total")), "gpus": gpus}


async def poll_forever(machine, hub, interval: float, minutes: MinuteLog | None = None,
                       sink=None, name: str | None = None) -> None:
    """minutes + sink: each finished minute of readings is handed to sink (the history).
    name: the name the readings go by (default: the one the reader gives)."""
    try:
        await machine.read()                       # the first reading only sets the CPU baseline
    except Exception as e:
        log.info("machine meters are off: can't read this system (%s: %s)", type(e).__name__, e)
        return
    while True:
        await asyncio.sleep(interval)
        try:
            ev = await machine.read()
            if name is not None:
                ev["name"] = name
            hub.publish(ev)                        # sets ev["ts"]
            if minutes is not None and sink is not None:
                row = minutes.add(ev)
                if row:
                    sink(row)
        except Exception:
            log.debug("machine: reading failed", exc_info=True)


# ------------------------------------------------------------- other machines

def parse_machines(text: str | None, local: str = "") -> list[tuple[str, str]]:
    """'gpu-box=http://192.168.1.20:11435, http://192.168.1.21:11435' -> [(name, url)].
    Without a name the address's host is the name. Raises ValueError with a readable reason."""
    found: list[tuple[str, str]] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, url = part.partition("=")
        if not sep or "://" in name:
            name, url = "", part
        url = url.strip().rstrip("/")
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"--machines: {part!r} is not an http(s) address")
        name = name.strip() or parts.hostname
        if len(name) > 40 or not all(c.isalnum() or c in "-_. " for c in name):
            raise ValueError(f"--machines: the name {name!r} must be at most 40 letters, digits, - _ . or spaces")
        if name == local or any(name == n for n, _ in found):
            raise ValueError(f"--machines: the name {name!r} is used twice (this machine is {local!r})")
        found.append((name, url))
    return found


def _num(v, lo: float = 0.0, hi: float = 1e15) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v) if lo <= v <= hi else None


def _whole(v) -> int | None:
    n = _num(v)
    return None if n is None else int(n)


def _text(v, limit: int = 80) -> str:
    return "".join(c for c in v if c.isprintable())[:limit] if isinstance(v, str) else ""


def clean_reading(data, name: str) -> dict:
    """A reading fetched from another machine comes from outside: keep only the fields we
    know, as numbers in range and short plain names. Raises ValueError if it is no reading
    or is old (the other machine's poller stopped)."""
    if not isinstance(data, dict) or data.get("type") != "machine":
        raise ValueError("not a machine reading")
    age = _num(data.get("age_s"), lo=-1e9)
    if age is not None and age > STALE_S:
        raise ValueError(f"its last reading is {int(age)} s old")
    gpus = data.get("gpus") if isinstance(data.get("gpus"), list) else []
    return {"type": "machine", "name": name, "remote": True, "host": _text(data.get("name")),
            "cores": _whole(data.get("cores")), "cpu_pct": _num(data.get("cpu_pct"), hi=100),
            "load1": _num(data.get("load1"), hi=1e6),
            "mem_used": _whole(data.get("mem_used")), "mem_total": _whole(data.get("mem_total")),
            "disk_used": _whole(data.get("disk_used")), "disk_total": _whole(data.get("disk_total")),
            "temp_c": _num(data.get("temp_c"), hi=200),
            "gpus": [{"name": _text(g.get("name")), "util_pct": _num(g.get("util_pct"), hi=100),
                      "mem_used": _whole(g.get("mem_used")), "mem_total": _whole(g.get("mem_total")),
                      "temp_c": _num(g.get("temp_c"), hi=200), "watts": _num(g.get("watts"), hi=1e5)}
                     for g in gpus[:MAX_GPUS] if isinstance(g, dict)]}


class Remote:
    """Another machine's meters, fetched from its lltop (GET only, a short timeout)."""

    def __init__(self, name: str, url: str, client):
        self.name, self.url, self.client = name, url, client

    async def read(self) -> dict:
        r = await self.client.get(self.url + METERS_PATH)
        r.raise_for_status()
        return clean_reading(r.json(), self.name)


async def collect_forever(remote: Remote, hub, interval: float, minutes: MinuteLog | None = None,
                          sink=None) -> None:
    """Keep fetching another machine's readings. While it doesn't answer, the hub is told
    once (the card shows it) and the fetching carries on."""
    up = None
    while True:
        try:
            ev = await remote.read()
            hub.publish(ev)                        # sets ev["ts"]: our clock, not the other machine's
            if up is not True:
                log.info("machine %s: readings arriving from %s", remote.name, remote.url)
            up = True
            if minutes is not None and sink is not None:
                row = minutes.add(ev)
                if row:
                    sink(row)
        except Exception as e:
            if up is not False:
                log.warning("machine %s: no reading from %s (%s: %s)", remote.name, remote.url,
                            type(e).__name__, e)
                hub.publish({"type": "machine", "name": remote.name, "remote": True, "offline": True,
                             "error": type(e).__name__})
            up = False
        await asyncio.sleep(interval)
