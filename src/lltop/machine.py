"""Meters for the machine lltop runs on: CPU, memory, disk, temperature, NVIDIA GPUs.

Read straight from Linux's /proc and /sys (no extra package), GPUs from `nvidia-smi`
when it is installed. Every few seconds a 'machine' event goes to the hub. Anything
that can't be read is left out (None); on a system without /proc (macOS, Windows) the
first reading fails and the poller stops, so the dashboard just shows no machine card.

This is lltop's machine. It is also Ollama's when both run on the same host, which is
the usual setup; with a remote --upstream, start lltop with --no-machine.
"""

import asyncio
import logging
import os
import shutil
import socket
from pathlib import Path

log = logging.getLogger("lltop")

CPU_CHIPS = ("coretemp", "k10temp", "zenpower", "cpu_thermal", "soc_thermal")
GPU_QUERY = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
MIB = 1024 * 1024


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


async def poll_forever(machine, hub, interval: float) -> None:
    try:
        await machine.read()                       # the first reading only sets the CPU baseline
    except Exception as e:
        log.info("machine meters are off: can't read this system (%s: %s)", type(e).__name__, e)
        return
    while True:
        await asyncio.sleep(interval)
        try:
            hub.publish(await machine.read())
        except Exception:
            log.debug("machine: reading failed", exc_info=True)
