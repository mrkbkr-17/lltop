"""Machine meters: read from a made-up /proc and /sys tree, and kept by the hub."""

import asyncio

import httpx
import pytest

from conftest import drain
from lltop.hub import TREND_MAX, Hub
from lltop.app import create_app, create_meters_app
from lltop.machine import (Machine, MinuteLog, Remote, clean_reading, collect_forever, parse_machines,
                           parse_nvidia, poll_forever, temperature)

pytestmark = pytest.mark.anyio

MIB = 1024 * 1024


def write(root, path, text):
    f = root / path
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)


def fake_system(root, idle=800, busy=200):
    write(root, "proc/stat", f"cpu  {busy} 0 0 {idle} 0 0 0 0 0 0\ncpu0 1 2 3 4\n")
    write(root, "proc/meminfo", "MemTotal:       16000000 kB\nMemFree:  1 kB\nMemAvailable:    4000000 kB\n")
    write(root, "proc/loadavg", "1.50 1.00 0.50 2/900 12345\n")
    write(root, "sys/class/hwmon/hwmon0/name", "nvme\n")
    write(root, "sys/class/hwmon/hwmon0/temp1_input", "71000\n")
    write(root, "sys/class/hwmon/hwmon1/name", "coretemp\n")
    write(root, "sys/class/hwmon/hwmon1/temp1_input", "48000\n")
    write(root, "sys/class/hwmon/hwmon1/temp2_input", "53500\n")


async def test_reads_cpu_memory_load_and_temperature(tmp_path):
    fake_system(tmp_path)
    m = Machine(tmp_path, nvidia_smi="", disk=str(tmp_path))
    first = await m.read()
    assert first["cpu_pct"] is None                    # needs two readings
    fake_system(tmp_path, idle=850, busy=350)          # 200 more jiffies, 50 of them idle
    ev = await m.read()
    assert ev["type"] == "machine"
    assert ev["cpu_pct"] == 75.0
    assert ev["mem_total"] == 16000000 * 1024
    assert ev["mem_used"] == 12000000 * 1024
    assert ev["load1"] == 1.5
    assert ev["temp_c"] == 53.5                        # the CPU chip, not the hotter NVMe drive
    assert ev["disk_total"] > 0 and ev["gpus"] == []


def test_temperature_falls_back_to_any_sensor(tmp_path):
    write(tmp_path, "sys/class/hwmon/hwmon0/name", "acpitz\n")
    write(tmp_path, "sys/class/hwmon/hwmon0/temp1_input", "44000\n")
    assert temperature(tmp_path) == 44.0
    assert temperature(tmp_path / "nothing") is None


def test_parse_nvidia():
    gpus = parse_nvidia("NVIDIA GeForce RTX 4090, 37, 10240, 24564, 61, 215.33\n"
                        "Tesla T4, [N/A], 100, 15360, 40, [N/A]\nnot a gpu line\n")
    assert gpus[0] == {"name": "NVIDIA GeForce RTX 4090", "util_pct": 37.0, "mem_used": 10240 * MIB,
                       "mem_total": 24564 * MIB, "temp_c": 61.0, "watts": 215.33}
    assert gpus[1]["util_pct"] is None and gpus[1]["watts"] is None and gpus[1]["mem_total"] == 15360 * MIB
    assert len(gpus) == 2


async def test_gpus_from_nvidia_smi(tmp_path):
    fake_system(tmp_path)
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho 'NVIDIA GeForce RTX 4090, 90, 12000, 24000, 70, 300.0'\n")
    smi.chmod(0o755)
    ev = await Machine(tmp_path, nvidia_smi=str(smi), disk=str(tmp_path)).read()
    assert ev["gpus"][0]["util_pct"] == 90.0 and ev["gpus"][0]["mem_used"] == 12000 * MIB


async def test_broken_nvidia_smi_is_ignored(tmp_path):
    fake_system(tmp_path)
    ev = await Machine(tmp_path, nvidia_smi=str(tmp_path / "missing"), disk=str(tmp_path)).read()
    assert ev["gpus"] == [] and ev["mem_total"]


async def test_poller_stops_quietly_without_proc(tmp_path):
    hub = Hub()
    await asyncio.wait_for(poll_forever(Machine(tmp_path, nvidia_smi=""), hub, 0.01), 2)
    assert hub.machines == {} and hub.snapshot()["machines"] == []


async def test_hub_keeps_the_last_reading_and_a_trend(tmp_path):
    fake_system(tmp_path)
    hub = Hub()
    sub = hub.subscribe()
    box = Machine(tmp_path, nvidia_smi="", disk=str(tmp_path))
    task = asyncio.create_task(poll_forever(box, hub, 0.01))
    try:
        async with asyncio.timeout(2):
            while len(hub.machine_trends.get(box.name, [])) < 2:
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
    assert [e["type"] for e in drain(sub)][:2] == ["machine", "machine"]
    snap = hub.snapshot()
    assert snap["machines"][0]["mem_total"] == 16000000 * 1024
    trend = snap["machine_trends"][box.name]
    assert trend[0]["mem"] == 75.0 and "ts" in trend[0]


def test_trend_is_bounded_and_holds_percentages_only():
    hub = Hub()
    for i in range(TREND_MAX + 20):
        hub.publish({"type": "machine", "name": "box", "cpu_pct": 10.0, "mem_used": 1, "mem_total": 4,
                     "gpus": [{"name": "g", "util_pct": 50.0, "mem_used": 1, "mem_total": 2}]})
    assert len(hub.machine_trends["box"]) == TREND_MAX
    assert hub.machine_trends["box"][-1] == {"ts": hub.machines["box"]["ts"], "cpu": 10.0, "mem": 25.0,
                                             "gpus": [{"util": 50.0, "mem": 50.0}]}


def reading(ts, cpu, gpu=None, temp=50.0):
    return {"type": "machine", "ts": ts, "cpu_pct": cpu, "mem_used": 2, "mem_total": 8, "load1": 1.0,
            "temp_c": temp, "disk_used": 1, "disk_total": 4,
            "gpus": [] if gpu is None else [{"name": "RTX", "util_pct": gpu, "mem_used": 1, "mem_total": 4,
                                             "temp_c": 60.0, "watts": 100.0}]}


def test_minute_log_averages_and_keeps_peaks():
    log = MinuteLog()
    assert log.add(reading(120.0, 10.0, gpu=20.0)) is None
    assert log.add(reading(150.0, 30.0, gpu=80.0, temp=71.0)) is None
    row = log.add(reading(181.0, 99.0))                # a new minute: the last one is done
    assert row["minute"] == 120 and row["samples"] == 2 and row["machine"] == ""
    assert row["cpu"] == 20.0 and row["cpu_max"] == 30.0 and row["temp"] == 71.0
    assert row["mem"] == 25.0 and row["mem_total"] == 8 and row["disk"] == 25.0
    assert row["gpus"] == [{"name": "RTX", "util": 50.0, "util_max": 80.0, "mem": 25.0, "temp": 60.0, "watts": 100.0}]
    last = log.flush()                                 # what is left when lltop stops
    assert last["minute"] == 180 and last["cpu"] == 99.0 and last["gpus"] == []
    assert log.flush() is None


async def test_poller_hands_finished_minutes_to_the_history(tmp_path):
    fake_system(tmp_path)

    class Clock(Machine):                              # a reading every "40 seconds"
        t = 1000.0

        async def read(self):
            ev = await super().read()
            Clock.t += 40
            return {**ev, "ts": Clock.t}

    rows = []
    task = asyncio.create_task(poll_forever(Clock(tmp_path, nvidia_smi="", disk=str(tmp_path)), Hub(), 0.01,
                                            MinuteLog(), rows.append))
    try:
        async with asyncio.timeout(2):
            while len(rows) < 2:
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
    assert rows[1]["minute"] == rows[0]["minute"] + 60 and rows[0]["samples"] in (1, 2)


# ------------------------------------------------------------- other machines

def test_parse_machines():
    assert parse_machines(None) == [] and parse_machines(" , ") == []
    assert parse_machines("gpu-box=http://10.0.0.2:11435/, http://10.0.0.3:11435") == [
        ("gpu-box", "http://10.0.0.2:11435"), ("10.0.0.3", "http://10.0.0.3:11435")]
    for bad in ("gpu-box", "ftp://10.0.0.2", "a=http://x, a=http://y", "here=http://x", "<b>=http://x"):
        with pytest.raises(ValueError):
            parse_machines(bad, local="here")


def test_a_fetched_reading_keeps_only_known_numbers():
    ev = clean_reading({"type": "machine", "name": "gpu\nbox" + "x" * 200, "cores": 32, "cpu_pct": 250,
                        "load1": "high", "mem_used": 5, "mem_total": float("nan"), "temp_c": 61.5,
                        "ts": 1.0, "age_s": 2, "extra": "<script>",
                        "gpus": [{"name": "RTX", "util_pct": 40, "mem_used": 1, "mem_total": 2, "watts": True},
                                 "junk"] + [{}] * 40}, "gpu-box")
    assert ev["name"] == "gpu-box" and ev["remote"] is True and len(ev["host"]) == 80 and "\n" not in ev["host"]
    assert ev["cores"] == 32 and ev["cpu_pct"] is None and ev["load1"] is None and ev["mem_total"] is None
    assert ev["temp_c"] == 61.5 and "extra" not in ev and "ts" not in ev
    assert len(ev["gpus"]) == 15 and ev["gpus"][0] == {"name": "RTX", "util_pct": 40.0, "mem_used": 1,
                                                       "mem_total": 2, "temp_c": None, "watts": None}
    for bad in ([], {"type": "snapshot"}, {"type": "machine", "age_s": 300}):
        with pytest.raises(ValueError):
            clean_reading(bad, "gpu-box")


class Fixed:
    """A machine reader with made-up readings."""
    def __init__(self, name, cpu=42.0):
        self.name, self.cpu = name, cpu

    async def read(self):
        return {"type": "machine", "name": self.name, "cores": 8, "cpu_pct": self.cpu, "load1": 1.0,
                "mem_used": 2, "mem_total": 8, "disk_used": 1, "disk_total": 4, "temp_c": 50.0, "gpus": []}


async def test_meters_only_serves_the_last_reading_and_nothing_else():
    app = create_meters_app(poll_interval=0.01, machine=Fixed("gpu-host"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://x") as c:
            async with asyncio.timeout(2):
                while (r := await c.get("/lltop/api/meters")).status_code == 404:
                    await asyncio.sleep(0.01)
            assert r.json()["cpu_pct"] == 42.0 and r.json()["age_s"] < 5
            for path in ("/lltop/", "/lltop/api/state", "/api/chat", "/api/tags"):
                assert (await c.get(path)).status_code == 404


async def test_another_machine_is_collected_shown_and_marked_when_it_stops_answering():
    answering = True

    def gate(request):                                   # the wire to the other machine
        if not answering:
            raise httpx.ConnectError("unreachable")
        return httpx.Response(200, json={"type": "machine", "name": "gpu-host", "cpu_pct": 77.0,
                                         "mem_used": 2, "mem_total": 8, "age_s": 0.5, "gpus": []})

    app = create_app("http://ollama", start_poller=False, machine=Fixed("here"), poll_interval=0.01,
                     machines=[("gpu-box", "http://10.0.0.2:11435")],
                     machines_transport=httpx.MockTransport(gate),
                     transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    hub = app.state.hub
    async with app.router.lifespan_context(app):
        async with asyncio.timeout(2):
            while len(hub.machine_trends.get("gpu-box", [])) < 2 or "here" not in hub.machine_trends:
                await asyncio.sleep(0.01)
        snap = hub.snapshot()
        assert [m["name"] for m in snap["machines"]] == ["here", "gpu-box"]
        assert snap["machines"][1]["host"] == "gpu-host" and snap["machines"][1]["cpu_pct"] == 77.0
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://x") as c:
            mine = (await c.get("/lltop/api/meters")).json()     # only this machine's, never a collected one
            assert mine["name"] == "here" and "remote" not in mine
        answering = False
        async with asyncio.timeout(2):
            while not hub.machines["gpu-box"].get("offline"):
                await asyncio.sleep(0.01)
        assert hub.machines["gpu-box"]["cpu_pct"] == 77.0         # the last reading stays, marked
        answering = True
        async with asyncio.timeout(2):
            while hub.machines["gpu-box"].get("offline"):
                await asyncio.sleep(0.01)


async def test_collector_hands_each_machine_its_own_minutes():
    class Far(Remote):
        t = 1000.0

        async def read(self):
            Far.t += 40
            return {"type": "machine", "name": self.name, "remote": True, "cpu_pct": 5.0, "ts": Far.t, "gpus": []}

    rows = []
    task = asyncio.create_task(collect_forever(Far("gpu-box", "http://x", None), Hub(), 0.01,
                                               MinuteLog("gpu-box"), rows.append))
    try:
        async with asyncio.timeout(2):
            while len(rows) < 2:
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
    assert {r["machine"] for r in rows} == {"gpu-box"} and rows[1]["minute"] == rows[0]["minute"] + 60
