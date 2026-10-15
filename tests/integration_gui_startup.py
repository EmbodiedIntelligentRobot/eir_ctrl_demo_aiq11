#!/usr/bin/python3
# SPDX-License-Identifier: BSD-3-Clause
"""Opt-in visual simulation process-lifecycle test; no robot is connected."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import yaml


def active(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def children(pid):
    try:
        return [int(value) for value in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except FileNotFoundError:
        return []


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition timed out")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        print("SKIP: visual simulation requires DISPLAY")
        return 77
    source = Path(args.config).resolve()
    config = yaml.safe_load(source.read_text())
    config["robot"] = str((source.parent / config["robot"]).resolve())
    if "flight_recorder" in config:
        config["flight_recorder"] = str((source.parent / config["flight_recorder"]).resolve())
    config["policies"] = {key: str((source.parent / value).resolve())
                          for key, value in config["policies"].items()}
    config["runtime"].update(backend="simulation", model=str(Path(args.model).resolve()),
                             suspend=True, release_after_ready=False)
    environment = dict(os.environ, ROS_LOCALHOST_ONLY="1", ROS_DOMAIN_ID="189",
                       QT_QPA_PLATFORM="offscreen")
    for name, enabled, headless, end_signal in (
            ("default_start_normal_exit", None, False, signal.SIGTERM),
            ("default_start_parent_killed", None, False, signal.SIGKILL),
            ("gui_disabled", False, False, signal.SIGTERM),
            ("headless_skips_gui", True, True, signal.SIGTERM)):
        with tempfile.TemporaryDirectory(prefix="eir-gui-startup-") as directory:
            path = Path(directory) / "config.yaml"
            if enabled is None:
                config["remote"].pop("gui", None)
            else:
                config["remote"]["gui"] = enabled
            config["runtime"]["headless"] = headless
            path.write_text(yaml.safe_dump(config))
            gui_pid = None
            with tempfile.TemporaryFile(mode="w+") as log:
                process = subprocess.Popen([args.executable, str(path)], env=environment,
                                           stdout=log, stderr=subprocess.STDOUT)
                try:
                    def running():
                        assert process.poll() is None, "controller exited during startup"
                        log.seek(0)
                        return "mode=STILL" in log.read()
                    wait_for(running)
                    if enabled is not False and not headless:
                        wait_for(lambda: bool(children(process.pid)))
                        gui_pid = children(process.pid)[0]
                        command = Path(f"/proc/{gui_pid}/cmdline").read_bytes()
                        assert b"remote_gui.py" in command and b"--parent-pid" in command
                        time.sleep(1)
                        assert active(gui_pid), "GUI exited unexpectedly"
                    else:
                        time.sleep(1)
                        assert not children(process.pid), "GUI launched despite skip configuration"
                    process.send_signal(end_signal)
                    process.wait(timeout=10)
                    if gui_pid is not None:
                        wait_for(lambda: not active(gui_pid), timeout=5)
                    print(f"PASS {name}", flush=True)
                except Exception:
                    log.seek(0)
                    print(log.read())
                    raise
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    if gui_pid is not None and active(gui_pid):
                        os.kill(gui_pid, signal.SIGKILL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
