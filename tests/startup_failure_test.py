#!/usr/bin/python3
# SPDX-License-Identifier: BSD-3-Clause
"""Reject an invalid ROS log directory without crashing during cleanup."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="eir-startup-failure-") as directory:
        log_path = Path(directory) / "log-file"
        log_path.write_text("This is a file, not a log directory.\n")
        env = os.environ.copy()
        env["ROS_LOG_DIR"] = str(log_path)
        result = subprocess.run(
            [args.executable], env=env, capture_output=True, text=True, timeout=15
        )
        assert result.returncode == 1, (
            f"Expected a startup error (1), got {result.returncode}:\n"
            f"{result.stdout}\n{result.stderr}"
        )
        assert str(log_path) in result.stderr, result.stderr


if __name__ == "__main__":
    main()
