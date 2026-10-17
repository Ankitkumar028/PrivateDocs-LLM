#!/usr/bin/env python3
"""Script to track billable GPU sessions and estimate spend on AMD Developer Cloud ($1.99/hr)."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

STATE_FILE = Path(".gpu_session_state.json")
LOG_FILE = Path("docs/gpu-session-log.md")
HOURLY_RATE = 1.99


def start_session(purpose: str) -> None:
    if STATE_FILE.exists():
        print(f"[Warning] Session already active: {STATE_FILE.read_text()}")
        sys.exit(1)

    start_time = datetime.datetime.now(datetime.UTC)
    state = {
        "start_time": start_time.isoformat(),
        "purpose": purpose,
        "rate": HOURLY_RATE,
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"Started GPU tracking at {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Purpose: {purpose} | Rate: ${HOURLY_RATE:.2f}/hr")


def stop_session() -> None:
    if not STATE_FILE.exists():
        print("[Error] No active GPU session found to stop.")
        sys.exit(1)

    state = json.loads(STATE_FILE.read_text())
    start_time = datetime.datetime.fromisoformat(state["start_time"])
    end_time = datetime.datetime.now(datetime.UTC)
    duration_hrs = (end_time - start_time).total_seconds() / 3600.0
    cost = duration_hrs * state.get("rate", HOURLY_RATE)

    print(f"Ended GPU tracking at {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Duration: {duration_hrs:.3f} hours ({duration_hrs * 60:.1f} minutes)")
    print(f"Incurred Session Cost: ${cost:.2f}")

    # Append to docs/gpu-session-log.md
    if LOG_FILE.exists():
        content = LOG_FILE.read_text()
        session_row = (
            f"| {end_time.strftime('%Y%m%d-%H%M')} | {start_time.strftime('%Y-%m-%d')} "
            f"| {start_time.strftime('%H:%M')} | {end_time.strftime('%H:%M')} "
            f"| {duration_hrs:.2f} | ${cost:.2f} | {state.get('purpose', '')} | Automated | Pending |\n"
        )
        LOG_FILE.write_text(content + session_row)
        print(f"Logged session entry to {LOG_FILE}")

    STATE_FILE.unlink(missing_ok=True)


def status_session() -> None:
    if not STATE_FILE.exists():
        print("No active GPU session.")
        return

    state = json.loads(STATE_FILE.read_text())
    start_time = datetime.datetime.fromisoformat(state["start_time"])
    now = datetime.datetime.now(datetime.UTC)
    duration_hrs = (now - start_time).total_seconds() / 3600.0
    cost = duration_hrs * state.get("rate", HOURLY_RATE)

    print(f"Active Session: {state.get('purpose')}")
    print(f"Elapsed Time: {duration_hrs:.2f} hrs ({duration_hrs * 60:.1f} mins)")
    print(f"Accrued Cost: ${cost:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Track billable GPU session time.")
    subparsers = parser.add_subparsers(dest="command")

    start_p = subparsers.add_parser("start", help="Start GPU session")
    start_p.add_argument("purpose", nargs="?", default="Experiment Run", help="Session purpose")

    subparsers.add_parser("stop", help="Stop GPU session")
    subparsers.add_parser("status", help="Show current session status")

    args = parser.parse_args()
    if args.command == "start":
        start_session(args.purpose)
    elif args.command == "stop":
        stop_session()
    elif args.command == "status":
        status_session()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
