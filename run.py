"""Cross-platform launcher for the Graphiti project.

Use this script through uv so dependency resolution and environment management
stay in one place.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Graphiti services through Python.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="chainlit",
        choices=("chainlit", "api"),
        help="Which entrypoint to launch. Default: chainlit",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, help="Port override for the selected mode")
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload for development servers.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    if args.mode == "api":
        port = args.port or 8011
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            args.host,
            "--port",
            str(port),
        ]
        if not args.no_reload:
            command.append("--reload")
        return command

    port = args.port or 8010
    command = [
        sys.executable,
        "-m",
        "chainlit",
        "run",
        "chainlit_app.py",
        "--host",
        args.host,
        "--port",
        str(port),
    ]
    if not args.no_reload:
        command.append("-w")
    return command


def main() -> int:
    args = parse_args()
    command = build_command(args)
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())