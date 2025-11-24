#!/usr/bin/env python3

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Titanic ML Application Launcher")

    parser.add_argument("--port", type=int, default=8501, help="Streamlit server port")
    parser.add_argument(
        "--host", type=str, default="localhost", help="Streamlit server host"
    )

    parser.add_argument("--data-path", type=str, help="Path to Titanic dataset")
    parser.add_argument(
        "--environment",
        type=str,
        choices=["development", "production", "testing"],
        help="Application environment",
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")

    args = parser.parse_args()

    cmd = [
        "streamlit",
        "run",
        "main.py",
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
        "--",
    ]

    if args.data_path:
        cmd.extend(["--data-path", args.data_path])
    if args.environment:
        cmd.extend(["--environment", args.environment])
    if args.config:
        cmd.extend(["--config", args.config])

    if "--cli" in sys.argv:
        cmd.append("--cli")

    print(f"Starting Titanic ML Application on http://{args.host}:{args.port}")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
