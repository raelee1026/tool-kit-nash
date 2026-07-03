from __future__ import annotations

import argparse

from .launcher import launch


def main() -> None:
    parser = argparse.ArgumentParser(description="NASH Eval toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Open the local web visualization.")
    serve.add_argument("results_json", nargs="?", help="Optional completed NASH evaluation result JSON.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--no-open", action="store_true")

    args = parser.parse_args()
    if args.command == "serve":
        launch(args.results_json, host=args.host, port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
