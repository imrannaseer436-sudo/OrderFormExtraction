#!/usr/bin/env python3
"""Starts the order-form review UI.

    .venv\\Scripts\\python.exe run_ui.py

Binds 0.0.0.0 so the same server is reachable from a phone on the same
network -- an operator can photograph a form and upload it straight from
the phone's camera, then do the actual review on a desktop where the grid
and the photo fit side by side.
"""

import argparse
import socket

import uvicorn


def lan_ip() -> str:
    """Best-guess LAN address to print. Uses a UDP socket to a public
    address purely to ask the OS which local interface would route there --
    no packet is actually sent, and it works offline on a LAN too."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return "127.0.0.1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0 -- reachable on the LAN)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development only)")
    args = parser.parse_args()

    print("\n  Order Form Review")
    print(f"    this machine : http://localhost:{args.port}")
    if args.host == "0.0.0.0":
        print(f"    phone / LAN  : http://{lan_ip()}:{args.port}")
    print("\n  First upload takes ~15s longer while the OCR models load.\n")

    uvicorn.run("app.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
