import argparse

import uvicorn

from .api import create_app
from .config import ServerConfig


def main():
    parser = argparse.ArgumentParser(prog="claude-subscription-adapter")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    args = parser.parse_args()
    try:
        config = ServerConfig.from_env()
        if args.host is not None:
            config.host = args.host
        if args.port is not None:
            config.port = args.port
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    uvicorn.run(create_app(config), host=config.host, port=config.port, workers=1)
