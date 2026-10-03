"""Run the DocOracle server."""

import argparse
from pathlib import Path

import uvicorn


def main():
    """Run the FastAPI server."""
    parser = argparse.ArgumentParser(description="DocOracle Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", default=8000, type=int, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", default=True, help="Enable auto-reload")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Get the directory of this file
    server_dir = Path(__file__).parent

    # Set log level based on debug flag
    log_level = "debug" if args.debug else "info"

    # If debug mode, also enable Python logging
    if args.debug:
        import http.client
        import logging

        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        http.client.HTTPConnection.debuglevel = 1
        logging.getLogger("requests").setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.DEBUG)

    # Run uvicorn
    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=log_level,
        # Use the parent directory so imports work
        app_dir=str(server_dir.parent),
    )


if __name__ == "__main__":
    main()
