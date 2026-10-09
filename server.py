"""Local server for AI Tic-Tac-Toe: serves the game UI and the Python brain.

    py server.py                 # http://127.0.0.1:8000 and opens the browser
    py server.py --port 9000 --no-browser

Endpoints
    GET  /              the game (web/index.html)
    GET  /api/health    {"ok": true, "engine": "python", "version": "..."}
    POST /api/analyze   body {"game": {...}, "opts": {...}} -> the analysis object

Python 3 standard library only. It listens on 127.0.0.1, so only this
computer can reach it.
"""
import argparse
import json
import math
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from brain import analyze
from brain.agent import ENGINE, ENGINE_VERSION

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_BODY = 64 * 1024


def jsonable(value):
    """JSON has no Infinity, so the open alpha/beta bounds are sent as the
    strings '+inf' and '-inf'. The UI turns them back into numbers."""
    if isinstance(value, float) and math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


class Handler(SimpleHTTPRequestHandler):
    # Windows can map .js to text/plain through the registry; be explicit.
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
                      ".json": "application/json", ".svg": "image/svg+xml", ".md": "text/plain"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == "/api/health":
            self.send_json({"ok": True, "engine": ENGINE, "version": ENGINE_VERSION})
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] != "/api/analyze":
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY:
                raise ValueError("request body is missing or too large")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            analysis = analyze(body.get("game") or {}, body.get("opts") or {})
        except (ValueError, TypeError, AttributeError) as e:
            self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            return
        except Exception as e:  # report instead of dropping the connection
            traceback.print_exc()
            self.send_json({"error": f"internal error: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_json(analysis)

    def send_json(self, obj, status=HTTPStatus.OK):
        data = json.dumps(jsonable(obj), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        if self.command == "POST" or (args and str(args[1])[:1] in "45"):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description="AI Tic-Tac-Toe local server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = parser.parse_args()

    server = None
    for port in range(args.port, args.port + 10):   # use the next free port if needed
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"no free port between {args.port} and {args.port + 9}")

    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"AI Tic-Tac-Toe is running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
