from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.config.settings import PROJECT_ROOT
from src.utils.dashboard_data import get_overview_payload, run_main_pipeline, search_predictions, static_file_path
from src.utils.live_analysis import append_feedback, predict_live

WEB_ROOT = PROJECT_ROOT / "webapp"


class DashboardHandler(BaseHTTPRequestHandler):
    # -------------------------------------------------------------------------
    # Thin HTTP Adapter for the Portable Dashboard
    # The dashboard server deliberately stays minimal: the scientific logic
    # lives in the model and analysis modules, while this handler is concerned
    # only with translating browser requests into JSON payloads and static
    # assets. That separation keeps the cheminformatics and ML methodology
    # testable independently from the presentation layer.
    # -------------------------------------------------------------------------
    server_version = "FINAL_KV6013Dashboard/1.0"

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        # Serialise API responses explicitly so every dashboard view consumes
        # the same structured evidence objects produced by the backend.
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        # Static assets are served from the packaged web directory so the
        # portable build remains self-contained and deployment-lightweight.
        body = path.read_bytes()
        mime_type, _ = mimetypes.guess_type(path.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        # POST endpoints exchange simple JSON envelopes rather than framework-
        # specific request objects, which keeps the deployment footprint small.
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw_body = self.rfile.read(content_length).decode("utf-8")
        if not raw_body.strip():
            return {}
        return json.loads(raw_body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # Silence default per-request access logging so the server output stays
        # focused on actionable startup or failure information.
        return

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"status": "ok"})
            return
        if parsed.path == "/api/overview":
            # `/api/overview` aggregates the benchmark tables, chart payloads,
            # and dataset summaries that support the analytics pages.
            self._send_json(get_overview_payload())
            return
        if parsed.path == "/api/search":
            # Saved-mode search replays archived external-test predictions so
            # the dissertation evidence remains inspectable molecule-by-molecule.
            query = parse_qs(parsed.query).get("q", [""])[0]
            self._send_json(search_predictions(query))
            return

        try:
            static_path = static_file_path(WEB_ROOT, parsed.path)
            self._send_file(static_path)
        except FileNotFoundError:
            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/run-main":
            # This endpoint allows the interface to refresh benchmark artefacts
            # without exposing the browser directly to the training scripts.
            payload = self._read_json_body()
            result = run_main_pipeline(
                force_rerun=bool(payload.get("force_rerun", False)),
                skip_data_prep=bool(payload.get("skip_data_prep", True)),
            )
            status = HTTPStatus.OK if result["returncode"] == 0 else HTTPStatus.INTERNAL_SERVER_ERROR
            self._send_json(result, status=status)
            return

        payload = self._read_json_body()
        if parsed.path == "/api/predict-live":
            try:
                # Live mode delegates to the route-aware analysis stack, which
                # combines applicability-domain checks, multi-engine inference,
                # and explanation assembly for arbitrary SMILES inputs.
                result = predict_live(str(payload.get("smiles", "")))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:  # pragma: no cover
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(result)
            return

        if parsed.path == "/api/feedback":
            try:
                label = int(payload.get("label"))
                if label not in (0, 1):
                    raise ValueError
            except (TypeError, ValueError):
                self._send_json({"error": "Feedback label must be 0 or 1."}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                # Confirmed labels are appended locally rather than modifying
                # model state immediately. This preserves a human-in-the-loop
                # feedback audit trail suitable for future retraining.
                result = append_feedback(
                    str(payload.get("smiles", "")),
                    label=label,
                    source=str(payload.get("source", "")),
                    note=str(payload.get("note", "")),
                )
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:  # pragma: no cover
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(result, status=HTTPStatus.CREATED)
            return

        self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)


def build_parser() -> argparse.ArgumentParser:
    # -------------------------------------------------------------------------
    # Lightweight Deployment Configuration
    # Host and port are parameterised so the same server can be used for local
    # dissertation demonstrations, bundled desktop execution, or containerised
    # hosting without altering the analytical code.
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Serve the final KV6013 nephrotoxicity predictor dashboard.")
    default_port = int(os.environ.get("PORT", "8000"))
    default_host = os.environ.get("HOST", "0.0.0.0" if "PORT" in os.environ else "127.0.0.1")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    return parser


def main() -> int:
    # -------------------------------------------------------------------------
    # Threaded Service Bootstrap
    # A threaded standard-library server is sufficient here because inference
    # requests are modest in throughput and the project prioritises portability
    # over introducing a heavier web framework dependency.
    # -------------------------------------------------------------------------
    parser = build_parser()
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[SERVE] Dashboard available at http://{args.host}:{args.port}")
    print(f"[SERVE] Project root: {PROJECT_ROOT}")
    print(f"[SERVE] Using web assets from {WEB_ROOT}")
    print("[SERVE] Press Ctrl+C in this terminal to stop the dashboard.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVE] Stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
