"""FastAPI server and JSON API for the KKP demo."""

from __future__ import annotations

import argparse
import io
import logging
import os
import socket
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from kkp.config import load_config
from kkp.demo_engine import (
    DEFAULT_MODEL_CONFIGS,
    DemoEngine,
    build_demo_engine,
    parse_source_json,
)

logger = logging.getLogger(__name__)


def resolve_frontend_dir() -> Path:
    """Locate frontend/ in dev (src tree), Docker (/app/frontend), or via env."""
    if override := os.environ.get("KKP_FRONTEND_DIR"):
        return Path(override)

    here = Path(__file__).resolve().parent
    candidates = (
        Path.cwd() / "frontend",
        here.parents[2] / "frontend",
        here.parents[1] / "frontend",
        Path("/app/frontend"),
    )
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate
    return Path.cwd() / "frontend"


FRONTEND_DIR = resolve_frontend_dir()


@lru_cache(maxsize=1)
def get_engine() -> DemoEngine:
    return build_demo_engine()


def create_app(*, engine: DemoEngine | None = None) -> FastAPI:
    app = FastAPI(title="KKP — AI vs Real", version="0.1.0")
    demo_engine = engine or get_engine()

    @app.get("/api/models")
    def list_models() -> dict:
        return demo_engine.list_models()

    @app.get("/api/random")
    def random_sample() -> dict:
        payload = demo_engine.pick_random_sample()
        if "error" in payload:
            raise HTTPException(status_code=404, detail=payload["error"])
        return payload

    @app.post("/api/analyze")
    async def analyze(
        image: Annotated[UploadFile, File()],
        model: Annotated[str, Form()],
        source: Annotated[str | None, Form()] = None,
    ) -> dict:
        if model not in demo_engine.models:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

        try:
            raw = await image.read()
            pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
            source_data = parse_source_json(source)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid image file") from exc

        try:
            return demo_engine.analyze(pil_image, model, source_data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if FRONTEND_DIR.is_dir() and (FRONTEND_DIR / "index.html").is_file():
        assets_dir = FRONTEND_DIR / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

        @app.get("/styles.css")
        def styles() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")

        @app.get("/app.js")
        def script() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")
    else:
        logger.warning("Frontend not found at %s — only /api/* will work", FRONTEND_DIR)

    return app


def _resolve_model_configs(paths: list[Path] | None) -> dict[str, Path]:
    if not paths:
        return DEFAULT_MODEL_CONFIGS.copy()
    keys = [load_config(path)["model"]["name"] for path in paths]
    return dict(zip(keys, paths, strict=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KKP web demo (FastAPI)")
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        dest="configs",
        help="Model config (repeatable)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def _resolve_server_port(host: str, preferred: int, *, attempts: int = 10) -> int:
    for offset in range(attempts):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    msg = f"No free port in range {preferred}-{preferred + attempts - 1}"
    raise OSError(msg)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    load_dotenv()
    args = parse_args()

    configs = _resolve_model_configs(args.configs)
    engine = build_demo_engine(configs, checkpoint_override=args.checkpoint)
    app = create_app(engine=engine)

    port = _resolve_server_port(args.host, args.port)
    if port != args.port:
        logger.info("Port %d busy, using %d instead", args.port, port)

    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
