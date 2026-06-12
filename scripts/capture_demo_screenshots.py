#!/usr/bin/env python3
"""Capture web UI screenshots from the running KKP app at http://127.0.0.1:7860/."""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "report_assets"
DEFAULT_URL = "http://127.0.0.1:7860/"
CHECKPOINTS = (
    ROOT / "outputs" / "ai_generated_efficientnet" / "checkpoints" / "best.pth",
    ROOT / "outputs" / "ai_generated" / "checkpoints" / "best.pth",
)


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _wait_for_server(host: str, port: int, *, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(host, port):
            return True
        time.sleep(0.5)
    return False


def _parse_url(url: str) -> tuple[str, int]:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def _screenshot_site(page, output: Path) -> None:
    page.wait_for_load_state("networkidle", timeout=120_000)
    page.wait_for_timeout(1500)
    page.evaluate("window.scrollTo(0, 0)")
    page.screenshot(path=str(output), full_page=True)


def capture_demo_screenshots(
    output_dir: Path,
    *,
    url: str = DEFAULT_URL,
    connect_only: bool = False,
) -> list[Path]:
    """Capture real UI from the KKP web app (make demo → http://127.0.0.1:7860/)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    host, port = _parse_url(url)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        msg = "Install playwright: uv pip install playwright && uv run playwright install chromium"
        raise RuntimeError(msg) from exc

    proc: subprocess.Popen | None = None
    started_server = False
    if not _port_open(host, port):
        if connect_only:
            msg = (
                f"Server not running at {url}. Start it first: make demo "
                f"(or run capture without --connect-only)"
            )
            raise RuntimeError(msg)
        if not all(path.is_file() for path in CHECKPOINTS):
            msg = "Checkpoints missing. Run: make train && make train-efficientnet"
            raise RuntimeError(msg)
        proc = subprocess.Popen(
            [sys.executable, "-m", "kkp.demo", "--host", host, "--port", str(port)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started_server = True
        if not _wait_for_server(host, port):
            if proc:
                proc.terminate()
            msg = f"Demo server did not start at {url}"
            raise RuntimeError(msg)

    overview = output_dir / "demo_overview.png"
    result = output_dir / "demo_result.png"
    saved: list[Path] = []

    try:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except Exception:
                browser = playwright.chromium.launch(channel="chrome")
            page = browser.new_page(viewport={"width": 1440, "height": 960})
            page.goto(url, wait_until="domcontentloaded", timeout=120_000)
            page.wait_for_selector(".hero-title, h1", timeout=60_000)
            _screenshot_site(page, overview)

            random_btn = page.get_by_role("button", name="Випадкове")
            if random_btn.count():
                random_btn.first.click()
                page.wait_for_timeout(1200)
            analyze_btn = page.get_by_role("button", name="Перевірити")
            if analyze_btn.count():
                analyze_btn.first.click()
                page.wait_for_selector(".results-card:not(.results-card--empty)", timeout=30_000)
                page.wait_for_timeout(800)
            _screenshot_site(page, result)
            browser.close()

        saved.extend([overview, result])
        print(f"Captured live UI from {url}: {overview.name}, {result.name}")
    finally:
        if proc is not None and started_server:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture KKP web UI screenshots for report")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--url", type=str, default=DEFAULT_URL, help="App URL (default: http://127.0.0.1:7860/)"
    )
    parser.add_argument(
        "--connect-only",
        action="store_true",
        help="Only connect to an already running server; do not start make demo",
    )
    args = parser.parse_args()
    capture_demo_screenshots(args.output_dir, url=args.url, connect_only=args.connect_only)


if __name__ == "__main__":
    main()
