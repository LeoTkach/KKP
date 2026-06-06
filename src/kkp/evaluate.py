"""Точка входа для оценки модели."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from kkp.config import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KKP model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    load_dotenv()
    args = parse_args()

    config = load_config(args.config)
    logger.info("Loaded config from %s", args.config)
    logger.info("Model: %s", config.get("model", {}).get("name"))
    logger.info("Evaluation scaffold ready — implement metrics when model exists.")


if __name__ == "__main__":
    main()
