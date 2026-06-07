"""Точка входа для обучения модели."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from kkp.config import load_config
from kkp.data import build_loaders_from_config
from kkp.training import train_model

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KKP model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ai_generated.yaml"),
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    load_dotenv()
    args = parse_args()

    config = load_config(args.config)
    logger.info("Loaded config from %s", args.config)
    logger.info("Project: %s", config.get("project", {}).get("name"))
    logger.info("Task: %s", config.get("data", {}).get("task", "—"))
    logger.info("Dataset: %s", config.get("datasets", {}).get("primary", "—"))
    logger.info("Model: %s", config.get("model", {}).get("name"))

    output_dir = Path(config["paths"]["output_dir"])
    loaders = build_loaders_from_config(config)

    for split, loader in loaders.items():
        logger.info("%s: %d samples, %d batches", split, len(loader.dataset), len(loader))

    train_model(config, loaders, output_dir)


if __name__ == "__main__":
    main()
