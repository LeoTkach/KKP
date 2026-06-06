"""Точка входа для обучения модели."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from kkp.config import load_config
from kkp.data import build_cifake_dataloaders

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KKP model")
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
    logger.info("Project: %s", config.get("project", {}).get("name"))
    logger.info("Task: %s", config.get("data", {}).get("task", "—"))
    logger.info("Model: %s", config.get("model", {}).get("name"))

    if config.get("datasets", {}).get("primary") != "cifake":
        return

    training = config["training"]
    project = config["project"]
    data_dir = Path(config["paths"]["data_dir"])

    loaders = build_cifake_dataloaders(
        data_dir,
        batch_size=training["batch_size"],
        image_size=training["image_size"],
        seed=project["seed"],
        val_fraction=config["data"]["val_split"],
    )

    for split, loader in loaders.items():
        logger.info("%s: %d samples, %d batches", split, len(loader.dataset), len(loader))


if __name__ == "__main__":
    main()
