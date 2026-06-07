"""Build PyTorch DataLoaders from project config."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from kkp.data.loaders import build_cifake_dataloaders, build_split_dataloaders


def build_loaders_from_config(config: dict) -> dict[str, DataLoader]:
    primary = config.get("datasets", {}).get("primary", "cifake")
    training = config["training"]
    project = config["project"]
    data_cfg = config.get("data", {})
    data_dir = Path(config["paths"]["data_dir"])
    min_side = int(data_cfg.get("min_side", 0))

    common = {
        "batch_size": training["batch_size"],
        "image_size": training["image_size"],
        "min_side": min_side,
        "strong_augment": data_cfg.get("strong_augment", False),
    }

    if primary == "split_folder":
        return build_split_dataloaders(data_dir, **common)

    if primary == "cifake":
        extra_train_dir = data_cfg.get("extra_train_dir")
        extra_path = Path(extra_train_dir) if extra_train_dir else None
        supplement_dir_cfg = config.get("paths", {}).get("supplement_dir")
        supplement_path = Path(supplement_dir_cfg) if supplement_dir_cfg else None
        return build_cifake_dataloaders(
            data_dir,
            seed=project["seed"],
            val_fraction=data_cfg["val_split"],
            extra_train_dir=extra_path,
            supplement_dir=supplement_path,
            supplement_max_per_class=data_cfg.get("supplement_max_per_class"),
            **common,
        )

    msg = f"Unsupported dataset primary: {primary!r}"
    raise ValueError(msg)
