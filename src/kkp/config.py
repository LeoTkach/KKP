"""Загрузка YAML-конфигов с подстановкой переменных окружения."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def _resolve_env(value: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2)
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value
        if default is not None:
            return default
        return match.group(0)

    return _ENV_PATTERN.sub(replacer, value)


def _walk(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _walk(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_walk(item) for item in node]
    if isinstance(node, str):
        return _resolve_env(node)
    return node


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return _walk(raw)
