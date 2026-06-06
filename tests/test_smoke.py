from kkp import __version__


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_load_config(tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "project:\n  seed: 7\npaths:\n  data_dir: ${DATA_DIR:-./data}\n",
        encoding="utf-8",
    )

    from kkp.config import load_config

    config = load_config(config_file)
    assert config["project"]["seed"] == 7
    assert config["paths"]["data_dir"] == "./data"
