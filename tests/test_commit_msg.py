from kkp.commit_msg import validate


def test_valid_feat() -> None:
    assert validate("feat: add data loader")


def test_valid_with_scope() -> None:
    assert validate("fix(data): correct image path")


def test_valid_chore() -> None:
    assert validate("chore: init project")


def test_valid_docs() -> None:
    assert validate("docs: update readme")


def test_invalid_no_type() -> None:
    assert not validate("init project")


def test_invalid_old_format() -> None:
    assert not validate("+ init project")


def test_invalid_trailing_space() -> None:
    assert not validate("feat: trailing ")


def test_merge_commit_ignored() -> None:
    assert validate("Merge branch 'main'")
