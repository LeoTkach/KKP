"""Utilities for converting OOXML templates to editable documents."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path


def convert_ooxml_template(
    src: Path,
    dst: Path,
    *,
    content_type_replacements: dict[str, str],
    skip_prefixes: tuple[str, ...] = (),
    text_replacements: dict[str, str] | None = None,
) -> Path:
    text_replacements = text_replacements or {}
    with zipfile.ZipFile(src, "r") as zin:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if any(item.filename.startswith(prefix) for prefix in skip_prefixes):
                    continue
                data = zin.read(item.filename)
                if item.filename.endswith((".xml", ".rels")):
                    text = data.decode("utf-8")
                    for old, new in content_type_replacements.items():
                        text = text.replace(old, new)
                    for old, new in text_replacements.items():
                        text = text.replace(old, new)
                    data = text.encode("utf-8")
                zout.writestr(item, data)
        dst.write_bytes(buffer.getvalue())
    return dst


def word_template_to_docx(
    template: Path,
    output: Path,
    *,
    text_replacements: dict[str, str] | None = None,
) -> Path:
    return convert_ooxml_template(
        template,
        output,
        content_type_replacements={
            "application/vnd.ms-word.template.macroEnabledTemplate.main+xml": (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
            ),
        },
        skip_prefixes=("word/vba",),
        text_replacements=text_replacements,
    )


def ppt_template_to_pptx(
    template: Path,
    output: Path,
    *,
    text_replacements: dict[str, str] | None = None,
) -> Path:
    return convert_ooxml_template(
        template,
        output,
        content_type_replacements={
            "application/vnd.openxmlformats-officedocument.presentationml.template.main+xml": (
                "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
            ),
        },
        text_replacements=text_replacements,
    )
