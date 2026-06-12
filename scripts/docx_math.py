"""Insert Word OMML equations via pandoc LaTeX conversion."""

from __future__ import annotations

import re
import subprocess
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree as ET

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.shared import Pt
from docx.text.paragraph import Paragraph
from docx.text.run import Run

MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_INLINE_MATH_SPLIT = re.compile(r"(\$[^$]+\$)")


def _pandoc_docx_root(markdown: str) -> ET.Element:
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "equation.docx"
        subprocess.run(
            ["pandoc", "-f", "markdown", "-t", "docx", "-o", str(output)],
            input=markdown.encode(),
            check=True,
        )
        return ET.fromstring(zipfile.ZipFile(output).read("word/document.xml"))


def latex_to_omath_xml(latex: str) -> str:
    root = _pandoc_docx_root(f"${latex.strip()}$")
    for element in root.iter(f"{{{MATH_NS}}}oMath"):
        ET.register_namespace("m", MATH_NS)
        return ET.tostring(element, encoding="unicode")
    msg = f"No inline equation found in LaTeX: {latex!r}"
    raise ValueError(msg)


def append_mixed_text(
    paragraph: Paragraph,
    text: str,
    *,
    font_callback: Callable[[Run], None],
) -> None:
    """Append plain text and inline $...$ math to an existing paragraph."""
    parts = _INLINE_MATH_SPLIT.split(text)
    for part in parts:
        if not part:
            continue
        if part.startswith("$") and part.endswith("$"):
            paragraph._element.append(parse_xml(latex_to_omath_xml(part[1:-1])))
        else:
            run = paragraph.add_run(part)
            font_callback(run)


def latex_to_omath_para_xml(latex: str) -> str:
    content = latex.strip()
    if not content.startswith("$"):
        content = f"$$\n{content}\n$$"
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "equation.docx"
        subprocess.run(
            ["pandoc", "-f", "latex", "-t", "docx", "-o", str(output)],
            input=content.encode(),
            check=True,
        )
        root = ET.fromstring(zipfile.ZipFile(output).read("word/document.xml"))
        for element in root.iter(f"{{{MATH_NS}}}oMathPara"):
            ET.register_namespace("m", MATH_NS)
            return ET.tostring(element, encoding="unicode")
    msg = f"No equation found in LaTeX: {latex!r}"
    raise ValueError(msg)


def add_math_paragraph(doc: Document, latex: str) -> Paragraph:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = paragraph.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.0
    paragraph._element.append(parse_xml(latex_to_omath_para_xml(latex)))
    return paragraph
