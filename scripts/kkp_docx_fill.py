"""Fill official KKP Word template with project-specific data."""
# ruff: noqa: E501

from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_UNDERLINE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from docx_math import add_math_paragraph, append_mixed_text
from kkp_meta import KkpMeta
from kkp_report_content import (
    CodeBlock,
    FigureBlock,
    MathBlock,
    TableBlock,
    conclusions,
    intro_paragraphs,
    references,
    report_sections,
)

DEFAULT_FONT_SIZE = Pt(14)


_UA_MONTHS = {
    "01": "січня",
    "02": "лютого",
    "03": "березня",
    "04": "квітня",
    "05": "травня",
    "06": "червня",
    "07": "липня",
    "08": "серпня",
    "09": "вересня",
    "10": "жовтня",
    "11": "листопада",
    "12": "грудня",
}


def _normalize(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u2018", "'").replace("\xa0", " ").strip()


def _set_paragraph_text(paragraph: Paragraph, text: str) -> None:
    if paragraph.runs:
        paragraph.runs[0].text = text
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(text)


def _delete_paragraph(paragraph: Paragraph) -> None:
    element = paragraph._element
    parent = element.getparent()
    if parent is not None:
        parent.remove(element)


def _delete_table(table: Table) -> None:
    element = table._element
    parent = element.getparent()
    if parent is not None:
        parent.remove(element)


def _delete_paragraphs(doc: Document, start: int, end: int | None = None) -> None:
    paragraphs = doc.paragraphs
    end_index = len(paragraphs) if end is None else end
    for paragraph in paragraphs[start:end_index][::-1]:
        _delete_paragraph(paragraph)


def _find_paragraph_index(doc: Document, prefix: str, *, start: int = 0) -> int | None:
    needle = _normalize(prefix)
    for index, paragraph in enumerate(doc.paragraphs[start:], start=start):
        if _normalize(paragraph.text).startswith(needle):
            return index
    return None


def _insert_paragraph_after(paragraph: Paragraph, text: str) -> Paragraph:
    new_element = OxmlElement("w:p")
    paragraph._element.addnext(new_element)
    new_paragraph = Paragraph(new_element, paragraph._parent)
    if text:
        new_paragraph.add_run(text)
    return new_paragraph


def _insert_paragraph_before(paragraph: Paragraph, text: str = "") -> Paragraph:
    new_element = OxmlElement("w:p")
    paragraph._element.addprevious(new_element)
    new_paragraph = Paragraph(new_element, paragraph._parent)
    if text:
        new_paragraph.add_run(text)
    return new_paragraph


def _format_ua_date(iso_date: str) -> str:
    day, month, year = iso_date.split(".")
    month_name = _UA_MONTHS.get(month, month)
    return f"„{day}” {month_name} {year}"


def _is_underlined(run) -> bool:
    rpr = run._element.rPr
    return rpr is not None and rpr.u is not None


def _clear_run_highlight(run: Run) -> None:
    rpr = run._element.rPr
    if rpr is None:
        return
    highlight = rpr.find(qn("w:highlight"))
    if highlight is not None:
        rpr.remove(highlight)
    shading = rpr.find(qn("w:shd"))
    if shading is not None:
        fill = shading.get(qn("w:fill"), "").upper()
        if fill in {"FFFF00", "YELLOW", "00FFFF"}:
            rpr.remove(shading)


def _ensure_font(run, *, size: Pt = DEFAULT_FONT_SIZE) -> None:
    run.font.name = "Times New Roman"
    run.font.size = size
    rpr = run._element.get_or_add_rPr()
    rpr.rFonts.set(qn("w:ascii"), "Times New Roman")
    rpr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rpr.rFonts.set(qn("w:cs"), "Times New Roman")
    rpr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    _clear_run_highlight(run)


def _set_run_text(run, text: str, *, size: Pt = DEFAULT_FONT_SIZE) -> None:
    run.text = text
    _ensure_font(run, size=size)


def _replace_underlined_span(paragraph: Paragraph, new_text: str) -> None:
    """Put new_text into the first underlined run that contains letters; clear other letter runs."""
    underlined = [r for r in paragraph.runs if _is_underlined(r)]
    if not underlined:
        return
    letter_runs = [
        r for r in underlined if any(ch.isalpha() or ch in "А-Яа-яІіЇїЄєҐґ" for ch in r.text)
    ]
    if not letter_runs:
        letter_runs = underlined
    _set_run_text(letter_runs[0], new_text)
    for run in letter_runs[1:]:
        _set_run_text(run, "")


def _replace_underlined_group(paragraph: Paragraph, new_text: str) -> None:
    """Replace consecutive underlined runs that form one editable field."""
    runs = paragraph.runs
    groups: list[list] = []
    current: list = []
    for run in runs:
        if _is_underlined(run):
            current.append(run)
        else:
            if current:
                groups.append(current)
                current = []
    if current:
        groups.append(current)

    for group in groups:
        combined = "".join(r.text for r in group)
        if not any(ch.isalpha() for ch in combined):
            continue
        _set_run_text(group[0], new_text)
        for run in group[1:]:
            _set_run_text(run, "")
        return


def _fill_course_group_line(paragraph: Paragraph, meta: KkpMeta) -> None:
    course_done = False
    group_done = False
    for run in paragraph.runs:
        if not _is_underlined(run):
            continue
        text = run.text.strip()
        if not course_done and text.isdigit():
            _set_run_text(run, meta.course)
            course_done = True
            continue
        if "ПЗПІ" in run.text and not group_done:
            _set_run_text(run, meta.group)
            group_done = True
            continue
        if group_done and text.isdigit():
            _set_run_text(run, "")


def _fill_title_name_line(paragraph: Paragraph, meta: KkpMeta) -> None:
    letter_runs = [
        run
        for run in paragraph.runs
        if _is_underlined(run) and any(ch.isalpha() for ch in run.text)
    ]
    if not letter_runs:
        return
    _set_run_text(letter_runs[0], meta.student_title_name)
    for run in letter_runs[1:]:
        _set_run_text(run, "")


def _set_run_underline(run, enabled: bool) -> None:
    run.font.underline = WD_UNDERLINE.SINGLE if enabled else None


def _insert_run_after(anchor: Run, text: str, *, underlined: bool = False) -> Run:
    new_element = OxmlElement("w:r")
    anchor._element.addnext(new_element)
    new_run = Run(new_element, anchor._parent)
    _set_run_text(new_run, text)
    _set_run_underline(new_run, underlined)
    return new_run


def _set_cell_text(cell, text: str, *, alignment: WD_ALIGN_PARAGRAPH | None = None) -> None:
    paragraph = cell.paragraphs[0]
    if paragraph.runs:
        _set_run_text(paragraph.runs[0], text)
        for run in paragraph.runs[1:]:
            _set_run_text(run, "")
    else:
        _set_run_text(paragraph.add_run(text), text)
    if alignment is not None:
        paragraph.alignment = alignment


def _apply_report_table_borders(table: Table) -> None:
    tbl = table._tbl
    tbl_pr = tbl.find(qn("w:tblPr"))
    if tbl_pr is None:
        tbl_pr = OxmlElement("w:tblPr")
        tbl.insert(0, tbl_pr)

    for tag in ("w:tblW", "w:jc", "w:tblInd", "w:tblBorders"):
        existing = tbl_pr.find(qn(tag))
        if existing is not None:
            tbl_pr.remove(existing)

    tbl_w = OxmlElement("w:tblW")
    tbl_w.set(qn("w:w"), "9854")
    tbl_w.set(qn("w:type"), "dxa")
    tbl_pr.append(tbl_w)

    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center")
    tbl_pr.append(jc)

    tbl_ind = OxmlElement("w:tblInd")
    tbl_ind.set(qn("w:w"), "-199")
    tbl_ind.set(qn("w:type"), "dxa")
    tbl_pr.append(tbl_ind)

    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), "auto")
        borders.append(border)
    tbl_pr.append(borders)


def _add_blank_paragraph(doc: Document) -> None:
    doc.add_paragraph("")


def _set_paragraph_keep_next(paragraph: Paragraph, *, enabled: bool = True) -> None:
    p_pr = paragraph._element.get_or_add_pPr()
    keep_next = p_pr.find(qn("w:keepNext"))
    if enabled:
        if keep_next is None:
            p_pr.append(OxmlElement("w:keepNext"))
    elif keep_next is not None:
        p_pr.remove(keep_next)


def _set_paragraph_keep_lines(paragraph: Paragraph, *, enabled: bool = True) -> None:
    p_pr = paragraph._element.get_or_add_pPr()
    keep_lines = p_pr.find(qn("w:keepLines"))
    if enabled:
        if keep_lines is None:
            p_pr.append(OxmlElement("w:keepLines"))
    elif keep_lines is not None:
        p_pr.remove(keep_lines)


def _set_paragraph_widow_control(paragraph: Paragraph, *, enabled: bool = True) -> None:
    p_pr = paragraph._element.get_or_add_pPr()
    widow = p_pr.find(qn("w:widowControl"))
    if enabled:
        if widow is None:
            widow_el = OxmlElement("w:widowControl")
            widow_el.set(qn("w:val"), "0")
            p_pr.append(widow_el)
    elif widow is not None:
        p_pr.remove(widow)


def _add_table_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph(text)
    _apply_body_paragraph_format(paragraph)
    pf = paragraph.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(3)
    _set_paragraph_keep_next(paragraph)
    _set_paragraph_keep_lines(paragraph)
    _set_paragraph_widow_control(paragraph)
    for run in paragraph.runs:
        _ensure_font(run)


_ADVISOR_PARTS = (
    "доц. кафедри ПІ ",
    "Ольга ",
    "ВОРОЧЕК",
)
_ADVISOR_SUFFIX_SPACES = 0


def _fill_advisor_line(paragraph: Paragraph, meta: KkpMeta) -> None:
    underlined = [run for run in paragraph.runs if _is_underlined(run)]
    content_runs = underlined[1:] if len(underlined) > 1 else underlined
    for index, run in enumerate(content_runs):
        if index < len(_ADVISOR_PARTS):
            suffix = " " * _ADVISOR_SUFFIX_SPACES if index == len(_ADVISOR_PARTS) - 1 else ""
            _set_run_text(run, _ADVISOR_PARTS[index] + suffix)
        else:
            _set_run_text(run, "")


def _fill_calendar_advisor_line(paragraph: Paragraph, meta: KkpMeta) -> None:
    """Keep signature underline tab; split advisor across runs; extend line."""
    underlined = [run for run in paragraph.runs if _is_underlined(run)]
    if not underlined:
        return
    content_runs = [
        run
        for run in underlined[1:]
        if any(ch.isalpha() for ch in run.text) or run.text.strip() in {"", "\t"}
    ]
    letter_runs = [run for run in content_runs if any(ch.isalpha() for ch in run.text)]
    if not letter_runs:
        return
    for index, run in enumerate(letter_runs):
        if index < len(_ADVISOR_PARTS):
            _set_run_text(run, _ADVISOR_PARTS[index])
        else:
            _set_run_text(run, "")
    try:
        tail_start = underlined.index(letter_runs[-1]) + 1
    except ValueError:
        tail_start = len(underlined)
    for offset, run in enumerate(underlined[tail_start:]):
        _set_run_text(run, " " if offset == 0 else "")


def _letter_underlined_runs(paragraph: Paragraph) -> list:
    return [
        run
        for run in paragraph.runs
        if _is_underlined(run) and any(ch.isalpha() or ch.isdigit() for ch in run.text)
    ]


def _fill_underlined_letter_runs(paragraph: Paragraph, new_text: str) -> None:
    letter_runs = _letter_underlined_runs(paragraph)
    if not letter_runs:
        return
    first = letter_runs[0]
    leading = first.text[: len(first.text) - len(first.text.lstrip("\t"))]
    _set_run_text(first, f"{leading}{new_text}" if leading else new_text)
    for run in letter_runs[1:]:
        _set_run_text(run, "")


def _fill_dative_line(paragraph: Paragraph, meta: KkpMeta) -> None:
    _fill_underlined_letter_runs(paragraph, meta.student_dative)


def _fill_underlined_field(paragraph: Paragraph, text: str) -> None:
    """Replace underlined content and keep a trailing tab to extend the line to the margin."""
    underlined = [run for run in paragraph.runs if _is_underlined(run)]
    if not underlined:
        return
    _set_run_text(underlined[0], text)
    for run in underlined[1:-1]:
        _set_run_text(run, "")
    if len(underlined) > 1:
        _set_run_text(underlined[-1], "\t")
    else:
        _insert_run_after(underlined[0], "\t", underlined=True)


def _fill_underlined_tab_line(paragraph: Paragraph) -> None:
    """Restore a full-width underlined line using tab characters."""
    underlined = [run for run in paragraph.runs if _is_underlined(run)]
    for run in underlined:
        _set_run_text(run, "\t")


def _fill_topic_line(paragraph: Paragraph, meta: KkpMeta) -> None:
    underlined = [run for run in paragraph.runs if _is_underlined(run)]
    letter_runs = [run for run in underlined if any(ch.isalpha() for ch in run.text)]
    if not letter_runs:
        return
    _set_run_text(letter_runs[0], meta.topic)
    for run in letter_runs[1:]:
        _set_run_text(run, "")
    if underlined[-1].text != "\t":
        _set_run_text(underlined[-1], "\t")


def _ensure_title_tab_stops(paragraph: Paragraph) -> None:
    """Match title-page underline width: tabs from left margin to right margin."""
    p_pr = paragraph._element.get_or_add_pPr()
    tabs_el = p_pr.find(qn("w:tabs"))
    if tabs_el is None:
        tabs_el = OxmlElement("w:tabs")
        p_pr.append(tabs_el)
    for tab in tabs_el.findall(qn("w:tab")):
        tabs_el.remove(tab)
    for position in ("1701", "9354"):
        tab = OxmlElement("w:tab")
        tab.set(qn("w:val"), "left")
        tab.set(qn("w:pos"), position)
        tabs_el.append(tab)


def _fill_centered_underlined_title(paragraph: Paragraph, text: str) -> None:
    """Topic title on a full-width underlined line (template-style tabs)."""
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    _ensure_title_tab_stops(paragraph)
    element = paragraph._element
    for child in list(element):
        if child.tag != qn("w:pPr"):
            element.remove(child)
    for content in ("\t", text, "\t"):
        run = paragraph.add_run(content)
        _set_run_underline(run, True)
        _ensure_font(run)


def _fill_topic_title_block(doc: Document, meta: KkpMeta) -> None:
    """Topic on the underlined line directly above (тема)."""
    if len(doc.paragraphs) > 15:
        _fill_underlined_tab_line(doc.paragraphs[15])
    if len(doc.paragraphs) > 16:
        _fill_centered_underlined_title(doc.paragraphs[16], meta.topic)


def _fill_program_type_line(paragraph: Paragraph, meta: KkpMeta, *, indent: str = "\t") -> None:
    letter_runs = _letter_underlined_runs(paragraph)
    if letter_runs:
        _set_run_text(letter_runs[0], f"{indent}{meta.program_type}")
    underlined = [run for run in paragraph.runs if _is_underlined(run)]
    if underlined:
        _set_run_text(underlined[-1], "\t")


def _fill_item3_lines(doc: Document, meta: KkpMeta) -> None:
    if len(doc.paragraphs) > 58:
        _fill_underlined_field(doc.paragraphs[58], meta.initial_data)
    if len(doc.paragraphs) > 59:
        _fill_underlined_tab_line(doc.paragraphs[59])


def _fill_item4_lines(doc: Document, meta: KkpMeta) -> None:
    if len(doc.paragraphs) > 61:
        _fill_underlined_field(doc.paragraphs[61], meta.work_questions)


def _split_specialty_two_lines(specialty: str) -> tuple[str, str]:
    marker = " програмного "
    if marker in specialty:
        before, after = specialty.split(marker, 1)
        return f"{before} програмного", after
    words = specialty.split()
    if len(words) <= 1:
        return specialty, ""
    mid = (len(words) + 1) // 2
    return " ".join(words[:mid]), " ".join(words[mid:])


def _fill_underlined_value_line(
    paragraph: Paragraph,
    text: str,
    *,
    leading_tab: bool = False,
    trailing_spaces: str = "",
) -> None:
    underlined = [run for run in paragraph.runs if _is_underlined(run)]
    if not underlined:
        return
    content = f"\t{text}" if leading_tab else f"{text}{trailing_spaces}"
    _set_run_text(underlined[0], content)
    for run in underlined[1:-1]:
        _set_run_text(run, "")
    if len(underlined) > 1:
        _set_run_text(underlined[-1], "\t")
    else:
        _insert_run_after(underlined[0], "\t", underlined=True)


def _fill_specialty_lines(doc: Document, meta: KkpMeta) -> None:
    first_line, second_line = _split_specialty_two_lines(meta.specialty)
    if len(doc.paragraphs) > 26:
        _fill_underlined_value_line(doc.paragraphs[26], first_line, leading_tab=True)
    if len(doc.paragraphs) > 27:
        if second_line:
            _fill_underlined_value_line(
                doc.paragraphs[27],
                second_line,
                trailing_spaces="       ",
            )
        else:
            _fill_underlined_tab_line(doc.paragraphs[27])


def _fill_assignment_header_block(doc: Document, meta: KkpMeta) -> None:
    """Page-2 header table rows — only replace underlined values, keep hints."""
    mapping = {
        44: meta.specialty,
        46: meta.educational_program,
    }
    for index, value in mapping.items():
        if index < len(doc.paragraphs):
            _fill_underlined_letter_runs(doc.paragraphs[index], value)
    if len(doc.paragraphs) > 45:
        _fill_program_type_line(doc.paragraphs[45], meta, indent="\t  ")


def _disable_run_caps(run) -> None:
    rpr = run._element.get_or_add_rPr()
    caps = rpr.find(qn("w:caps"))
    if caps is not None:
        rpr.remove(caps)


def _fill_title_year_line(paragraph: Paragraph, year: str) -> None:
    element = paragraph._element
    for child in list(element):
        if child.tag != qn("w:pPr"):
            element.remove(child)
    year_run = paragraph.add_run(year)
    _ensure_font(year_run)
    space_run = paragraph.add_run(" ")
    _ensure_font(space_run)
    suffix_run = paragraph.add_run("р.")
    _ensure_font(suffix_run)
    _disable_run_caps(suffix_run)


def _fill_assignment_date_line(paragraph: Paragraph, meta: KkpMeta) -> None:
    day, month, _year = meta.assignment_date.split(".")
    month_name = _UA_MONTHS.get(month, month)
    for run in paragraph.runs:
        if run.text.strip() == "___":
            _set_run_text(run, day)
            _set_run_underline(run, True)
        elif run.text.strip() == "_____":
            _set_run_text(run, month_name)
            _set_run_underline(run, True)
        elif "р." in run.text:
            _set_run_text(run, f"{meta.year} р.")
            _disable_run_caps(run)
            _ensure_font(run)


def _fill_submission_line(paragraph: Paragraph, meta: KkpMeta) -> None:
    day, month, year = meta.submission_date.split(".")
    month_name = _UA_MONTHS.get(month, month)
    date_run = None
    for run in paragraph.runs:
        if "____" in run.text or meta.year in run.text:
            date_run = run
            break
    if date_run is None:
        return
    _set_run_text(date_run, " „")
    _set_run_underline(date_run, False)
    anchor = _insert_run_after(date_run, day, underlined=True)
    anchor = _insert_run_after(anchor, "” ", underlined=False)
    anchor = _insert_run_after(anchor, month_name, underlined=True)
    year_run = _insert_run_after(anchor, f" {year} р.", underlined=False)
    _disable_run_caps(year_run)


def _fill_plain_fields(doc: Document, meta: KkpMeta) -> None:
    """Replace only fields without underline/hint structure."""
    if len(doc.paragraphs) > 38:
        _fill_title_year_line(doc.paragraphs[38], meta.year)

    if len(doc.paragraphs) > 17:
        _fill_topic_title_block(doc, meta)
    _fill_specialty_lines(doc, meta)
    if len(doc.paragraphs) > 29:
        _fill_program_type_line(doc.paragraphs[29], meta)
    if len(doc.paragraphs) > 30:
        _fill_underlined_letter_runs(doc.paragraphs[30], meta.educational_program)
    _fill_assignment_header_block(doc, meta)


def _fill_commission_table(doc: Document, meta: KkpMeta) -> None:
    for table in doc.tables:
        if not table.rows:
            continue
        header = _normalize(table.rows[0].cells[0].text)
        if not header.startswith("Члени комісії"):
            continue
        for row_index, member in enumerate(meta.commission_members, start=1):
            if row_index >= len(table.rows):
                break
            _set_cell_text(table.rows[row_index].cells[0], member)
        return


def _ensure_assignment_sheet_starts_on_new_page(doc: Document, meta: KkpMeta) -> None:
    """Keep «2026 р.» at the bottom of the title page; assignment starts on the next page."""
    year_idx = _find_paragraph_index(doc, f"{meta.year} р.")
    if year_idx is None:
        year_idx = _find_paragraph_index(doc, "2026 р.")
    if year_idx is None:
        return

    year_paragraph = doc.paragraphs[year_idx]
    year_ppr = year_paragraph._element.pPr
    if year_ppr is not None:
        page_break = year_ppr.find(qn("w:pageBreakBefore"))
        if page_break is not None:
            year_ppr.remove(page_break)

    for index in range(year_idx + 1, len(doc.paragraphs)):
        paragraph = doc.paragraphs[index]
        if _normalize(paragraph.text).startswith("Харківський"):
            _set_page_break_before(paragraph)
            return


def _fill_course_group_table(doc: Document, meta: KkpMeta) -> None:
    if len(doc.tables) < 2:
        return
    table = doc.tables[1]
    if len(table.rows) < 1 or len(table.rows[0].cells) < 6:
        return
    row = table.rows[0].cells
    _set_cell_text(row[1], meta.course)
    _set_cell_text(row[3], meta.group)
    _set_cell_text(row[5], meta.semester)


def _fill_calendar_table(doc: Document, meta: KkpMeta) -> None:
    if len(doc.tables) < 3:
        return
    table = doc.tables[2]
    schedule = [
        "03.04.2026 – 17.04.2026",
        "18.04.2026 – 01.05.2026",
        "02.05.2026 – 15.05.2026",
        "16.05.2026 – 25.05.2026",
        "26.05.2026 – 31.05.2026",
        "01.06.2026 – 05.06.2026",
        "06.06.2026 – 10.06.2026",
    ]
    for row_index, term in enumerate(schedule, start=1):
        if row_index >= len(table.rows):
            break
        cells = table.rows[row_index].cells
        if len(cells) >= 4:
            _set_cell_text(cells[2], term)
            _set_cell_text(cells[3], "виконано")


def _remove_example_tables(doc: Document) -> None:
    while len(doc.tables) > 3:
        _delete_table(doc.tables[3])


def _trim_executor_label_indent(doc: Document) -> None:
    """Remove leading tab/space before «Виконав:» so the title page footer fits."""
    if len(doc.paragraphs) <= 21:
        return
    paragraph = doc.paragraphs[21]
    if not paragraph.runs:
        return
    first = paragraph.runs[0]
    if first.text.startswith(("\t", " ")):
        first.text = first.text[1:]


def fill_front_matter(doc: Document, meta: KkpMeta) -> None:
    _trim_executor_label_indent(doc)
    _fill_plain_fields(doc, meta)

    if len(doc.paragraphs) > 22:
        _fill_course_group_line(doc.paragraphs[22], meta)
    if len(doc.paragraphs) > 23:
        _fill_title_name_line(doc.paragraphs[23], meta)
    if len(doc.paragraphs) > 33:
        _fill_advisor_line(doc.paragraphs[33], meta)
    if len(doc.paragraphs) > 54:
        _fill_dative_line(doc.paragraphs[54], meta)
    if len(doc.paragraphs) > 56:
        _fill_topic_line(doc.paragraphs[56], meta)
    if len(doc.paragraphs) > 57:
        _fill_submission_line(doc.paragraphs[57], meta)
    if len(doc.paragraphs) > 68:
        _fill_assignment_date_line(doc.paragraphs[68], meta)
    if len(doc.paragraphs) > 73:
        _fill_calendar_advisor_line(doc.paragraphs[73], meta)
    _fill_item3_lines(doc, meta)
    _fill_item4_lines(doc, meta)
    # paras 4,7,24,28,31,34,47,55,70,71,74 — hints and signature lines left intact

    _fill_commission_table(doc, meta)
    _ensure_assignment_sheet_starts_on_new_page(doc, meta)
    _fill_course_group_table(doc, meta)
    _fill_calendar_table(doc, meta)


def _set_paragraph_content(paragraph: Paragraph, text: str) -> None:
    _set_paragraph_text(paragraph, text)
    if paragraph.runs:
        _ensure_font(paragraph.runs[0])
    elif text:
        _ensure_font(paragraph.add_run(text))


def _ensure_toc_tab_stops(paragraph: Paragraph) -> None:
    p_pr = paragraph._element.get_or_add_pPr()
    tabs_el = p_pr.find(qn("w:tabs"))
    if tabs_el is None:
        tabs_el = OxmlElement("w:tabs")
        p_pr.append(tabs_el)
    for tab in tabs_el.findall(qn("w:tab")):
        tabs_el.remove(tab)
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "right")
    tab.set(qn("w:leader"), "dot")
    tab.set(qn("w:pos"), "9344")
    tabs_el.append(tab)


def _toc_entry_level(title: str) -> int:
    token = title.split()[0] if title else ""
    if not token or not token[0].isdigit():
        return 0
    return token.count(".")


def _apply_toc_paragraph_format(paragraph: Paragraph, level: int) -> None:
    pf = paragraph.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.left_indent = Cm(0)
    pf.first_line_indent = Cm(1.25 * level) if level else Cm(0)
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.5
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE


def _set_toc_entry(paragraph: Paragraph, title: str, page: str = "") -> None:
    element = paragraph._element
    for child in list(element):
        if child.tag != qn("w:pPr"):
            element.remove(child)
    if not title:
        return
    _apply_toc_paragraph_format(paragraph, _toc_entry_level(title))
    _ensure_toc_tab_stops(paragraph)
    run = paragraph.add_run(f"{title}\t{page}")
    _ensure_font(run)


def _paragraph_has_page_break(paragraph: Paragraph) -> bool:
    for run in paragraph.runs:
        for br in run._element.findall(qn("w:br")):
            if br.get(qn("w:type")) == "page":
                return True
    p_pr = paragraph._element.find(qn("w:pPr"))
    return p_pr is not None and p_pr.find(qn("w:pageBreakBefore")) is not None


def _paragraph_is_truly_empty(paragraph: Paragraph) -> bool:
    return not paragraph.text.strip()


def _paragraph_is_empty_or_page_break_only(paragraph: Paragraph) -> bool:
    if paragraph.text.strip():
        return False
    return (
        not paragraph.runs
        or all(not run.text.strip() for run in paragraph.runs)
        or _paragraph_has_page_break(paragraph)
    )


def _add_page_break_after(paragraph: Paragraph) -> None:
    if paragraph.runs:
        paragraph.runs[-1].add_break(WD_BREAK.PAGE)
    else:
        paragraph.add_run().add_break(WD_BREAK.PAGE)


def _strip_leading_page_break(paragraph: Paragraph) -> None:
    p_pr = paragraph._element.find(qn("w:pPr"))
    if p_pr is not None:
        page_break_before = p_pr.find(qn("w:pageBreakBefore"))
        if page_break_before is not None:
            p_pr.remove(page_break_before)
    for run in paragraph.runs:
        for br in list(run._element.findall(qn("w:br"))):
            if br.get(qn("w:type")) == "page" and not run.text:
                run._element.remove(br)


def _strip_trailing_page_break(paragraph: Paragraph) -> None:
    for run in reversed(paragraph.runs):
        for br in list(run._element.findall(qn("w:br"))):
            if br.get(qn("w:type")) == "page":
                run._element.remove(br)


def _set_page_break_before(paragraph: Paragraph) -> None:
    p_pr = paragraph._element.get_or_add_pPr()
    if p_pr.find(qn("w:pageBreakBefore")) is None:
        page_break = OxmlElement("w:pageBreakBefore")
        p_pr.insert(0, page_break)


def _ensure_page_break_before_heading(
    doc: Document, heading_prefix: str, *, start: int = 0
) -> None:
    """Start major section on a new page (works even when a table precedes the heading)."""
    heading_index = _find_paragraph_index(doc, heading_prefix, start=start)
    if heading_index is None or heading_index == 0:
        return
    for index in range(heading_index - 1, 0, -1):
        paragraph = doc.paragraphs[index]
        if _paragraph_is_truly_empty(paragraph) or (
            not paragraph.text.strip() and _paragraph_has_page_break(paragraph)
        ):
            _delete_paragraph(paragraph)
            heading_index = _find_paragraph_index(doc, heading_prefix, start=start)
            if heading_index is None or heading_index == 0:
                return
        else:
            break
    heading_index = _find_paragraph_index(doc, heading_prefix, start=start)
    if heading_index is None or heading_index == 0:
        return
    if heading_index > 0:
        _strip_trailing_page_break(doc.paragraphs[heading_index - 1])
    _strip_leading_page_break(doc.paragraphs[heading_index])
    _set_page_break_before(doc.paragraphs[heading_index])


def _paragraph_page_break_before(paragraph: Paragraph) -> bool:
    p_pr = paragraph._element.find(qn("w:pPr"))
    return p_pr is not None and p_pr.find(qn("w:pageBreakBefore")) is not None


def _estimate_paragraph_lines(paragraph: Paragraph) -> float:
    text = paragraph.text.strip()
    if not text:
        return 0.4
    if text.startswith("Рис."):
        return 1.0
    if text.startswith("Таблиця"):
        return 1.5
    if text == text.upper() and len(text) < 70:
        return 2.0
    if text[:3].replace(".", "").isdigit():
        return 2.0
    run_font = paragraph.runs[0].font.name if paragraph.runs else None
    if run_font and "Courier" in str(run_font):
        return max(1.0, len(text) / 80.0)
    if paragraph._element.findall(f".//{qn('w:drawing')}"):
        return 14.0
    return max(1.0, len(text) / 68.0) * 1.45


def _estimate_table_lines(table: Table) -> float:
    return max(3.0, len(table.rows) * 1.3)


def _build_paragraph_page_map(
    doc: Document, *, anchor_index: int, anchor_page: int
) -> dict[int, int]:
    lines_per_page = 27.0
    page_map: dict[int, int] = {}
    current_page = 1
    lines_used = 0.0
    para_index = 0
    table_index = 0

    for child in doc.element.body:
        tag = child.tag.split("}")[-1]
        if tag == "p":
            paragraph = doc.paragraphs[para_index]
            needs_break = _paragraph_page_break_before(paragraph)
            if para_index > 0 and _paragraph_has_page_break(doc.paragraphs[para_index - 1]):
                needs_break = True
            if needs_break:
                current_page += 1
                lines_used = 0.0

            if para_index == anchor_index:
                current_page = anchor_page
                lines_used = 0.0
            if para_index >= anchor_index:
                page_map[para_index] = current_page

            lines_used += _estimate_paragraph_lines(paragraph)
            while lines_used >= lines_per_page:
                current_page += 1
                lines_used -= lines_per_page

            para_index += 1
        elif tag == "tbl":
            if table_index < len(doc.tables):
                lines_used += _estimate_table_lines(doc.tables[table_index])
                while lines_used >= lines_per_page:
                    current_page += 1
                    lines_used -= lines_per_page
            table_index += 1

    return page_map


def _page_number_for_paragraph(
    doc: Document, para_index: int, *, perelik_index: int, perelik_page: int = 8
) -> str:
    page_map = _build_paragraph_page_map(doc, anchor_index=perelik_index, anchor_page=perelik_page)
    return str(page_map.get(para_index, perelik_page))


def _strip_document_highlights(doc: Document) -> None:
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            _clear_run_highlight(run)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        _clear_run_highlight(run)


def _trim_abstract_before_toc(doc: Document) -> None:
    toc = _find_paragraph_index(doc, "ЗМІСТ")
    if toc is None:
        return
    for index in range(toc - 1, 0, -1):
        paragraph = doc.paragraphs[index]
        if _paragraph_is_truly_empty(paragraph) or (
            not paragraph.text.strip() and _paragraph_has_page_break(paragraph)
        ):
            _delete_paragraph(paragraph)
        else:
            break


def _cleanup_before_abbreviations(doc: Document) -> None:
    note = _find_paragraph_index(doc, "Примітка*")
    perelik = _find_paragraph_index(doc, "ПЕРЕЛІК СКОРОЧЕНЬ")
    if note is not None and perelik is not None and note < perelik:
        _delete_paragraphs(doc, note, perelik)
        perelik = _find_paragraph_index(doc, "ПЕРЕЛІК СКОРОЧЕНЬ")
    toc = _find_paragraph_index(doc, "ЗМІСТ")
    if toc is None or perelik is None:
        return
    for index in range(perelik - 1, toc, -1):
        paragraph = doc.paragraphs[index]
        if not paragraph.text.strip():
            _delete_paragraph(paragraph)


def _looks_like_toc_entry(text: str) -> bool:
    parts = text.rsplit("\t", 1)
    return len(parts) == 2 and parts[1].strip().isdigit()


def _find_toc_target_paragraph(doc: Document, entry: str, *, start: int) -> int | None:
    parts = entry.split(maxsplit=1)
    prefix = parts[0]
    title_hint = parts[1].lower() if len(parts) > 1 else entry.lower()
    for index in range(start, len(doc.paragraphs)):
        text = doc.paragraphs[index].text.strip()
        if not text or _looks_like_toc_entry(text):
            continue
        normalized = _normalize(text)
        if normalized.lower() == entry.lower() or normalized.upper() == entry.upper():
            return index
        if prefix.replace(".", "").isdigit():
            section_prefix = prefix.split(".")[0]
            if not text.startswith(section_prefix):
                continue
            if "." in prefix:
                if not text.startswith(prefix):
                    continue
            elif text[len(section_prefix) : len(section_prefix) + 1] == ".":
                continue
            if title_hint[:12] in text.lower():
                return index
        elif text.upper().startswith(entry.upper()):
            return index
    return None


def _content_report_stats(meta: KkpMeta, *, include_figures: bool = True) -> tuple[int, int, int]:
    """Figures/tables/sources from report content (not caption paragraphs)."""
    figures = 0
    tables = 0
    for _section_title, blocks in report_sections(meta):
        for _subtitle, items in blocks:
            for item in items:
                if isinstance(item, FigureBlock):
                    if include_figures:
                        figures += 1
                elif isinstance(item, TableBlock):
                    tables += 1
    return figures, tables, len(references(meta))


def _compute_report_stats(
    doc: Document,
    meta: KkpMeta,
    *,
    pages: int | None = None,
    include_figures: bool = True,
) -> tuple[int, int, int, int]:
    figures, tables, sources = _content_report_stats(meta, include_figures=include_figures)
    if pages is not None:
        return pages, figures, tables, sources
    perelik = _find_paragraph_index(doc, "ПЕРЕЛІК СКОРОЧЕНЬ") or 0
    last_index = len(doc.paragraphs) - 1
    pages = int(_page_number_for_paragraph(doc, last_index, perelik_index=perelik))
    return pages, figures, tables, sources


def update_abstract_stats(
    doc: Document, meta: KkpMeta, *, pages: int, include_figures: bool = True
) -> None:
    """Refresh РЕФЕРАТ stats line after layout (real page count from PDF)."""
    start = _find_paragraph_index(doc, "РЕФЕРАТ")
    if start is None:
        return
    figures, tables, sources = _content_report_stats(meta, include_figures=include_figures)
    line = (
        f"Пояснювальна записка містить: {pages} с., {figures} рис., "
        f"{tables} табл., {sources} джерел."
    )
    for index in range(start + 1, min(start + 6, len(doc.paragraphs))):
        if doc.paragraphs[index].text.strip().startswith("Пояснювальна записка містить:"):
            _set_paragraph_content(doc.paragraphs[index], line)
            return


def fill_abstract(
    doc: Document, meta: KkpMeta, *, stats: tuple[int, int, int, int] | None = None
) -> None:
    start = _find_paragraph_index(doc, "РЕФЕРАТ")
    end = _find_paragraph_index(doc, "ЗМІСТ", start=(start or 0) + 1)
    if start is None or end is None:
        return

    for index in range(end - 1, start, -1):
        text = doc.paragraphs[index].text.strip()
        if text.startswith("Кількість сторінок") or text.startswith("В англомовній"):
            _delete_paragraph(doc.paragraphs[index])

    start = _find_paragraph_index(doc, "РЕФЕРАТ")
    end = _find_paragraph_index(doc, "ЗМІСТ", start=(start or 0) + 1)
    if start is None or end is None:
        return

    keywords_ua = (
        "AI-ГЕНЕРАЦІЯ, БІНАРНА КЛАСИФІКАЦІЯ, COMPUTER VISION, DEEP LEARNING, "
        "EFFICIENTNET, GRADIO, PYTORCH, RESNET, TRANSFER LEARNING."
    )
    keywords_en = (
        "AI GENERATION, BINARY CLASSIFICATION, COMPUTER VISION, DEEP LEARNING, "
        "EFFICIENTNET, GRADIO, PYTORCH, RESNET, TRANSFER LEARNING."
    )

    if stats is None:
        stats = _compute_report_stats(doc, meta)
    pages, figures, tables, sources = stats

    replacements: dict[int, str] = {
        start + 1: (
            f"Пояснювальна записка містить: {pages} с., {figures} рис., {tables} табл., {sources} джерел."
        ),
        start + 3: keywords_ua,
        start + 5: (
            "Об'єкт розробки — це програмна система, яка автоматично визначає, "
            "чи було зображення згенеровано штучним інтелектом."
        ),
        start + 6: (
            "Мета розробки — спроєктувати і реалізувати повний цикл машинного "
            "навчання: від завантаження hi-res датасету та тренування двох "
            "згорткових нейронних мереж до порівняння метрик і створення "
            "інтерактивного демо."
        ),
        start + 7: (
            "Метод рішення — в якості основної мови програмування обрано Python "
            "з використанням фреймворку PyTorch та бібліотеки scikit-learn для "
            "машинного навчання. В основі рішення лежить донавчання вже готових "
            "архітектур (ResNet18 та EfficientNet-B0). Для реалізації фронтенду "
            "було використано Gradio, для тестування — pytest, а для деплою та "
            "автоматизації процесів застосовано Docker і GitHub Actions."
        ),
        start + 8: (
            "У результаті EfficientNet-B0 досягла accuracy 98,5 % на контрольній "
            "вибірці проти 93,6 % у ResNet18 на датасеті Parveshiiii/AI-vs-Real."
        ),
        start + 10: keywords_en,
        start + 12: (
            "The object of development is a software system that automatically "
            "determines whether an image was generated by artificial intelligence."
        ),
        start + 13: (
            "The purpose of the work is to design and implement a full machine "
            "learning cycle: from loading a hi-res dataset and training two "
            "convolutional neural networks to comparing metrics and creating an "
            "interactive demo."
        ),
        start + 14: (
            "The solution method — Python was chosen as the main programming "
            "language, using the PyTorch framework and the scikit-learn library "
            "for machine learning. The solution is based on fine-tuning pre-trained "
            "architectures (ResNet18 and EfficientNet-B0). Gradio was used for "
            "the frontend, pytest for testing, and Docker and GitHub Actions for "
            "deployment and process automation."
        ),
        start + 15: (
            "As a result, EfficientNet-B0 achieved 98.5% accuracy on the test set "
            "versus 93.6% for ResNet18 on the Parveshiiii/AI-vs-Real dataset."
        ),
    }

    for index, text in replacements.items():
        if index < len(doc.paragraphs):
            _set_paragraph_content(doc.paragraphs[index], text)

    for index in (start + 2, start + 4, start + 9, start + 11):
        if index < len(doc.paragraphs):
            _set_paragraph_content(doc.paragraphs[index], "")


def _replace_abbreviations(doc: Document) -> None:
    abbreviations = [
        "AI — Artificial Intelligence (штучний інтелект)",
        "AUC — Area Under Curve (площа під кривою)",
        "CI — Continuous Integration (безперервна інтеграція)",
        "CNN — Convolutional Neural Network (згорткова нейронна мережа)",
        "F1 — F1-score (гармонійне середнє precision і recall)",
        "Gradio — Python framework для ML demo",
        "PyTorch — бібліотека deep learning",
        "ROC — Receiver Operating Characteristic",
        "YAML — YAML Ain't Markup Language",
    ]
    start = _find_paragraph_index(doc, "ПЕРЕЛІК СКОРОЧЕНЬ")
    if start is None:
        return
    end = _find_paragraph_index(doc, "ВСТУП", start=start + 1)
    if end is None:
        return
    for offset, item in enumerate(abbreviations):
        index = start + 1 + offset
        if index < end:
            _set_paragraph_content(doc.paragraphs[index], item)
    abbrev_end = start + len(abbreviations)
    for index in range(end - 1, abbrev_end, -1):
        if index < len(doc.paragraphs):
            _delete_paragraph(doc.paragraphs[index])


TOC_ENTRIES = [
    "Перелік скорочень",
    "Вступ",
    "1 Аналіз предметної галузі",
    "1.1 Аналіз предметної галузі",
    "1.2 Виявлення та вирішення проблем",
    "1.2.1 Цільова аудиторія",
    "1.2.2 Обмеження та припущення",
    "1.3 Аналіз аналогів програмного забезпечення",
    "2 Постановка задачі",
    "3 Архітектура та проєктування програмного забезпечення",
    "3.1 Сценарії використання системи",
    "3.2 Проєктування архітектури ПЗ",
    "3.3 Проєктування структури зберігання даних",
    "3.4 Приклади використаних алгоритмів та методів",
    "3.5 Проєктування UI/UX",
    "3.6 Опис підготовки даних",
    "4 Опис прийнятих програмних рішень",
    "4.1 Пайплайн навчання та оцінки",
    "4.2 Реалізація моделей та factory",
    "4.3 Препроцесинг та аугментації",
    "4.4 Веб-інтерфейс системи та inference",
    "4.5 DevOps та якість коду",
    "5 Аналіз отриманих результатів",
    "5.1 Теоретичний аналіз моделей",
    "5.2 Методика проведення експериментів",
    "5.3 Практичні результати та їх аналіз",
    "6 Тестування програмного забезпечення",
    "6.1 Модульне тестування",
    "6.2 CI та статичний аналіз",
    "6.3 Приймальні тест-кейси",
    "7 Впровадження програмного забезпечення",
    "7.1 Визначення плану впровадження",
    "7.2 Розгортання та використання системи",
    "7.3 Тест-кейси впровадження",
    "Висновки",
    "Перелік джерел посилання",
]


_MAJOR_SECTION_BREAKS = [
    "ВСТУП",
    "1 АНАЛІЗ",
    "2 ПОСТАНОВКА",
    "3 АРХІТЕКТУРА",
    "4 ОПИС",
    "5 АНАЛІЗ",
    "6 ТЕСТУВАННЯ",
    "7 ВПРОВАДЖЕННЯ",
    "ВИСНОВКИ",
    "ПЕРЕЛІК ДЖЕРЕЛ",
]


def _ensure_toc_capacity(doc: Document, needed: int) -> None:
    toc_start = _find_paragraph_index(doc, "ЗМІСТ")
    perelik = _find_paragraph_index(doc, "ПЕРЕЛІК СКОРОЧЕНЬ")
    if toc_start is None or perelik is None:
        return
    while perelik - toc_start - 1 < needed:
        perelik_para = doc.paragraphs[perelik]
        _insert_paragraph_before(perelik_para)
        perelik = _find_paragraph_index(doc, "ПЕРЕЛІК СКОРОЧЕНЬ")
        if perelik is None:
            return


def _apply_major_section_page_breaks(doc: Document) -> None:
    for prefix in _MAJOR_SECTION_BREAKS:
        _ensure_page_break_before_heading(doc, prefix)


def fill_table_of_contents(doc: Document, pages: list[str] | None = None) -> None:
    _ensure_toc_capacity(doc, len(TOC_ENTRIES))
    toc_start = _find_paragraph_index(doc, "ЗМІСТ")
    perelik = _find_paragraph_index(doc, "ПЕРЕЛІК СКОРОЧЕНЬ")
    if toc_start is None or perelik is None:
        return
    for index in range(toc_start + 1, perelik):
        offset = index - toc_start - 1
        if offset < len(TOC_ENTRIES):
            page = pages[offset] if pages and offset < len(pages) else ""
            _set_toc_entry(doc.paragraphs[index], TOC_ENTRIES[offset], page)
        else:
            _set_toc_entry(doc.paragraphs[index], "")

    _ensure_page_break_before_heading(doc, "ПЕРЕЛІК СКОРОЧЕНЬ")


def _remove_example_body(doc: Document) -> None:
    start = _find_paragraph_index(doc, "ВСТУП")
    if start is None:
        return
    _delete_paragraphs(doc, start, None)
    _remove_example_tables(doc)


def _apply_heading_paragraph_format(paragraph: Paragraph, *, gap_before_body: bool = False) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = paragraph.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(12) if gap_before_body else Pt(0)
    pf.line_spacing = 1.5
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE


def _apply_subheading_paragraph_format(paragraph: Paragraph) -> None:
    pf = paragraph.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.first_line_indent = Cm(1.25)
    pf.space_before = Pt(12)
    pf.space_after = Pt(12)
    pf.line_spacing = 1.5
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE


def _apply_body_paragraph_format(paragraph: Paragraph) -> None:
    pf = paragraph.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.first_line_indent = Cm(1.25)
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.5
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE


def _add_heading(
    doc: Document,
    text: str,
    *,
    page_break: bool = False,
    gap_before_body: bool = False,
) -> Paragraph:
    paragraph = doc.add_paragraph()
    if page_break:
        run = paragraph.add_run()
        run.add_break(WD_BREAK.PAGE)
    run = paragraph.add_run(text)
    run.bold = True
    _ensure_font(run)
    _apply_heading_paragraph_format(paragraph, gap_before_body=gap_before_body)
    return paragraph


def _add_subheading(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    run = paragraph.add_run(text)
    _ensure_font(run)
    _apply_subheading_paragraph_format(paragraph)


def _add_body(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    _apply_body_paragraph_format(paragraph)
    append_mixed_text(paragraph, text, font_callback=_ensure_font)


def _add_code_block(doc: Document, code: str) -> None:
    _add_blank_paragraph(doc)
    for line in code.strip("\n").splitlines():
        paragraph = doc.add_paragraph()
        pf = paragraph.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.left_indent = Cm(1.25)
        pf.first_line_indent = Cm(0)
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
        run = paragraph.add_run(line if line else " ")
        run.font.name = "Courier New"
        run.font.size = Pt(10)
        rpr = run._element.get_or_add_rPr()
        rpr.rFonts.set(qn("w:ascii"), "Courier New")
        rpr.rFonts.set(qn("w:hAnsi"), "Courier New")
        rpr.rFonts.set(qn("w:cs"), "Courier New")
    _add_blank_paragraph(doc)


def _add_math(doc: Document, latex: str) -> None:
    add_math_paragraph(doc, latex)


def _add_figure_caption(doc: Document, caption: str, *, tight_after: bool = False) -> Paragraph:
    paragraph = doc.add_paragraph(caption)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = paragraph.paragraph_format
    pf.space_before = Pt(2)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    _set_paragraph_keep_lines(paragraph)
    _set_paragraph_widow_control(paragraph)
    if not tight_after:
        pf.space_after = Pt(6)
    for run in paragraph.runs:
        _ensure_font(run)
    return paragraph


def _add_figure_caption_only(
    doc: Document,
    caption: str,
    *,
    leading_blank: bool = True,
    tight_after: bool = False,
    blank_after_caption: bool = False,
) -> None:
    if leading_blank:
        _add_blank_paragraph(doc)
    _add_figure_caption(doc, caption, tight_after=tight_after and not blank_after_caption)
    if blank_after_caption:
        _add_blank_paragraph(doc)


def _add_figure(
    doc: Document,
    path,
    caption: str,
    *,
    leading_blank: bool = True,
    tight_after: bool = False,
    blank_after_caption: bool = False,
    width_cm: float = 14,
) -> None:
    if leading_blank:
        _add_blank_paragraph(doc)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = paragraph.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    run = paragraph.add_run()
    run.add_picture(str(path), width=Cm(width_cm))
    _set_paragraph_keep_next(paragraph)
    _set_paragraph_keep_lines(paragraph)
    _add_figure_caption(doc, caption, tight_after=tight_after and not blank_after_caption)
    if blank_after_caption:
        _add_blank_paragraph(doc)


def _populate_table_row(
    table: Table,
    row_index: int,
    values: list[str],
    *,
    center_from: int = 1,
) -> None:
    for col, value in enumerate(values):
        alignment = WD_ALIGN_PARAGRAPH.CENTER if col >= center_from else None
        _set_cell_text(table.rows[row_index].cells[col], value, alignment=alignment)


def _add_report_table(
    doc: Document,
    caption: str,
    headers: list[str],
    rows: list[list[str]],
    *,
    center_from: int = 1,
    leading_blank: bool = True,
) -> None:
    if leading_blank:
        _add_blank_paragraph(doc)
    _add_table_caption(doc, caption)
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    _apply_report_table_borders(table)
    _populate_table_row(table, 0, headers, center_from=0)
    for row_index, row_data in enumerate(rows, start=1):
        _populate_table_row(table, row_index, row_data, center_from=center_from)
    _add_blank_paragraph(doc)


def _add_test_case_table(
    doc: Document,
    table_id: str,
    title: str,
    fields: list[tuple[str, str]],
    *,
    leading_blank: bool = True,
    trailing_blank: bool = True,
) -> None:
    if leading_blank:
        _add_blank_paragraph(doc)
    _add_table_caption(doc, f"Таблиця {table_id} – {title}")
    table = doc.add_table(rows=len(fields), cols=2)
    _apply_report_table_borders(table)
    for row_index, (field, value) in enumerate(fields):
        _set_cell_text(table.rows[row_index].cells[0], field)
        _set_cell_text(table.rows[row_index].cells[1], value)
    if trailing_blank:
        _add_blank_paragraph(doc)


def _add_dataset_table(doc: Document, *, leading_blank: bool = True) -> None:
    _add_report_table(
        doc,
        "Таблиця 3.1 – Структура датасету Parveshiiii/AI-vs-Real",
        ["Split", "REAL", "FAKE", "Разом"],
        [
            ["train", "800", "800", "1600"],
            ["val", "100", "100", "200"],
            ["test", "102", "102", "204"],
        ],
        leading_blank=leading_blank,
    )


def _add_hyperparams_table(doc: Document, *, leading_blank: bool = True) -> None:
    _add_report_table(
        doc,
        "Таблиця 4.1 – Гіперпараметри навчання обох моделей",
        ["Параметр", "ResNet18", "EfficientNet-B0"],
        [
            ["batch_size", "32", "32"],
            ["epochs", "10", "10"],
            ["learning_rate", "0,001", "0,001"],
            ["image_size", "224×224", "224×224"],
            ["pretrained", "ImageNet", "ImageNet"],
            ["augmentations", "strong", "strong"],
            ["optimizer", "Adam", "Adam"],
            ["loss", "CrossEntropy", "CrossEntropy"],
        ],
        leading_blank=leading_blank,
    )


def _add_models_theory_table(doc: Document, *, leading_blank: bool = True) -> None:
    _add_report_table(
        doc,
        "Таблиця 5.1 – Теоретичне порівняння архітектур CNN",
        ["Характеристика", "ResNet18", "EfficientNet-B0"],
        [
            ["Тип архітектури", "Residual CNN (18 шарів)", "MBConv + compound scaling"],
            ["Pretrained ваги", "ImageNet", "ImageNet"],
            ["Параметри (орієнтовно)", "≈ 11 млн", "≈ 5 млн"],
            ["Глибина / width", "Фіксована", "Масштабована (depth, width, resolution)"],
            [
                "Очікувана поведінка",
                "Швидкий baseline, стабільна збіжність",
                "Краща точність при менших params",
            ],
            ["Потенційні слабкості", "Менша ємність для fine-grained ознак", "Більше часу на CPU"],
        ],
        leading_blank=leading_blank,
    )


def _add_metrics_table(doc: Document, *, leading_blank: bool = True) -> None:
    _add_report_table(
        doc,
        "Таблиця 5.2 – Порівняння метрик на test set",
        ["Модель", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        [
            ["ResNet18", "93,6 %", "0,908", "0,971", "0,938", "0,992"],
            ["EfficientNet-B0", "98,5 %", "0,971", "1,000", "0,986", "1,000"],
        ],
        leading_blank=leading_blank,
    )


def _add_error_analysis_table(doc: Document, *, leading_blank: bool = True) -> None:
    _add_report_table(
        doc,
        "Таблиця 5.3 – Приклади помилкових класифікацій на test set",
        ["Модель", "Файл", "Справжній клас", "Передбачення", "Confidence", "Тип помилки"],
        [
            [
                "ResNet18",
                "test_real_0049.jpg",
                "real",
                "ai_generated",
                "98,4 %",
                "FP (хибна тривога)",
            ],
            [
                "ResNet18",
                "test_fake_0043.jpg",
                "ai_generated",
                "real",
                "58,5 %",
                "FN (пропуск AI)",
            ],
            [
                "ResNet18",
                "test_real_0001.jpg",
                "real",
                "ai_generated",
                "89,8 %",
                "FP; стійка для обох моделей",
            ],
            [
                "EfficientNet-B0",
                "test_real_0004.jpg",
                "real",
                "ai_generated",
                "99,3 %",
                "FP (висока впевненість)",
            ],
            [
                "EfficientNet-B0",
                "test_real_0043.jpg",
                "real",
                "ai_generated",
                "85,4 %",
                "FP; ResNet класифікує правильно",
            ],
            [
                "Обидві",
                "test_real_0001.jpg",
                "real",
                "ai_generated",
                "60–90 %",
                "Складний зразок домену",
            ],
        ],
        leading_blank=leading_blank,
    )


def _add_bootstrap_ci_table(doc: Document, *, leading_blank: bool = True) -> None:
    _add_report_table(
        doc,
        "Таблиця 5.5 – Bootstrap 95 % CI для accuracy (test set, N=204)",
        ["Модель", "Accuracy (point)", "CI 95 % (нижня)", "CI 95 % (верхня)", "Ресемплів"],
        [
            ["ResNet18", "93,6 %", "90,2 %", "96,6 %", "2000"],
            ["EfficientNet-B0", "98,5 %", "96,6 %", "100,0 %", "2000"],
        ],
        leading_blank=leading_blank,
    )


def _load_ablation_factorial_rows() -> list[list[str]]:
    artifacts = Path(__file__).resolve().parents[1] / "docs" / "artifacts" / "ablation_runs.json"
    if not artifacts.is_file():
        return [["10", "так", "9", "98,5 %", "0,986", "3"]]
    rows = json.loads(artifacts.read_text(encoding="utf-8"))
    order = {"baseline": 0, "no_augment": 1, "epochs5": 2}
    rows = sorted(rows, key=lambda r: order.get(r["key"], 99))
    table: list[list[str]] = []
    for row in rows:
        aug = "так" if row["strong_augment"] else "ні"
        table.append(
            [
                str(row["epochs"]),
                aug,
                str(row.get("best_epoch", "—")),
                f"{row['test_accuracy'] * 100:.1f} %".replace(".", ","),
                f"{row['test_f1']:.3f}".replace(".", ","),
                str(row.get("errors", "—")),
            ],
        )
    return table


def _add_ablation_table(doc: Document, *, leading_blank: bool = True) -> None:
    _add_report_table(
        doc,
        "Таблиця 5.6 – Ablation-сітка EfficientNet-B0 (test set, N=204)",
        ["Epochs", "strong_augment", "Best ep.", "Test accuracy", "F1", "Помилок"],
        _load_ablation_factorial_rows(),
        leading_blank=leading_blank,
    )


def _add_generalization_table(doc: Document, *, leading_blank: bool = True) -> None:
    _add_report_table(
        doc,
        "Таблиця 5.4 – In-domain vs зовнішня вибірка (EfficientNet-B0)",
        ["Вибірка", "N", "Accuracy", "F1", "ROC-AUC", "Примітка"],
        [
            [
                "Test set (Parveshiiii/AI-vs-Real)",
                "204",
                "98,5 %",
                "0,986",
                "1,000",
                "In-domain, той самий датасет",
            ],
            [
                "real_world (data/real_world/)",
                "30",
                "80,0 %",
                "0,800",
                "0,858",
                "15 real + 15 AI, поза HF split; джерела — stock/profile",
            ],
        ],
        leading_blank=leading_blank,
    )


def _add_deployment_table(doc: Document, *, leading_blank: bool = True) -> None:
    _add_report_table(
        doc,
        "Таблиця 7.1 – План впровадження системи детекції AI-зображень",
        ["Назва етапу", "Дата початку", "Дата завершення"],
        [
            ["Аналіз предметної галузі та постановка задачі", "03.04.2026", "17.04.2026"],
            ["Проєктування архітектури та підготовка даних", "18.04.2026", "01.05.2026"],
            ["Реалізація пайплайну train/evaluate", "02.05.2026", "15.05.2026"],
            ["Навчання моделей та порівняння результатів", "16.05.2026", "25.05.2026"],
            ["Gradio demo та модульне тестування", "26.05.2026", "31.05.2026"],
            ["Оформлення документації та звіту ККП", "01.06.2026", "05.06.2026"],
            ["Підготовка до захисту та демонстрація", "06.06.2026", "10.06.2026"],
        ],
        center_from=1,
        leading_blank=leading_blank,
    )


def _add_test_case_demo(
    doc: Document,
    *,
    leading_blank: bool = True,
    trailing_blank: bool = True,
) -> None:
    _add_test_case_table(
        doc,
        "6.1",
        "Тест-кейс №1 — перевірка веб-інтерфейсу",
        [
            ("Ідентифікатор тесту", "TC-DEMO-01"),
            ("Опис функції", "Класифікація зображення через веб-інтерфейс http://127.0.0.1:7860/"),
            ("Передумови", "make demo; checkpoint-и ResNet18 та EfficientNet-B0 навчені"),
            (
                "Кроки виконання",
                "1. Запустити make demo. 2. Обрати EfficientNet-B0. "
                "3. Завантажити JPEG/PNG. 4. Натиснути «Перевірити».",
            ),
            (
                "Очікуваний результат",
                "Відображається verdict «Реальне фото» або «AI-генерація» "
                "та confidence score у відсотках.",
            ),
        ],
        leading_blank=leading_blank,
        trailing_blank=trailing_blank,
    )


def _add_test_case_metrics(
    doc: Document,
    *,
    leading_blank: bool = True,
    trailing_blank: bool = True,
) -> None:
    _add_test_case_table(
        doc,
        "6.2",
        "Тест-кейс №2 — модульні метрики",
        [
            ("Ідентифікатор тесту", "TC-METRICS-01"),
            (
                "Опис функції",
                "Перевірка compute_classification_metrics при ідеальних передбаченнях",
            ),
            ("Передумови", "pytest, модуль kkp.training.metrics"),
            (
                "Кроки виконання",
                "1. Запустити make test. 2. Перевірити test_compute_classification_metrics_perfect_predictions.",
            ),
            (
                "Очікуваний результат",
                "accuracy = precision = recall = F1 = ROC-AUC = 1.0; confusion matrix = [[2,0],[0,2]].",
            ),
        ],
        leading_blank=leading_blank,
        trailing_blank=trailing_blank,
    )


def _add_test_case_deploy(
    doc: Document,
    *,
    leading_blank: bool = True,
    trailing_blank: bool = True,
) -> None:
    _add_test_case_table(
        doc,
        "7.2",
        "Тест-кейс №1 — розгортання Docker",
        [
            ("Ідентифікатор тесту", "TC-DEPLOY-01"),
            ("Опис функції", "Збірка Docker-образу та запуск тестів у контейнері"),
            ("Передумови", "Docker Desktop, docker compose, .env скопійовано з .env.example"),
            (
                "Кроки виконання",
                "1. docker compose build. 2. docker compose run --rm test. "
                "3. docker compose run --rm lint.",
            ),
            (
                "Очікуваний результат",
                "Усі pytest-тести проходять; Ruff lint/format без помилок; образ збирається успішно.",
            ),
        ],
        leading_blank=leading_blank,
        trailing_blank=trailing_blank,
    )


def _add_test_case_inference(
    doc: Document,
    *,
    leading_blank: bool = True,
    trailing_blank: bool = True,
) -> None:
    _add_test_case_table(
        doc,
        "7.3",
        "Тест-кейс №2 — inference на test set",
        [
            ("Ідентифікатор тесту", "TC-DEPLOY-02"),
            ("Опис функції", "Оцінка навченої моделі на test set і експорт metrics.json"),
            ("Передумови", "Навчений checkpoint у outputs/ai_generated_efficientnet/checkpoints/"),
            (
                "Кроки виконання",
                "1. make evaluate-efficientnet. 2. Перевірити outputs/.../metrics.json. "
                "3. make compare.",
            ),
            (
                "Очікуваний результат",
                "accuracy ≥ 95 %; створено PNG-графіки в outputs/comparison/.",
            ),
        ],
        leading_blank=leading_blank,
        trailing_blank=trailing_blank,
    )


_TABLE_BUILDERS = {
    "dataset": _add_dataset_table,
    "hyperparams": _add_hyperparams_table,
    "models_theory": _add_models_theory_table,
    "metrics": _add_metrics_table,
    "error_analysis": _add_error_analysis_table,
    "bootstrap_ci": _add_bootstrap_ci_table,
    "generalization": _add_generalization_table,
    "ablation": _add_ablation_table,
    "deployment": _add_deployment_table,
    "test_case_demo": _add_test_case_demo,
    "test_case_metrics": _add_test_case_metrics,
    "test_case_deploy": _add_test_case_deploy,
    "test_case_inference": _add_test_case_inference,
}


_TEST_CASE_TABLE_KEYS = frozenset(
    {"test_case_demo", "test_case_metrics", "test_case_deploy", "test_case_inference"},
)

_FIGURE_REF_NUMBERS: dict[str, str] = {
    "use_cases": "3.1",
    "pipeline": "3.2",
    "modules": "3.3",
    "demo_overview": "4.1",
    "demo_result": "4.2",
    "metrics_comparison": "5.3",
    "learning_curves": "5.4",
    "confusion_matrices": "5.5",
    "roc_curves": "5.6",
    "misclassification_examples": "5.7",
    "gradcam_examples": "5.8",
    "testing_pyramid": "6.1",
    "ui_mockup": "3.4",
}

_TABLE_REF_NUMBERS: dict[str, str] = {
    "dataset": "3.1",
    "hyperparams": "4.1",
    "models_theory": "5.1",
    "metrics": "5.2",
    "error_analysis": "5.3",
    "generalization": "5.4",
    "bootstrap_ci": "5.5",
    "ablation": "5.6",
    "deployment": "7.1",
    "test_case_demo": "6.1",
    "test_case_metrics": "6.2",
    "test_case_deploy": "7.2",
    "test_case_inference": "7.3",
}

_FIGURE_REF_RE = re.compile(
    r"рис\.?\s*(\d+\.\d+)(?:\s*[–\-]\s*(\d+\.\d+))?",
    re.IGNORECASE,
)
_TABLE_REF_RE = re.compile(
    r"(?:табл\.?|таблиця)\s*(\d+\.\d+)(?:\s*[–\-]\s*(\d+\.\d+))?",
    re.IGNORECASE,
)


def _ref_tuple(number: str) -> tuple[int, int]:
    major, minor = number.split(".", 1)
    return int(major), int(minor)


def _ref_in_range(number: str, start: str, end: str | None) -> bool:
    target = _ref_tuple(number)
    start_ref = _ref_tuple(start)
    if end is None:
        return target == start_ref
    end_ref = _ref_tuple(end)
    return start_ref <= target <= end_ref


def _text_mentions_figure(text: str, number: str) -> bool:
    for match in _FIGURE_REF_RE.finditer(text.lower()):
        if _ref_in_range(number, match.group(1), match.group(2)):
            return True
    return False


def _text_mentions_table(text: str, number: str) -> bool:
    for match in _TABLE_REF_RE.finditer(text.lower()):
        if _ref_in_range(number, match.group(1), match.group(2)):
            return True
    return False


def _block_ref_number(item: FigureBlock | TableBlock) -> str | None:
    if isinstance(item, FigureBlock):
        return _FIGURE_REF_NUMBERS.get(item.key)
    return _TABLE_REF_NUMBERS.get(item.key)


def _text_mentions_block(text: str, item: FigureBlock | TableBlock) -> bool:
    number = _block_ref_number(item)
    if number is None:
        return False
    if isinstance(item, FigureBlock):
        return _text_mentions_figure(text, number)
    return _text_mentions_table(text, number)


def _first_mention_index(items: list, item: FigureBlock | TableBlock) -> int | None:
    for index, candidate in enumerate(items):
        if isinstance(candidate, str) and _text_mentions_block(candidate, item):
            return index
    return None


def _reorder_figures_tables_after_mentions(items: list) -> list:
    """Place each FigureBlock/TableBlock immediately after its first in-text mention."""
    blocks = [
        (index, item)
        for index, item in enumerate(items)
        if isinstance(item, FigureBlock | TableBlock)
    ]
    if not blocks:
        return items

    insert_after: dict[int, list[FigureBlock | TableBlock]] = {}
    trailing: list[FigureBlock | TableBlock] = []

    for block_index, block in blocks:
        mention_index = _first_mention_index(items, block)
        if mention_index is None:
            trailing.append(block)
            continue
        if block_index <= mention_index:
            insert_after.setdefault(mention_index, []).append(block)

    if not insert_after:
        return items

    placed_keys = {
        (type(block), block.key) for block_list in insert_after.values() for block in block_list
    }
    ordered: list = []
    for index, item in enumerate(items):
        if isinstance(item, FigureBlock | TableBlock):
            if (type(item), item.key) in placed_keys:
                continue
            ordered.append(item)
            continue
        ordered.append(item)
        for block in insert_after.get(index, []):
            ordered.append(block)

    for block in trailing:
        if (type(block), block.key) not in placed_keys:
            ordered.append(block)
    return ordered


def _render_section_items(
    doc: Document,
    items: list,
    *,
    figure_registry: dict[str, tuple],
    include_figures: bool = True,
) -> None:
    items = _reorder_figures_tables_after_mentions(items)
    for index, item in enumerate(items):
        next_item = items[index + 1] if index + 1 < len(items) else None
        if isinstance(item, str):
            _add_body(doc, item)
        elif isinstance(item, MathBlock):
            _add_math(doc, item.latex)
        elif isinstance(item, CodeBlock):
            _add_code_block(doc, item.code)
        elif isinstance(item, FigureBlock):
            entry = figure_registry.get(item.key)
            if entry is None:
                continue
            path, caption = entry
            prev = items[index - 1] if index > 0 else None
            leading_blank = not isinstance(prev, FigureBlock | str)
            blank_after_caption = isinstance(next_item, str)
            tight_after = isinstance(next_item, CodeBlock | MathBlock)
            if not include_figures:
                _add_figure_caption_only(
                    doc,
                    caption,
                    leading_blank=leading_blank,
                    tight_after=tight_after,
                    blank_after_caption=blank_after_caption,
                )
                continue
            if path.is_file():
                width_cm = {
                    "testing_pyramid": 11.0,
                    "pipeline": 17.5,
                    "ui_mockup": 14.0,
                }.get(item.key, 14.0)
                _add_figure(
                    doc,
                    path,
                    caption,
                    leading_blank=leading_blank,
                    tight_after=tight_after,
                    blank_after_caption=blank_after_caption,
                    width_cm=width_cm,
                )
        elif isinstance(item, TableBlock):
            builder = _TABLE_BUILDERS.get(item.key)
            if builder is None:
                continue
            if item.key in _TEST_CASE_TABLE_KEYS:
                prev = items[index - 1] if index > 0 else None
                leading_blank = not (
                    isinstance(prev, TableBlock) and prev.key in _TEST_CASE_TABLE_KEYS
                )
                builder(doc, leading_blank=leading_blank, trailing_blank=True)
            else:
                builder(doc, leading_blank=True)


def append_main_body(
    doc: Document,
    meta: KkpMeta,
    *,
    comparison_dir,
    assets_dir,
    include_figures: bool = True,
) -> None:
    _cleanup_before_abbreviations(doc)
    _replace_abbreviations(doc)
    _remove_example_body(doc)

    perelik = _find_paragraph_index(doc, "ПЕРЕЛІК СКОРОЧЕНЬ")
    if perelik is not None:
        while len(doc.paragraphs) > perelik + 1 and _paragraph_is_truly_empty(doc.paragraphs[-1]):
            _delete_paragraph(doc.paragraphs[-1])

    _add_heading(doc, "ВСТУП", gap_before_body=True)
    for paragraph in intro_paragraphs(meta):
        _add_body(doc, paragraph)

    comparison_dir = Path(comparison_dir)
    assets_dir = Path(assets_dir)
    artifacts_dir = assets_dir.parent / "artifacts"

    def _plot_path(name: str) -> Path:
        primary = comparison_dir / name
        if primary.is_file():
            return primary
        fallback = artifacts_dir / name
        return fallback if fallback.is_file() else primary

    figure_registry = {
        "use_cases": (
            assets_dir / "use_cases.png",
            "Рис. 3.1 – Діаграма сценаріїв використання (use case)",
        ),
        "pipeline": (assets_dir / "pipeline.png", "Рис. 3.2 – Пайплайн навчання та експлуатації"),
        "modules": (assets_dir / "modules.png", "Рис. 3.3 – Файлова структура проєкту KKP"),
        "ui_mockup": (
            assets_dir / "ui_mockup.png",
            "Рис. 3.4 – Макет веб-інтерфейсу (wireframe)",
        ),
        "testing_pyramid": (
            assets_dir / "testing_pyramid.png",
            "Рис. 6.1 – Рівні тестування програмного комплексу",
        ),
        "demo_overview": (
            assets_dir / "demo_overview.png",
            "Рис. 4.1 – Веб-інтерфейс системи (http://127.0.0.1:7860/)",
        ),
        "demo_result": (
            assets_dir / "demo_result.png",
            "Рис. 4.2 – Результат класифікації (predict_image, inference.py; make demo)",
        ),
        "metrics_comparison": (
            _plot_path("metrics_comparison.png"),
            "Рис. 5.3 – Порівняння метрик на test set (docs/artifacts/*_test_metrics.json, compare.py)",
        ),
        "learning_curves": (
            _plot_path("learning_curves.png"),
            "Рис. 5.4 – Криві навчання (outputs/*/history.json, compare.py)",
        ),
        "confusion_matrices": (
            _plot_path("confusion_matrices.png"),
            "Рис. 5.5 – Матриці помилок на test set (поле confusion_matrix у metrics.json)",
        ),
        "roc_curves": (
            _plot_path("roc_curves.png"),
            "Рис. 5.6 – ROC-криві (fpr/tpr з metrics.json, roc_auc_score у metrics.py)",
        ),
        "misclassification_examples": (
            assets_dir / "misclassification_examples.png",
            "Рис. 5.7 – Приклади помилкових класифікацій на test set",
        ),
        "gradcam_examples": (
            assets_dir / "gradcam_examples.png",
            "Рис. 5.8 – Grad-CAM для типових помилок (gradcam.py, target layer → heatmap)",
        ),
    }

    for section_title, blocks in report_sections(meta):
        _add_heading(doc, section_title)
        for subtitle, items in blocks:
            if subtitle:
                _add_subheading(doc, subtitle)
            _render_section_items(
                doc, items, figure_registry=figure_registry, include_figures=include_figures
            )

    _add_heading(doc, "ВИСНОВКИ", gap_before_body=True)
    for paragraph in conclusions(meta):
        _add_body(doc, paragraph)

    _add_heading(doc, "ПЕРЕЛІК ДЖЕРЕЛ ПОСИЛАННЯ", gap_before_body=True)
    for index, ref in enumerate(references(meta), 1):
        _add_body(doc, f"{index}. {ref}")


def fill_kkp_document(
    doc: Document,
    meta: KkpMeta,
    *,
    comparison_dir,
    assets_dir,
    include_figures: bool = True,
) -> None:
    fill_front_matter(doc, meta)
    _trim_abstract_before_toc(doc)
    append_main_body(
        doc,
        meta,
        comparison_dir=comparison_dir,
        assets_dir=assets_dir,
        include_figures=include_figures,
    )
    stats = _compute_report_stats(doc, meta, include_figures=include_figures)
    fill_abstract(doc, meta, stats=stats)
    _apply_major_section_page_breaks(doc)
    _ensure_page_break_before_heading(doc, "ЗМІСТ")
    fill_table_of_contents(doc)
    _strip_document_highlights(doc)
