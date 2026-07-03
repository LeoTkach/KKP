"""Metadata for KKP report and presentation generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KkpMeta:
    topic: str
    student_full: str
    student_title_name: str
    student_dative: str
    student_short: str
    group: str
    course: str
    semester: str
    faculty: str
    department: str
    specialty: str
    program_type: str
    educational_program: str
    advisor_line: str
    advisor_short: str
    assignment_date: str
    submission_date: str
    year: str
    city: str
    github: str
    initial_data: str
    work_questions: str
    commission_members: tuple[str, ...]


DEFAULT_META = KkpMeta(
    topic="Система детекції AI-генерованих зображень",
    student_full="Ткач Леонід Ярославович",
    student_title_name="Леонід ТКАЧ",
    student_dative="Ткачу Леоніду Ярославовичу",
    student_short="Ткач Л. Я.",
    group="ПЗПІ-23-5",
    course="3",
    semester="6",
    faculty="комп'ютерних наук",
    department="програмної інженерії",
    specialty="121 – Інженерія програмного забезпечення",
    program_type="Освітньо-професійна",
    educational_program="Програмна інженерія",
    advisor_line="доц. кафедри ПІ Ольга ВОРОЧЕК",
    advisor_short="доц. кафедри ПІ О. ВОРОЧЕК",
    assignment_date="03.04.2026",
    submission_date="10.06.2026",
    year="2026",
    city="Харків",
    github="https://github.com/LeoTkach/KKP",
    initial_data=(
        "датасет Parveshiiii/AI-vs-Real (Hugging Face), методичні вказівки до ККП, "
        "технічне завдання на ML-пайплайн."
    ),
    work_questions=(
        "аналіз предметної галузі; постановка задачі; проєктування архітектури ПЗ; "
        "навчання та порівняння CNN-моделей; аналіз результатів; Gradio demo; тестування та CI."
    ),
    commission_members=(
        "Володимир КОБЗЄВ",
        "Віталій КАУК",
        "Дмитро КОЛЕСНИКОВ",
    ),
)
