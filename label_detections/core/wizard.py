"""A tiny declarative wizard framework.

Both wizards in this tool -- add a label, build a recipe -- are defined as
*data*: a list of pages, each a list of questions. The Qt layer renders
whatever it is handed and knows nothing about labels or recipes, so:

* adding a question is a one-line data change, not a UI patch;
* the question set is unit testable, which matters because the questions *are*
  the domain knowledge being captured;
* the whole questionnaire can be printed to a terminal for review by someone
  who will never install PySide6.

Conditional questions are supported because the alternative -- asking every
question every time -- is how operators learn to click Next without reading.
A label with no barcode should never see a symbology field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# What the Qt layer must know how to render.
KINDS = (
    "text", "textarea", "int", "float", "bool", "choice", "multichoice",
    "path", "paths", "rect_mm", "size_mm", "roi", "frame_size", "table",
    "label_picker", "count",
)


@dataclass
class Question:
    key: str
    label: str
    kind: str = "text"
    help: str = ""
    default: Any = None
    choices: list[Any] = field(default_factory=list)
    # Choices that only exist once earlier answers are given -- the views a
    # recipe declared, the labels already in the library. Names another answer
    # key; ``choices_field`` pulls one column when that answer is a table.
    choices_from: str = ""
    choices_field: str = ""
    required: bool = False
    # Shown only when every entry matches the current answers. A list value
    # means "any of these".
    visible_when: dict[str, Any] = field(default_factory=dict)
    # Columns for kind == "table"; each is itself a Question.
    columns: list["Question"] = field(default_factory=list)
    # Returns an error string, or "" when the answer is acceptable.
    validator: Callable[[Any, dict[str, Any]], str] | None = None
    placeholder: str = ""

    def is_visible(self, answers: dict[str, Any]) -> bool:
        for key, expected in self.visible_when.items():
            got = answers.get(key)
            if isinstance(expected, (list, tuple, set)):
                if got not in expected:
                    return False
            elif got != expected:
                return False
        return True

    def resolve_choices(self, answers: dict[str, Any]) -> list[Any]:
        """Static choices, or the ones implied by answers given earlier."""
        if not self.choices_from:
            return list(self.choices)
        source = answers.get(self.choices_from)
        if isinstance(source, list):
            if self.choices_field:
                return [row.get(self.choices_field) for row in source
                        if isinstance(row, dict) and row.get(self.choices_field)]
            return [v for v in source if v not in (None, "")]
        return list(self.choices)

    def blank(self) -> Any:
        if self.default is not None:
            return self.default
        return {
            "bool": False, "int": 0, "float": 0.0,
            "multichoice": [], "paths": [], "table": [],
            "rect_mm": [], "size_mm": [0.0, 0.0], "roi": [],
            "frame_size": [0, 0], "count": 1,
        }.get(self.kind, "")


@dataclass
class Page:
    key: str
    title: str
    questions: list[Question] = field(default_factory=list)
    blurb: str = ""
    visible_when: dict[str, Any] = field(default_factory=dict)

    def is_visible(self, answers: dict[str, Any]) -> bool:
        for key, expected in self.visible_when.items():
            got = answers.get(key)
            if isinstance(expected, (list, tuple, set)):
                if got not in expected:
                    return False
            elif got != expected:
                return False
        return True

    def visible_questions(self, answers: dict[str, Any]) -> list[Question]:
        return [q for q in self.questions if q.is_visible(answers)]


def _question_lines(question: "Question", answers: dict[str, Any],
                    indent: str) -> list[str]:
    """One question as text, with a table's columns nested underneath it."""
    flag = "*" if question.required else " "
    detail = f"[{question.kind}]"
    choices = question.resolve_choices(answers) or question.choices
    if choices:
        detail += " " + "/".join(str(c) for c in choices)
    elif question.choices_from:
        detail += f" (from {question.choices_from})"
    lines = [f"{indent}{flag} {question.label}  {detail}"]
    if question.help:
        lines.append(f"{indent}    {question.help}")
    for column in question.columns:
        lines.extend(_question_lines(column, answers, indent + "      "))
    return lines


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def _validate_table(question: "Question", rows: list[Any],
                    answers: dict[str, Any]) -> list[str]:
    """Apply a table's column rules to each of its rows.

    Errors are prefixed with the row number: an operator who filled in six
    barcodes needs to know which one is wrong, not that one of them is.
    """
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            errors.append(f"{question.label} row {index}: not a valid row.")
            continue
        for column in question.columns:
            if not column.is_visible({**answers, **row}):
                continue
            value = row.get(column.key)
            if column.required and _is_blank(value):
                errors.append(f"{question.label} row {index}: {column.label} is required.")
                continue
            if _is_blank(value):
                continue
            choices = column.resolve_choices(answers)
            if column.kind in ("choice",) and choices and value not in choices:
                errors.append(
                    f"{question.label} row {index}: {column.label} '{value}' is not one of "
                    + ", ".join(str(c) for c in choices) + "."
                )
            if column.validator:
                message = column.validator(value, {**answers, **row})
                if message:
                    errors.append(f"{question.label} row {index}: {message}")
    return errors


@dataclass
class Flow:
    """A whole questionnaire, plus how to turn its answers into a domain object."""
    key: str
    title: str
    pages: list[Page] = field(default_factory=list)
    build: Callable[[dict[str, Any]], Any] | None = None
    # Advisory cross-question notes that no single field validator can see.
    # Deliberately separate from validation: these warn, they never block. An
    # operator who cannot finish a wizard because of a suggestion will find a
    # way around the wizard.
    review: Callable[[dict[str, Any]], list[str]] | None = None

    def visible_pages(self, answers: dict[str, Any]) -> list[Page]:
        return [p for p in self.pages if p.is_visible(answers)]

    def questions(self) -> list[Question]:
        return [q for p in self.pages for q in p.questions]

    def question(self, key: str) -> Question | None:
        for q in self.questions():
            if q.key == key:
                return q
        return None

    def defaults(self) -> dict[str, Any]:
        return {q.key: q.blank() for q in self.questions()}

    def validate_page(self, page: Page, answers: dict[str, Any]) -> list[str]:
        """Errors on one page. Empty means Next is allowed."""
        errors: list[str] = []
        for question in page.visible_questions(answers):
            value = answers.get(question.key)
            if question.required and _is_blank(value):
                errors.append(f"{question.label} is required.")
                continue
            if question.kind == "table" and isinstance(value, list):
                errors.extend(_validate_table(question, value, answers))
            if question.validator and not _is_blank(value):
                message = question.validator(value, answers)
                if message:
                    errors.append(message)
        return errors

    def validate(self, answers: dict[str, Any]) -> list[str]:
        """Hard errors across every visible page. Empty means Finish is allowed."""
        errors: list[str] = []
        for page in self.visible_pages(answers):
            errors.extend(self.validate_page(page, answers))
        return errors

    def notes(self, answers: dict[str, Any]) -> list[str]:
        """Advisory warnings for the summary page. Never blocks finishing."""
        return list(self.review(answers)) if self.review else []

    def result(self, answers: dict[str, Any]) -> Any:
        if self.build is None:
            return dict(answers)
        return self.build(answers)

    def to_text(self, answers: dict[str, Any] | None = None) -> str:
        """The questionnaire as plain text, for review without a GUI."""
        answers = answers if answers is not None else self.defaults()
        lines = [self.title, "=" * len(self.title)]
        for page in self.visible_pages(answers):
            lines.append("")
            lines.append(f"-- {page.title} " + "-" * max(0, 60 - len(page.title)))
            if page.blurb:
                lines.append(f"   {page.blurb}")
            for question in page.visible_questions(answers):
                lines.extend(_question_lines(question, answers, indent=" "))
        return "\n".join(lines)
