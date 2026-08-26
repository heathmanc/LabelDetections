from __future__ import annotations

from label_detections.core.wizard import Flow, Page, Question


def flow():
    return Flow("f", "Test", pages=[
        Page("one", "One", questions=[
            Question("name", "Name", "text", required=True),
            Question("has_code", "Has a code", "bool", default=False),
            Question("symbology", "Symbology", "choice", choices=["qr", "datamatrix"],
                     visible_when={"has_code": True}, required=True),
        ]),
        Page("two", "Two", visible_when={"has_code": True}, questions=[
            Question("rows", "Rows", "table", columns=[
                Question("view", "View", "choice", required=True,
                         choices_from="views", choices_field="view"),
                Question("n", "N", "int", validator=lambda v, a: "" if int(v) > 0 else "N must be positive"),
            ]),
        ]),
    ])


def test_defaults_cover_every_question():
    answers = flow().defaults()
    assert answers["name"] == "" and answers["has_code"] is False
    assert answers["rows"] == []


def test_conditional_questions_stay_hidden_and_unvalidated():
    """Asking every question every time is how operators learn to click Next."""
    f = flow()
    answers = f.defaults()
    answers["name"] = "x"
    assert f.validate(answers) == []
    page = f.pages[0]
    assert [q.key for q in page.visible_questions(answers)] == ["name", "has_code"]


def test_revealing_a_question_also_makes_it_required():
    f = flow()
    answers = f.defaults()
    answers.update(name="x", has_code=True)
    assert any("Symbology" in e for e in f.validate(answers))


def test_a_hidden_page_is_skipped_entirely():
    f = flow()
    answers = f.defaults()
    answers["name"] = "x"
    assert [p.key for p in f.visible_pages(answers)] == ["one"]


def test_table_rows_are_validated_against_their_columns_with_row_numbers():
    f = flow()
    answers = f.defaults()
    answers.update(name="x", has_code=True, symbology="qr",
                   views=[{"view": "side_a"}],
                   rows=[{"view": "side_a", "n": 1}, {"view": "ghost", "n": 0}])
    errors = f.validate(answers)
    assert any("row 2" in e and "ghost" in e for e in errors)
    assert any("row 2" in e and "positive" in e for e in errors)
    assert not any("row 1" in e for e in errors)


def test_dynamic_choices_come_from_earlier_answers():
    column = flow().pages[1].questions[0].columns[0]
    assert column.resolve_choices({"views": [{"view": "a"}, {"view": "b"}]}) == ["a", "b"]
    assert column.resolve_choices({}) == []


def test_notes_are_advisory_and_never_block_finishing():
    f = flow()
    f.review = lambda answers: ["a suggestion"]
    answers = f.defaults()
    answers["name"] = "x"
    assert f.validate(answers) == []
    assert f.notes(answers) == ["a suggestion"]


def test_to_text_renders_the_questionnaire_without_a_gui():
    text = flow().to_text()
    assert "Test" in text and "Name" in text
    assert "Symbology" not in text        # hidden at defaults
