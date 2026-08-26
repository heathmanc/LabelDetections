"""Print a wizard's questions to a terminal.

The questionnaires *are* the domain knowledge this tool captures, so they need
to be reviewable by people who will never install Qt -- process engineers,
quality, whoever decides what a label must carry.

    python -m label_detections.preview labels
    python -m label_detections.preview recipe
"""
from __future__ import annotations

import sys

from .core import label_wizard, recipe_wizard

FLOWS = {"labels": label_wizard.FLOW, "label": label_wizard.FLOW,
         "recipe": recipe_wizard.FLOW, "recipes": recipe_wizard.FLOW}


def main(argv: list[str] | None = None) -> int:
    args = (argv if argv is not None else sys.argv[1:]) or ["labels"]
    flow = FLOWS.get(args[0].lower())
    if flow is None:
        print(f"Unknown wizard '{args[0]}'. Choose one of: {', '.join(sorted(set(FLOWS)))}",
              file=sys.stderr)
        return 2
    answers = flow.defaults()
    # Reveal the conditional questions too: a printed questionnaire that hides
    # half its questions is not a review of the questionnaire.
    for question in flow.questions():
        for key, expected in question.visible_when.items():
            answers[key] = expected[0] if isinstance(expected, (list, tuple)) else expected
    print(flow.to_text(answers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
