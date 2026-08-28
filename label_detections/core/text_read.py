"""Deciding what a line of read text says about a label's identity.

The same job ``core/codes`` does for a barcode, for labels whose code the
camera cannot resolve. On the rig this was written for, the UPC-A gets 3.35
pixels per module -- below what a photographed, rectified symbol survives --
while the part number printed beside it gets a 40 pixel cap height, which is
comfortable. The barcode is the hardest thing on that label to read, not the
easiest: it packs the same part number into 95 narrow bars where the text
spells it out across three times the width.

What makes this workable is that it is not general text recognition. Nobody
needs to know what the label says; they need to know which of a handful of
ENROLLED part numbers it is, or that it is none of them. A read only has to
land closer to one enrolled string than to any other, by a margin, and match
nothing when the label was never enrolled. That is a far lower bar than
transcription, and it is the bar that matters -- because the failure being
prevented is a label nobody has ever shown the system being reported as one
that was.

The honest cost against a barcode: there is no checksum. A barcode either
decodes correctly or not at all, while a misread character is silent. Matching
against the enrolled set rather than accepting free text is what closes most of
that gap -- a misread that lands near nothing is refused, and a misread that
still lands nearest the right label is right anyway.

Nothing here runs an OCR engine, opens an image, or knows what one looks like.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .live_detect import UNKNOWN

# Policies a text field can carry, matching labels.TextField.
DEMANDING = ("must_be_present", "must_match_pattern")

CONTRADICTED = "contradicted"   # read something, and it is not this label
UNREADABLE = "unreadable"       # text was demanded and none came back
PRESENT = "present"             # read something, nothing to check it against
NOT_CHECKED = "not_checked"     # this label declares no text worth checking
CONFIRMED = "confirmed"         # read something, and it is this label

# How alike a read and an enrolled string must be before the read is that
# label. Printed part numbers are short and distinctive, and OCR on 40 px text
# is nearly exact -- so this is set to tolerate a character or two of damage in
# a ten-character part number, not to guess.
MIN_SIMILARITY = 0.78

# And how far ahead of the runner-up. Two enrolled labels whose part numbers
# differ by one character are a real thing, and a read damaged in exactly that
# character sits equally close to both. Picking the higher score there would be
# choosing by noise, so it is refused instead.
MIN_MARGIN = 0.08


@dataclass
class Read:
    """One line of recognised text."""
    text: str = ""
    confidence: float = 0.0

    def __bool__(self) -> bool:
        return bool(str(self.text).strip())


@dataclass
class Verdict:
    """What the read text says about which label this is."""
    state: str = NOT_CHECKED
    label_id: str = ""
    detail: str = ""
    score: float = 0.0
    reads: list = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.state == CONFIRMED

    @property
    def blocks(self) -> bool:
        return self.state in (CONTRADICTED, UNREADABLE)


def normalise(text: str) -> str:
    """What two strings have to share before they are compared.

    Case, spacing and punctuation carry no identity here and are exactly what
    OCR gets wrong first -- a dropped space between the part number and its
    bracket, a full-width parenthesis where a narrow one was printed. Stripping
    them removes a whole class of false mismatch without touching the
    characters that actually distinguish one part from another.

    Digits and letters are deliberately NOT folded together. Mapping O to 0 and
    I to 1 would fix some misreads and would also make genuinely different part
    numbers collide, which is the failure this exists to prevent.
    """
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def similarity(expected: str, read: str) -> float:
    """How well ``expected`` appears anywhere in ``read``, from 0 to 1.

    Anywhere, not end to end: the crop is a region of a label and will often
    carry more than the part number -- a voltage, a bracketed alias, whatever
    sits on the same line. Comparing the whole read against a short part number
    would score a correct read badly for the crime of containing extra words.
    """
    want, got = normalise(expected), normalise(read)
    if not want or not got:
        return 0.0
    if want in got:
        return 1.0
    if len(got) <= len(want):
        return SequenceMatcher(None, want, got).ratio()
    # Slide a window of the expected length and keep the best fit.
    best = 0.0
    for start in range(len(got) - len(want) + 1):
        window = got[start:start + len(want)]
        best = max(best, SequenceMatcher(None, want, window).ratio())
        if best == 1.0:
            break
    return best


def best_match(read: str, expected: dict[str, list[str]]) -> tuple[str, float, float]:
    """``(label_id, score, runner_up_score)`` for the closest enrolled label."""
    scored: list[tuple[float, str]] = []
    for label_id, wanted in sorted((expected or {}).items()):
        best = max((similarity(w, read) for w in wanted or [] if w), default=0.0)
        if best > 0:
            scored.append((best, str(label_id)))
    if not scored:
        return "", 0.0, 0.0
    scored.sort(reverse=True)
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    return scored[0][1], scored[0][0], runner_up


def demanded(fields) -> list:
    """The text fields on a label that inspection must do something about."""
    return [f for f in (fields or [])
            if str(getattr(f, "policy", "") or "") in DEMANDING]


def verdict(proposed: str, fields, reads, expected: dict[str, list[str]]) -> Verdict:
    """What the read text says about a detection the classifier called ``proposed``.

    ``expected`` is every enrolled label's strings, not just this one's -- the
    same rule the code reader follows, and for the same reason: a read that
    belongs to a different enrolled label should relabel the detection rather
    than merely fail it, and a read that belongs to none of them is the case
    this was built for.
    """
    reads = [r for r in (reads or []) if r]
    wanted = demanded(fields)
    if not wanted:
        return Verdict(NOT_CHECKED, proposed,
                       "no text on this label is marked for inspection",
                       reads=reads)
    if not reads:
        return Verdict(UNREADABLE, "",
                       f"{len(wanted)} text field(s) required, none read",
                       reads=reads)

    joined = " ".join(str(r.text) for r in reads)
    if not any(expected.get(label) for label in expected):
        return Verdict(PRESENT, proposed,
                       f"text was read but no label declares what to expect",
                       reads=reads)

    label_id, score, runner_up = best_match(joined, expected)
    if score < MIN_SIMILARITY:
        return Verdict(CONTRADICTED, "", score=score, reads=reads,
                       detail=f"read '{_short(joined)}', nearest enrolled label "
                              f"{label_id or '-'} at {score:.0%} -- under the "
                              f"{MIN_SIMILARITY:.0%} needed to call it that")
    if score - runner_up < MIN_MARGIN:
        return Verdict(CONTRADICTED, "", score=score, reads=reads,
                       detail=f"read '{_short(joined)}', which sits equally "
                              f"close to more than one enrolled label "
                              f"({score:.0%} against {runner_up:.0%}) -- too "
                              f"close to call")
    if label_id == proposed:
        return Verdict(CONFIRMED, proposed, "the printed text matches this label",
                       score=score, reads=reads)
    return Verdict(CONTRADICTED, label_id, score=score, reads=reads,
                   detail=f"the printed text reads as {label_id}, not {proposed}")


def _short(text: str, limit: int = 32) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit - 1] + "…"


def resolve(proposed: str, verdict_: Verdict) -> str:
    """The identity to report once the text has had its say."""
    if verdict_.state in (CONFIRMED, PRESENT, NOT_CHECKED):
        return proposed
    if verdict_.state == CONTRADICTED and verdict_.label_id:
        return verdict_.label_id
    return UNKNOWN


def plate_note(verdict_: Verdict, proposed: str = "") -> str:
    """The short form for the drawn plate."""
    whose = f" ({proposed})" if proposed else ""
    return {
        CONFIRMED: "text ok",
        CONTRADICTED: "WRONG TEXT" if verdict_.label_id else f"NO MATCH{whose}",
        UNREADABLE: f"no text{whose}",
        PRESENT: "text unchecked",
    }.get(verdict_.state, "")


def expected_for(labels) -> dict[str, list[str]]:
    """``{label_id: [string, ...]}`` -- what each label's text should read.

    A field's ``pattern`` is the string to look for. It is a regex in the
    library, and a plain part number is a valid regex for itself, so the
    common case needs nothing special; the anchors that a regex might carry are
    stripped, since this compares text rather than matching a rule.
    """
    out: dict[str, list[str]] = {}
    for label in labels or []:
        found = []
        for spec in demanded(getattr(label, "text_fields", None)):
            pattern = str(getattr(spec, "pattern", "") or "").strip()
            if pattern:
                found.append(pattern.lstrip("^").rstrip("$"))
        if found:
            out[str(getattr(label, "label_id", ""))] = found
    return out
