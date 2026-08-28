"""Deciding what a decoded barcode says about a label's identity.

Every other check in this pipeline is a *rejection heuristic*. The detector
says a label is probably there; the classifier says it most resembles PC680;
the novelty profile says it sits near where PC680 crops sit. All three answer
"what does this look like", and all three can be confidently wrong about a
label nobody has ever shown the system -- which is exactly the failure this
line cannot afford, because a wrong id is indistinguishable downstream from a
correct read.

A decoded code is different in kind. It is not an opinion about appearance, it
is the part number printed on the part. It does not need to have seen the
wrong label before in order to refuse it, because the refusal comes from the
label's own printing rather than from a boundary drawn around training data.
That is the whole reason this is worth building over another model.

The library has described this since the label wizard was written: every
``CodeSpec`` carries a symbology, a region as fractions of the label, a policy
saying what inspection must be able to do with it, and a regex the decoded
text must match. What was missing was the part that reads it and the part that
decides. This is the deciding half; ``core/code_reader`` is the reading half.

Nothing here imports a decoder, opens an image, or knows what zxing is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .live_detect import UNKNOWN

# Policies that demand something of a code. `ignore` is the fourth, and a
# label carrying only ignored codes is one this check has no opinion about.
DEMANDING = ("must_be_present", "must_decode", "must_match_pattern")

# What a verdict can be. Ordered worst to best for reporting.
CONTRADICTED = "contradicted"   # a code was read and it is not this label's
UNREADABLE = "unreadable"       # a code was demanded and none came back
PRESENT = "present"             # something decoded, nothing to check it against
NOT_CHECKED = "not_checked"     # this label declares no code worth checking
CONFIRMED = "confirmed"         # a code was read and it is this label's


@dataclass
class Read:
    """One decoded symbol."""
    text: str = ""
    symbology: str = ""

    def __bool__(self) -> bool:
        return bool(self.text)

    def candidates(self) -> list[str]:
        """Every spelling of this read a pattern may reasonably be written for.

        One spelling, except for the UPC-A / EAN-13 overlap, which is a trap
        worth spending code on. A UPC-A is a 12-digit code and the label prints
        12 digits, so that is what anyone types a pattern against -- but the
        two symbologies share an encoding, and decoders commonly report a
        UPC-A as its EAN-13 form with a leading zero. The pattern then fails
        on every genuine part, which is the expensive direction: good stock
        rejected, and the readout blaming the printing.

        Only ever adds spellings, never replaces one, so this cannot turn a
        read that already matched into one that does not. The zero-stripped
        form is the same code by definition, so admitting it is not a
        loosening of the check.
        """
        text = str(self.text or "")
        out = [text]
        if len(text) == 13 and text.startswith("0") and text.isdigit():
            out.append(text[1:])          # EAN-13 carrying a UPC-A
        elif len(text) == 12 and text.isdigit():
            out.append("0" + text)        # the same code, written the other way
        return out


@dataclass
class Verdict:
    """What the codes on one detection say about which label it is."""
    state: str = NOT_CHECKED
    # The identity the codes support. The proposed one when confirmed, a
    # different enrolled one when the code belongs to that instead, and empty
    # when the codes support no enrolled label at all.
    label_id: str = ""
    detail: str = ""
    reads: list[Read] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        """Did a code positively identify this, as opposed to not objecting?"""
        return self.state == CONFIRMED

    @property
    def blocks(self) -> bool:
        """Must this detection not be reported as the label proposed?"""
        return self.state in (CONTRADICTED, UNREADABLE)


def matches(pattern: str, text: str) -> bool:
    """Does the decoded text satisfy a label's pattern?

    An empty pattern matches nothing rather than everything. A label that
    declares a code but never says what it should read is not a label this can
    verify, and treating "no rule" as "any code passes" would report the
    strongest state -- confirmed -- on the weakest evidence there is.
    """
    if not pattern or not text:
        return False
    try:
        return re.search(str(pattern), str(text)) is not None
    except re.error:
        # A pattern that does not compile is a data-entry mistake, and matching
        # nothing is the direction that fails loudly rather than passing
        # everything through a rule that was never applied.
        return False


def demanded(specs) -> list:
    """The code specs on a label that inspection must do something about."""
    return [s for s in (specs or [])
            if str(getattr(s, "policy", "") or "") in DEMANDING]


def owners(patterns: dict[str, list[str]], reads) -> list[str]:
    """Every enrolled label whose pattern one of these reads satisfies.

    More than one means two labels claim the same printing, which is a library
    problem rather than a line problem -- and worth surfacing as one, because
    picking either would be arbitrary.
    """
    found = []
    for label_id, label_patterns in sorted((patterns or {}).items()):
        for pattern in label_patterns or []:
            if any(matches(pattern, text)
                   for r in reads or [] for text in r.candidates()):
                found.append(label_id)
                break
    return found


def verdict(proposed: str, specs, reads, patterns: dict[str, list[str]]) -> Verdict:
    """What the codes say about a detection the classifier called ``proposed``.

    ``patterns`` is every enrolled label's patterns, not just this one's. That
    is what lets a read identify rather than only verify: a code belonging to a
    different enrolled label relabels the detection instead of merely failing
    it, and a code belonging to none of them is the case this was built for.
    """
    reads = [r for r in (reads or []) if r]
    wanted = demanded(specs)
    if not wanted:
        return Verdict(NOT_CHECKED, proposed,
                       "no code on this label is marked for inspection", reads)

    if not reads:
        return Verdict(UNREADABLE, "",
                       f"{len(wanted)} code(s) required, none decoded", reads)

    claimed = owners(patterns, reads)
    if proposed in claimed:
        if len(claimed) > 1:
            others = ", ".join(c for c in claimed if c != proposed)
            return Verdict(CONFIRMED, proposed,
                           f"matches {proposed}, but also {others} -- two "
                           f"labels claim this printing", reads)
        return Verdict(CONFIRMED, proposed, "code matches this label", reads)

    if claimed:
        return Verdict(CONTRADICTED, claimed[0],
                       f"the code belongs to {claimed[0]}, not {proposed}", reads)

    # Something decoded and no enrolled label claims it. Either this label has
    # no pattern to check against -- in which case nothing was verified and
    # saying so is the honest answer -- or it has one and the code failed it,
    # which is the case this whole feature exists for.
    if any(getattr(s, "pattern", "") for s in wanted):
        return Verdict(CONTRADICTED, "",
                       f"decoded '{_short(reads[0].text)}', which matches no "
                       f"enrolled label", reads)
    return Verdict(PRESENT, proposed,
                   f"a code is present but {proposed} has no pattern to check "
                   f"it against", reads)


def _short(text: str, limit: int = 24) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit - 1] + "…"


def resolve(proposed: str, verdict_: Verdict) -> str:
    """The identity to report, once the codes have had their say.

    The code outranks the classifier wherever the two disagree. That is the
    point of reading it: the classifier is guessing from appearance and the
    code is printed on the part.
    """
    if verdict_.state in (CONFIRMED, PRESENT, NOT_CHECKED):
        return proposed
    if verdict_.state == CONTRADICTED and verdict_.label_id:
        return verdict_.label_id
    return UNKNOWN


def plate_note(verdict_: Verdict, proposed: str = "") -> str:
    """The short form for the drawn plate, read off a moving part.

    A refusal names the label whose code it wanted. "unknown ... no code" says
    a code is missing without saying whose, and the box has just been stripped
    of the only clue -- the identity it was refused for. A relabel is the one
    case that needs no name, because the box already reads as the label the
    code claimed.
    """
    whose = f" ({proposed})" if proposed else ""
    return {
        CONFIRMED: "code ok",
        CONTRADICTED: "WRONG CODE" if verdict_.label_id else f"NO MATCH{whose}",
        UNREADABLE: f"no code{whose}",
        PRESENT: "code unchecked",
    }.get(verdict_.state, "")


# --- what this actually protects -------------------------------------------
#
# The same rule the novelty profile follows: a check that covers some labels
# and not others has to say which, at a moment somebody can act on it. "Code
# verification is on" is not a useful sentence when half the library declares
# no code.

FULL = "verified"        # demanded, and a pattern to check the read against
WEAK = "presence only"   # demanded, but no pattern -- any code passes
NONE = "unchecked"       # nothing demanded


def coverage(label) -> str:
    """How well one label is protected by code reading."""
    wanted = demanded(getattr(label, "codes", None))
    if not wanted:
        return NONE
    return FULL if any(getattr(s, "pattern", "") for s in wanted) else WEAK


def patterns_for(labels) -> dict[str, list[str]]:
    """``{label_id: [pattern, ...]}`` for every label that has one."""
    out: dict[str, list[str]] = {}
    for label in labels or []:
        found = [str(getattr(s, "pattern", "") or "")
                 for s in demanded(getattr(label, "codes", None))
                 if getattr(s, "pattern", "")]
        if found:
            out[str(getattr(label, "label_id", ""))] = found
    return out


def readiness(labels) -> str:
    """Which labels code reading can actually vouch for, and which it cannot."""
    labels = list(labels or [])
    if not labels:
        return "No labels in the library, so there is nothing to verify against."
    buckets: dict[str, list[str]] = {FULL: [], WEAK: [], NONE: []}
    for label in labels:
        buckets[coverage(label)].append(str(getattr(label, "label_id", "")))

    lines = [f"{len(buckets[FULL])} of {len(labels)} label(s) can be verified "
             f"from their printing."]
    if buckets[FULL]:
        lines.append(f"  Verified: {', '.join(sorted(buckets[FULL]))}")
    if buckets[WEAK]:
        lines.append(f"  Presence only: {', '.join(sorted(buckets[WEAK]))}")
        lines.append("   These demand a code but give no pattern, so any code "
                     "that decodes passes. Add the pattern its part number "
                     "matches and they become verifiable.")
    if buckets[NONE]:
        lines.append(f"  Unchecked: {', '.join(sorted(buckets[NONE]))}")
        lines.append("   These declare no code for inspection, so a detection "
                     "called one of them rests on the classifier alone -- which "
                     "is what lets an unenrolled label through. Draw the code "
                     "region on the artwork and set its policy.")
    if not buckets[FULL]:
        lines.append("")
        lines.append("Nothing is verifiable yet, so this check cannot stop an "
                     "unenrolled label. It is doing nothing until at least one "
                     "label has a code region and a pattern.")
    return "\n".join(lines)
