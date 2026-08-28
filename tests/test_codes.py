"""What a decoded code says about which label it is.

The line cannot pass a wrong id, and every other check in this pipeline is a
guess about appearance that can be confidently wrong about a label nobody has
ever enrolled. A decoded code is the part number printed on the part, so it can
refuse a label it has never seen -- which is the whole reason for building it
over another model.

That makes the failure modes matter more than the happy path here. A verifier
that quietly passes everything is worse than no verifier, because the readout
says a check is running.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detections.core import codes as cd
from label_detections.core.live_detect import UNKNOWN


class _Spec:
    def __init__(self, policy="must_decode", pattern="", region=(0.1, 0.1, 0.2, 0.2)):
        self.policy = policy
        self.pattern = pattern
        self.region = list(region)
        self.role = "part_number"
        self.symbology = "qr"


class _Label:
    def __init__(self, label_id, codes=()):
        self.label_id = label_id
        self.codes = list(codes)


def _read(text):
    return [cd.Read(text=text, symbology="QRCode")]


PC680 = _Spec(pattern=r"^ODS-AGM16L")
G31 = _Spec(pattern=r"^NP16-12B")
PATTERNS = {"PC680": [r"^ODS-AGM16L"], "sp_g31": [r"^NP16-12B"]}


# --- the failure this was built for -----------------------------------------

def test_a_code_matching_no_enrolled_label_refuses_the_detection():
    """The Genesys label: never trained on, die cut matches PC680, and the
    classifier says PC680 at 0.99. Its printing says something else entirely,
    and nothing had to have seen it before for that to be decisive."""
    v = cd.verdict("PC680", [PC680], _read("NP16-99Z-UNSEEN"), PATTERNS)
    assert v.state == cd.CONTRADICTED
    assert v.label_id == ""
    assert cd.resolve("PC680", v) == UNKNOWN
    assert "matches no enrolled label" in v.detail


def test_a_code_belonging_to_another_enrolled_label_relabels_it():
    """The printing outranks the classifier. It is not a tiebreak -- one is
    guessing from appearance and the other is printed on the part."""
    v = cd.verdict("PC680", [PC680], _read("NP16-12B-0798"), PATTERNS)
    assert v.state == cd.CONTRADICTED
    assert cd.resolve("PC680", v) == "sp_g31"
    assert "belongs to sp_g31" in v.detail


def test_the_right_label_is_confirmed_rather_than_merely_not_refused():
    v = cd.verdict("PC680", [PC680], _read("ODS-AGM16L-16AH"), PATTERNS)
    assert v.state == cd.CONFIRMED and v.verified
    assert cd.resolve("PC680", v) == "PC680"


def test_a_demanded_code_that_does_not_decode_does_not_pass():
    """"Could not verify" is not "verified". The library said this label
    carries a code that inspection must read; not reading it is a failure to
    establish the identity, not a reason to take the classifier's word."""
    v = cd.verdict("PC680", [PC680], [], PATTERNS)
    assert v.state == cd.UNREADABLE and v.blocks
    assert cd.resolve("PC680", v) == UNKNOWN


# --- refusing to claim more than it knows -----------------------------------

def test_a_label_with_no_code_marked_for_inspection_is_left_alone():
    """Not every label carries a code. Reporting those as failures would make
    the check unusable on a mixed library, and the classifier is what stands
    behind them -- which the readiness report says out loud."""
    v = cd.verdict("PC680", [_Spec(policy="ignore")], [], PATTERNS)
    assert v.state == cd.NOT_CHECKED
    assert cd.resolve("PC680", v) == "PC680"
    assert not v.blocks


def test_a_code_with_no_pattern_reports_presence_not_confirmation():
    """Any code that decodes would satisfy it, so calling that CONFIRMED would
    be the strongest state on the weakest evidence there is."""
    v = cd.verdict("PC680", [_Spec(pattern="")], _read("anything at all"),
                   {})
    assert v.state == cd.PRESENT
    assert not v.verified
    assert cd.resolve("PC680", v) == "PC680"
    assert "no pattern to check it against" in v.detail


def test_an_empty_pattern_matches_nothing_rather_than_everything():
    """The direction of this default decides whether a mis-entered label
    silently passes every code on earth."""
    assert cd.matches("", "ODS-AGM16L") is False
    assert cd.matches(r"^ODS", "") is False


def test_a_pattern_that_does_not_compile_fails_shut():
    """A typo in a regex is a data-entry mistake. Matching everything would
    turn one typo into a label that accepts any code silently."""
    assert cd.matches(r"^ODS-[AGM", "ODS-AGM16L") is False


def test_two_labels_claiming_one_code_is_confirmed_but_named():
    """A library problem rather than a line problem, and picking either
    silently would be arbitrary."""
    patterns = {"PC680": [r"^ODS"], "other": [r"^ODS"]}
    v = cd.verdict("PC680", [PC680], _read("ODS-AGM16L"), patterns)
    assert v.state == cd.CONFIRMED
    assert "two labels claim this printing" in v.detail


def test_a_blank_read_is_not_a_read():
    v = cd.verdict("PC680", [PC680], [cd.Read(text="")], PATTERNS)
    assert v.state == cd.UNREADABLE


# --- what the operator sees -------------------------------------------------

def test_every_state_says_something_on_the_plate_except_the_ones_that_should_not():
    """A refusal with nothing on it cannot be told from a bug, and a confirmed
    read is the thing a quality gate exists to show."""
    def note(state, label_id=""):
        return cd.plate_note(cd.Verdict(state, label_id))

    assert note(cd.CONFIRMED) == "code ok"
    assert note(cd.CONTRADICTED, "sp_g31") == "WRONG CODE"
    assert note(cd.CONTRADICTED) == "NO MATCH"
    assert note(cd.UNREADABLE) == "no code"
    assert note(cd.PRESENT) == "code unchecked"
    # Nothing was asked of this label, so nothing is claimed about it.
    assert note(cd.NOT_CHECKED) == ""


# --- what it actually protects ----------------------------------------------

def test_readiness_names_the_labels_resting_on_the_classifier_alone():
    """"Code verification is on" is not useful when half the library declares
    no code, and that half is exactly where an unenrolled label still gets in."""
    text = cd.readiness([
        _Label("PC680", [PC680]),
        _Label("weak", [_Spec(pattern="")]),
        _Label("bare"),
    ])
    assert "1 of 3 label(s) can be verified" in text
    assert "Verified: PC680" in text
    assert "Presence only: weak" in text
    assert "Unchecked: bare" in text
    assert "rests on the classifier alone" in text


def test_readiness_is_blunt_when_it_can_protect_nothing():
    """The state where the checkbox is ticked and the check is decorative."""
    text = cd.readiness([_Label("bare"), _Label("also_bare")])
    assert "Nothing is verifiable yet" in text
    assert "cannot stop an unenrolled label" in text


def test_readiness_on_an_empty_library_does_not_claim_coverage():
    assert "nothing to verify against" in cd.readiness([])


def test_coverage_grades_one_label():
    assert cd.coverage(_Label("a", [PC680])) == cd.FULL
    assert cd.coverage(_Label("a", [_Spec(pattern="")])) == cd.WEAK
    assert cd.coverage(_Label("a", [_Spec(policy="ignore", pattern="x")])) == cd.NONE
    assert cd.coverage(_Label("a")) == cd.NONE


def test_patterns_are_collected_only_from_codes_that_demand_something():
    """An ignored code's pattern must not be used to identify anything: the
    library says nobody checks it, so it may well be stale."""
    labels = [_Label("a", [PC680]), _Label("b", [_Spec(policy="ignore", pattern="^X")])]
    assert cd.patterns_for(labels) == {"a": [r"^ODS-AGM16L"]}


# --- getting the pixels -----------------------------------------------------

def test_the_code_region_is_taken_from_the_full_resolution_frame():
    """The one crop in this pipeline that must not be shrunk. Stage 2's crop is
    320 px of a whole label, so a barcode occupying a tenth of it arrives as
    32 px -- below what any decoder can resolve. Reusing that crop would make
    this feature look broken rather than be it."""
    import inspect

    from label_detections.core import code_reader

    source = inspect.getsource(code_reader.read_label)
    assert "rectify_quad(frame, placed)" in source, (
        "the region crop must come off the frame with no size cap")


def test_the_region_is_grown_so_the_quiet_zone_is_included():
    """A region drawn tight to the printed bars has no quiet zone, and
    decoders need one."""
    from label_detections.core import code_reader

    x, y, w, h = code_reader._expand([0.4, 0.4, 0.2, 0.1], 0.25)
    assert w > 0.2 and h > 0.1
    assert x < 0.4 and y < 0.4


def test_growing_a_region_at_the_edge_stays_inside_the_label():
    """place_unit_rect maps fractions of the label; over 1.0 would walk the
    crop off the artwork."""
    from label_detections.core import code_reader

    x, y, w, h = code_reader._expand([0.0, 0.9, 1.0, 0.1], 0.5)
    assert x >= 0.0 and y >= 0.0
    assert x + w <= 1.001 and y + h <= 1.001


def test_no_decoder_installed_reads_as_nothing_decoded_not_a_crash(monkeypatch):
    """The dependency is optional at runtime. A missing one has to leave the
    rest of live detect working, and the load message names it."""
    from label_detections.core import code_reader

    monkeypatch.setattr(code_reader, "_BACKEND", None)
    monkeypatch.setattr(code_reader, "_REASON", "not installed")
    assert code_reader.decode(object()) == []
    assert code_reader.read_label(object(), [[0, 0], [1, 0], [1, 1], [0, 1]],
                                  [PC680]) == []
    ok, why = code_reader.available()
    assert ok is False and why


def test_a_snapshot_is_taken_rather_than_the_live_library(monkeypatch):
    """The worker runs off the GUI thread and the library is edited on it."""
    from label_detections.core import code_reader

    label = _Label("PC680", [PC680])

    class _Lib:
        def all(self):
            return [label]

    specs, patterns = code_reader.library_snapshot(_Lib())
    assert patterns == {"PC680": [r"^ODS-AGM16L"]}
    # Editing the library afterwards must not change what the worker holds.
    label.codes[0].pattern = "^CHANGED"
    assert specs["PC680"][0].pattern == r"^ODS-AGM16L"


# --- the wiring, where a bookkeeping slip becomes a wrong id ----------------

class _Boxes:
    def __init__(self, xyxy, ids=None):
        self.xyxy = xyxy
        self.conf = [0.9] * len(xyxy)
        self.id = ids


class _Res:
    def __init__(self, xyxy, ids=None):
        self.obb = None
        self.boxes = _Boxes(xyxy, ids)


def _worker(**kwargs):
    from label_detections.ui.live_detect import InferenceWorker

    worker = InferenceWorker("det.pt", 640, 0.5, None, classifier_path="cls.pt",
                             read_codes=True, code_specs={"PC680": [PC680]},
                             code_patterns=PATTERNS, **kwargs)
    worker._codes_on = True
    return worker


def test_track_ids_come_out_in_the_same_order_as_the_quads():
    """These two lists are zipped. Any divergence caches one label's verdict
    against another label's track -- the wrong-id failure this feature exists
    to prevent, reintroduced as a bookkeeping mistake."""
    worker = _worker()
    results = [_Res([[0, 0, 10, 10], [20, 20, 30, 30]], ids=[7, 9]),
               _Res([[40, 40, 50, 50]], ids=[11])]
    assert len(worker._detection_track_ids(results)) == len(
        worker._detection_quads(results)) == 3
    assert worker._detection_track_ids(results) == [7, 9, 11]


def test_an_untracked_run_yields_no_ids_rather_than_a_short_list():
    """A plain predict has boxes.id None, and a short list would slide every
    later detection onto the wrong verdict."""
    worker = _worker()
    results = [_Res([[0, 0, 10, 10], [20, 20, 30, 30]])]
    assert worker._detection_track_ids(results) == [None, None]


def test_a_verdict_is_cached_per_track_rather_than_decoded_every_frame(monkeypatch):
    """A part sits in front of the camera for a whole takt and its printing
    does not change. Decoding on all ~340 of those frames is 339 warps for an
    answer already in hand."""
    from label_detections.core import code_reader

    calls = []
    monkeypatch.setattr(code_reader, "read_label",
                        lambda *a, **k: calls.append(1) or _read("ODS-AGM16L"))
    worker = _worker()
    results = [_Res([[0, 0, 10, 10]], ids=[7])]
    for _ in range(5):
        out = worker._verify_codes(object(), results, [("PC680", 0.99, "")])
    assert len(calls) == 1, "decoded the same track more than once"
    assert out[0][0] == "PC680" and "code ok" in out[0][2]


def test_an_unreadable_frame_is_not_cached(monkeypatch):
    """Usually a pose or a blur. Caching it would hold a failure over a part
    whose very next frame reads perfectly."""
    from label_detections.core import code_reader

    monkeypatch.setattr(code_reader, "read_label", lambda *a, **k: [])
    worker = _worker()
    results = [_Res([[0, 0, 10, 10]], ids=[7])]
    out = worker._verify_codes(object(), results, [("PC680", 0.99, "")])
    assert out[0][0] == UNKNOWN and "no code" in out[0][2]
    assert worker._code_cache == {}


def test_the_worker_relabels_from_the_code_not_the_classifier(monkeypatch):
    from label_detections.core import code_reader

    monkeypatch.setattr(code_reader, "read_label",
                        lambda *a, **k: _read("NP16-12B-0798"))
    worker = _worker()
    results = [_Res([[0, 0, 10, 10]], ids=[7])]
    out = worker._verify_codes(object(), results, [("PC680", 0.99, "")])
    assert out[0][0] == "sp_g31"
    assert "WRONG CODE" in out[0][2]


def test_the_novelty_note_and_the_code_note_both_survive(monkeypatch):
    """Two checks ran and the plate has room to say what each concluded."""
    from label_detections.core import code_reader

    monkeypatch.setattr(code_reader, "read_label",
                        lambda *a, **k: _read("ODS-AGM16L"))
    worker = _worker()
    results = [_Res([[0, 0, 10, 10]], ids=[7])]
    out = worker._verify_codes(object(), results, [("PC680", 0.99, "nov 0.31x")])
    assert out[0][2] == "nov 0.31x code ok"


def test_a_label_with_nothing_demanded_is_not_decoded_at_all(monkeypatch):
    """Warping and decoding for a label the library asks nothing of is pure
    cost on a frame that has to keep up with the camera."""
    from label_detections.core import code_reader

    calls = []
    monkeypatch.setattr(code_reader, "read_label",
                        lambda *a, **k: calls.append(1) or [])
    worker = _worker()
    worker._code_specs = {"PC680": [_Spec(policy="ignore")]}
    results = [_Res([[0, 0, 10, 10]], ids=[7])]
    out = worker._verify_codes(object(), results, [("PC680", 0.99, "")])
    assert not calls
    assert out[0] == ("PC680", 0.99, "", "PC680")


def test_an_already_unknown_detection_is_not_decoded(monkeypatch):
    """There is no label to look up a code region on."""
    from label_detections.core import code_reader

    calls = []
    monkeypatch.setattr(code_reader, "read_label",
                        lambda *a, **k: calls.append(1) or [])
    worker = _worker()
    results = [_Res([[0, 0, 10, 10]], ids=[7])]
    worker._verify_codes(object(), results, [(UNKNOWN, 0.2, "")])
    assert not calls


def test_the_check_turns_itself_off_when_no_label_can_be_verified(monkeypatch):
    """Ticked and decorative is the worst state: the readout would say a check
    is running while nothing could ever fail."""
    from label_detections.core import code_reader

    monkeypatch.setattr(code_reader, "available", lambda: (True, ""))
    worker = _worker()
    worker._code_patterns = {}
    worker._load_codes()
    assert worker._codes_on is False
    assert "no label has both" in worker._code_note


def test_a_missing_decoder_is_reported_before_anything_else(monkeypatch):
    """It is the more fundamental problem, and a note about patterns would
    send somebody to edit the library when the fix is a pip install.

    Forced rather than inferred from the environment: zxing-cpp is a real
    dependency and this has to keep testing the missing case on a machine that
    has it installed.
    """
    from label_detections.core import code_reader

    monkeypatch.setattr(code_reader, "available",
                        lambda: (False, "zxing-cpp is not installed. pip install zxing-cpp"))
    worker = _worker()
    worker._load_codes()
    assert worker._codes_on is False
    assert "zxing" in worker._code_note and "pip install" in worker._code_note


def test_the_load_message_names_the_labels_it_cannot_verify(monkeypatch):
    """On reads as 'all of them' unless it says otherwise."""
    from label_detections.core import code_reader

    monkeypatch.setattr(code_reader, "available", lambda: (True, ""))
    worker = _worker()
    worker._code_specs = {"PC680": [PC680], "bare": [_Spec(policy="ignore")]}
    worker._code_patterns = {"PC680": [r"^ODS-AGM16L"]}
    worker._load_codes()
    assert worker._codes_on is True
    assert "1 label(s) verifiable" in worker._code_note
    assert "NOT verifiable: bare" in worker._code_note


# --- the print spec: what it does and does not do ---------------------------
#
# Width, module size and quiet zone are never handed to a decoder. Two of them
# feed one advisory number -- how many pixels the camera needs -- and the third
# was stored and read by nothing at all until it was given this job.

def test_the_crop_margin_falls_back_to_a_guess_with_no_print_spec():
    """Most labels will never have these numbers typed in, and the check has
    to work anyway."""
    from label_detections.core import code_reader as cr

    assert cr.region_margin(cr.Spec()) == cr.REGION_MARGIN


def test_a_declared_quiet_zone_widens_the_crop_when_the_code_is_small():
    """The failure this prevents: an 8 mm DataMatrix with a 2.5 mm quiet zone
    needs more than half its own width again either side. A fixed guess crops
    inside the quiet zone, the decode fails, and a good part reads 'no code'
    with the readout blaming the printing."""
    from label_detections.core import code_reader as cr

    small = cr.Spec(quiet_zone_mm=2.5, code_width_mm=8.0)
    assert cr.region_margin(small) == pytest.approx(0.625)


def test_a_spec_asking_for_less_than_the_guess_does_not_tighten_the_crop():
    """Every label that works today has to keep working. The spec may only
    widen the crop, never narrow it."""
    from label_detections.core import code_reader as cr

    roomy = cr.Spec(quiet_zone_mm=1.0, code_width_mm=46.0)   # 0.043 on its own
    assert cr.region_margin(roomy) == cr.REGION_MARGIN


def test_a_quiet_zone_typed_in_the_wrong_units_still_leaves_a_crop():
    """Millimetres and thousandths of an inch are both plausible things to type
    into a box labelled mm, and one of them is 25x the other."""
    from label_detections.core import code_reader as cr

    assert cr.region_margin(cr.Spec(quiet_zone_mm=250.0, code_width_mm=8.0)) \
        == cr.MAX_REGION_MARGIN


# --- UPC-A, which is what the batteries actually carry ----------------------

UPC = "635241140996"
UPC_PATTERNS = {"PC680": [f"^{UPC}$"]}
UPC_SPEC = [_Spec(pattern=f"^{UPC}$")]


def test_a_upc_read_as_ean13_still_matches_the_twelve_digit_pattern():
    """The trap. A UPC-A is 12 digits and the label prints 12 digits, so that
    is what anyone writes a pattern against -- but the two symbologies share an
    encoding and decoders commonly return the EAN-13 form with a leading zero.
    Every genuine part would read NO MATCH, which is the expensive direction."""
    v = cd.verdict("PC680", UPC_SPEC, [cd.Read("0" + UPC, "EAN13")], UPC_PATTERNS)
    assert v.state == cd.CONFIRMED


def test_a_pattern_written_for_the_ean13_form_matches_a_upca_read():
    """The same trap from the other side, for anyone who copied the 13-digit
    form out of a decoder's output."""
    patterns = {"PC680": [f"^0{UPC}$"]}
    v = cd.verdict("PC680", [_Spec(pattern=f"^0{UPC}$")],
                   [cd.Read(UPC, "UPCA")], patterns)
    assert v.state == cd.CONFIRMED


def test_the_leading_zero_rule_does_not_make_a_different_code_match():
    """Admitting the same code spelled two ways is not a loosening. A code that
    is genuinely different still has to fail."""
    v = cd.verdict("PC680", UPC_SPEC, [cd.Read("635241140997", "UPCA")],
                   UPC_PATTERNS)
    assert v.state == cd.CONTRADICTED and v.label_id == ""


def test_only_all_digit_reads_get_the_leading_zero_treatment():
    """A 13-character alphanumeric that happens to start with 0 is not a UPC."""
    assert cd.Read("0ABC123456789").candidates() == ["0ABC123456789"]
    assert cd.Read("").candidates() == [""]


# --- saying why a code did not read -----------------------------------------
#
# "no code" is the end of a chain with several places to fail and no way to
# tell them apart from outside: the region lands wrong, the crop is too few
# pixels, the print is out of focus, the decoder is absent. The diagnostic
# exists to split that chain, and the split that matters is region-vs-whole.

def _report(region_reads=(), whole_reads=(), region_px=(300, 90), ok=True):
    import numpy as np

    from label_detections.core import code_reader as cr

    crop = np.zeros((region_px[1], region_px[0], 3), dtype=np.uint8)
    return {
        "ok": ok, "reason": "" if ok else "zxing-cpp is not installed",
        "regions": [{"spec": _Spec(), "crop": crop,
                     "reads": list(region_reads), "note": ""}],
        "whole": {"crop": np.zeros((200, 900, 3), dtype=np.uint8),
                  "reads": list(whole_reads)},
    }, cr


def test_a_region_that_reads_hands_back_the_text_to_compare():
    """The next failure after a good decode is a pattern that does not match
    it, and that is settled by looking at the two strings side by side."""
    report, cr = _report(region_reads=[cd.Read("635241140996", "UPCA")])
    text = cr.diagnosis_text(report)
    assert "635241140996" in text
    assert "character for character" in text


def test_a_whole_label_read_with_an_empty_region_blames_the_placement():
    """The decisive fork. The code is legible in this very frame, so nothing is
    wrong with the picture -- the crop simply did not contain it."""
    report, cr = _report(whole_reads=[cd.Read("635241140996", "UPCA")])
    text = cr.diagnosis_text(report)
    assert "LANDING IN THE WRONG PLACE" in text
    assert "DETECTOR" in text, "must name why the region can move at runtime"


def test_both_empty_blames_the_picture_rather_than_the_placement():
    """No placement change fixes a code the optics never resolved, so the reply
    has to point at the camera rather than at the region."""
    report, cr = _report()
    text = cr.diagnosis_text(report)
    assert "NOT DECODING ANYWHERE" in text
    assert "fill more of the frame" in text
    assert "LANDING IN THE WRONG PLACE" not in text


def test_the_pixels_per_module_is_reported_for_a_fixed_width_symbology():
    """The one number that says whether the crop could ever have worked, and
    the thing that turns "it is intermittent" into "it is too far away"."""
    assert "MARGINAL" in _resolution_line(398)


def _resolution_line(crop_width: int) -> str:
    """Just the measured verdict, not the surrounding advice -- which mentions
    MARGINAL too, and would make this pass whatever the number said."""
    report, cr = _report(region_px=(crop_width, 211))
    report["regions"][0]["spec"].symbology = "upca"
    report["regions"][0]["spec"].region = [0.842, 0.239, 0.151, 0.197]
    return next(line for line in cr.diagnosis_text(report).split("\n")
                if "px per module" in line)


def test_a_roomy_crop_is_not_called_marginal():
    line = _resolution_line(1200)
    assert "MARGINAL" not in line and "comfortable" in line


def test_a_variable_width_symbology_gets_no_invented_number():
    """Code 128 varies with how much data it carries, so there is no honest
    pixels-per-module without decoding it first."""
    report, cr = _report(region_px=(398, 211))
    report["regions"][0]["spec"].symbology = "code128"
    report["regions"][0]["spec"].region = [0.842, 0.239, 0.151, 0.197]
    assert "px per module" not in cr.diagnosis_text(report)


def test_the_crop_size_is_always_reported():
    """The one number that decides whether it could ever have worked."""
    report, cr = _report(region_px=(140, 40))
    assert "140x40 px" in cr.diagnosis_text(report)


def test_a_missing_decoder_is_said_first_and_stops_there():
    """Everything below it would read as a picture problem otherwise, and send
    somebody to re-shoot a label when the fix is a pip install."""
    report, cr = _report(ok=False)
    text = cr.diagnosis_text(report)
    assert text.startswith("No decoder")
    assert "picture problem" not in text


def test_diagnose_is_safe_with_nothing_to_work_on():
    from label_detections.core import code_reader as cr

    report = cr.diagnose(None, None, [])
    assert report["regions"] == [] and report["whole"] is None


# --- naming the label a refusal was about -----------------------------------
#
# A code refusal rewrites the name to "unknown", which strips the box of the
# one clue anybody investigating it needs: which label's code was expected.

def test_a_refusal_names_the_label_whose_code_was_wanted():
    """"unknown ... no code" says a code is missing without saying whose."""
    assert cd.plate_note(cd.Verdict(cd.UNREADABLE), "PC680") == "no code (PC680)"
    assert cd.plate_note(cd.Verdict(cd.CONTRADICTED), "PC680") == "NO MATCH (PC680)"


def test_a_relabel_does_not_repeat_the_name_it_moved_to():
    """The box already reads as the label the code claimed."""
    assert cd.plate_note(cd.Verdict(cd.CONTRADICTED, "sp_g31"), "PC680") \
        == "WRONG CODE"


def test_a_confirmed_read_stays_short():
    assert cd.plate_note(cd.Verdict(cd.CONFIRMED), "PC680") == "code ok"


def test_the_proposal_survives_onto_the_item():
    """Once the name is rewritten to unknown, this is the only place stage 2's
    answer still exists -- and it is what a code region is looked up by."""
    from label_detections.core import live_detect as ld

    items = ld.apply_identities(
        [{"name": "label", "conf": 0.97}],
        [("unknown", 0.99, "no code (PC680)", "PC680")])
    assert items[0]["proposed"] == "PC680"
    assert items[0]["name"] == "unknown"


def test_an_identity_with_no_proposal_falls_back_to_its_own_name():
    from label_detections.core import live_detect as ld

    items = ld.apply_identities([{"name": "label", "conf": 0.9}],
                                [("PC680", 0.99)])
    assert items[0]["proposed"] == "PC680"


# --- testing the thing on screen, not the thing in a dropdown ---------------
#
# An operator holding a PC680 under the camera, asking why its box said "no
# code", was told that '2220-9199' declares no code for inspection -- an answer
# about a label nobody was presenting. The Class dropdown and the detection in
# front of the lens are different things, and the interesting case is exactly
# when they differ.

@pytest.fixture
def win():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from label_detections.ui.main_window import MainWindow
    return MainWindow()


def _quad():
    return [[10.0, 10.0], [110.0, 10.0], [110.0, 70.0], [10.0, 70.0]]


def test_a_live_detection_is_preferred_over_the_open_label(win):
    import numpy as np

    win.label_id = "2220-9199"
    win._live_result_frame = np.zeros((200, 400, 3), dtype=np.uint8)
    win._live_result_items = [
        {"name": "unknown", "proposed": "PC680", "points": _quad()}]
    source, label_id, frame, quad = win._code_test_subject()
    assert label_id == "PC680", "answered about the dropdown, not the camera"
    assert "live detection" in source
    assert quad == _quad() and frame is not None


def test_the_identity_comes_from_before_the_code_refused_it(win):
    """A refusal rewrites the name to unknown, so `name` alone would leave
    nothing to look a code region up by -- which is the state every box in the
    failing screenshot was in."""
    import numpy as np

    win.label_id = ""
    win._live_result_frame = np.zeros((200, 400, 3), dtype=np.uint8)
    win._live_result_items = [
        {"name": "unknown", "proposed": "PC680", "points": _quad()}]
    assert win._code_test_subject()[1] == "PC680"


def test_detections_with_no_identity_at_all_say_so_rather_than_guessing(win, monkeypatch):
    import numpy as np

    import label_detections.ui.main_window as mw

    said = []
    monkeypatch.setattr(mw.QMessageBox, "information",
                        staticmethod(lambda *a, **k: said.append(a[2])))
    win._live_result_frame = np.zeros((200, 400, 3), dtype=np.uint8)
    win._live_result_items = [{"name": "unknown", "points": _quad()}]
    assert win._code_test_subject()[1] is None
    assert "none of them carries a label identity" in said[0].replace("\n", " ")


def test_with_no_live_result_it_falls_back_to_the_drawn_box(win, monkeypatch):
    """Testing a label without the camera running is still worth doing."""
    import numpy as np

    import label_detections.ui.main_window as mw

    monkeypatch.setattr(mw.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    win._live_result_items = []
    win._live_result_frame = None
    win.label_id = "not_a_real_label"
    assert win._code_test_subject()[1] is None      # no such label, no crash


# --- is it the camera or the framing ----------------------------------------
#
# "Get closer" is not an instruction anybody can act on, and "you need a better
# camera" is usually wrong: a label in the middle of a wide field is spending
# most of its pixels on the table around it. The share of the frame the label
# fills is measurable with a test shot, and it is the thing that has to change.

def test_the_framing_is_blamed_before_the_sensor():
    """The real case: 20 MP, and the label using 38% of the width of it."""
    note = cd.framing_note("upca", 0.151, label_px=2109, frame_px=5496)
    assert "fills 38%" in note
    assert "needs to fill 46%" in note and "1.19x tighter" in note
    assert "sensor is not the limit" in note


def test_adequate_framing_is_confirmed_rather_than_nagged_at():
    note = cd.framing_note("upca", 0.151, label_px=4400, frame_px=5496)
    assert "enough for this code" in note
    assert "tighter" not in note


def test_a_code_too_small_for_the_sensor_says_so_honestly():
    """Sometimes it really is the camera, and saying "frame it tighter" then
    sends somebody to move a mount that cannot help."""
    note = cd.framing_note("upca", 0.03, label_px=2109, frame_px=5496)
    assert "even filling it completely" in note
    assert "more pixels across" in note


def test_no_framing_advice_without_the_numbers_to_base_it_on():
    assert cd.framing_note("upca", 0.151, 0, 5496) == ""
    assert cd.framing_note("code128", 0.151, 2109, 5496) == ""


def test_the_pixels_per_module_thresholds_are_the_measured_ones():
    """These came from running real symbols through the real pipeline, not
    from a datasheet, and the note's credibility rests on them."""
    assert cd.MIN_PX_PER_MODULE == 4.0
    assert cd.MODULES["upca"] == 95 and cd.MODULES["ean13"] == 95
