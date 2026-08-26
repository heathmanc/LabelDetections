# LabelVision Studio

Capture, label and train battery side-label detection — **one label at a time**.

A fork of [BungVision Label Studio](https://github.com/heathmanc/bunglabel).
The application it inherits — camera capture, the OBB canvas, model test,
training, evaluation, the Windows build and installer — carried over close to
intact. What was replaced is the data model, because bungs and labels are not
the same problem.

**There is no recipe authoring here.** Which labels a battery must carry, and
where each one belongs, is the vision front end's business — authored and
stored there. This tool's whole job is producing a trained label.

---

## 1. What changed from bunglabel

| | bunglabel | LabelVision Studio |
|---|---|---|
| A dataset is | one recipe's captures | one **label's** images |
| The review gate asks | does every battery hold N bungs? | does this image carry the label it was collected for? |
| Classes are | battery / bung / retainer | ~7 coarse **families**, stable for years |
| Identity comes from | the class | the **library**, resolved after detection |
| The train/val split | shuffles images | never separates a capture group |
| Recipes | authored here | authored in the front end |

Carried over unchanged: the data-root resolution with its network-share
fallback, the review-marker discipline, reviewed-only export, force review,
background samples, the active-learning queue, bulk relabel, undo/redo, and
every UI constraint in §11–12 of the old README (compact 1920×1080 layout,
cached recipe index, no autosave on mouse-move).

---

## 2. The two-stage idea

Do **not** make each label artwork its own detector class. That is the obvious
first move and it is the one that kills these projects: every new SKU and every
artwork revision then means re-labeling and retraining the whole model.

Instead a box carries two fields that never learn about each other:

```json
{ "label": "spec_plate",              // detector FAMILY -- what the model finds
  "label_id": "spec_plate_31agm",     // library IDENTITY -- which exact label
  "kind": "obb", "points": [[…],[…],[…],[…]],
  "regions": [
    { "role": "code", "code_role": "serial", "symbology": "datamatrix",
      "decoded": "SN0000142771", "decode_ok": true }
  ] }
```

The model learns `spec_plate`. Which spec plate a detection *is* gets resolved
afterwards by decoding its code and matching its artwork against the library.
So adding a label SKU is a library row plus its own dataset — never a retrain
of everything else. Adding a genuinely new *kind* of label is one new family
and one retrain.

The seven shipped families: `battery_side`, `spec_plate`, `warning_label`,
`cert_mark`, `trace_tag`, `promo_label`, `code_patch`.

---

## 3. Working on one label

1. **Label tab → Add Label…** The wizard asks what the label is: size in mm,
   reference images, surface, rotation policy, any barcodes and *where they sit
   on the artwork*, text fields, look-alikes, how many images to gather.
2. Capture or import images into that label's dataset.
3. Draw its oriented box. Drawing under the label's own family stamps the
   identity automatically — the Class combo follows the label you opened,
   because a box drawn under the wrong family is a mislabel that survives all
   the way into training.
4. **Save** approves an image that carries the label. An image that does not is
   saved un-reviewed: editing is not approving.
5. **Mark Background** for negatives — a bare fixture, a battery without this
   label. They teach the model where *not* to fire.
6. **Force Review** for deliberate defect examples. It asks what is wrong
   (`torn_or_wrinkled`, `smeared_code`, `wrong_revision`, …) and records the
   answer, so "do I have enough torn-label examples yet?" is answerable.
7. **Dataset Health** shows every label against its own target.
8. **Export All**, then **Train**.

### Labels train together

They are *gathered* one at a time; they are *trained* together — one detector
over all the families. A model trained on a single class has nothing to tell it
apart from. `Export Dataset` does one label in isolation for a spot check;
`Export All` is the normal path.

---

## 4. Why the add-a-label questions are what they are

`python -m label_detections.preview labels` prints the whole questionnaire.
The ones that earn their keep:

| Question | Why |
|---|---|
| **Read-regions** | The areas inside the label that have to be read on their own. See §5. |
| **Variable data + anchor** | A label carrying a per-unit serial matches against its unchanging artwork only. Without this, every unit looks like a mismatch. |
| **Printed width + X-dimension** | Optional, off the print spec. Turns "is the camera sharp enough?" into a number, quoted back in the wizard, before anyone runs a trial. |
| **Surface** | Gloss and foil glare. Decides whether artwork matching works at all and whether the line needs cross-polarised lighting. |
| **Looks like these labels** | The look-alikes. Their images are the hard negatives that stop the two being swapped. |

Physical size is **optional and unused** — nothing computes with it. Region
placement is proportional, so nothing has to be measured or calibrated for
distance.

The wizard is **data**: pages of questions with kinds, validators and
`visible_when` conditions, in `core/label_wizard.py`. Adding a question is a
one-line change and nothing in `ui/` moves. Conditional questions matter more
than they look — asking every question every time is how operators learn to
click Next without reading.

---

## 5. Read-regions: nesting an area inside a label

A region is an area *inside* a label that inspection reads on its own — a
barcode, a serial, a date code.

**You never need an artwork file, and the Add Label wizard never asks for one.**
A label's dataset is keyed by its id, so no image of it can exist until the
label does — asking for artwork up front would be a circle. It comes from a
capture afterwards:

1. Add the label. Capture images of it (the preview stays live, so a session of
   twenty captures is twenty presses, not twenty reopenings).
2. Open one and draw the label's box, as normal.
3. **Define Regions…** on the Annotation rail (or `Ctrl+Shift+R`). The box you
   drew is warped straight-on — tilt and perspective removed — and saved as
   that label's artwork.
4. Drag regions on it. Name each, pick code / text field / static anchor.

**Capture Reference** on the Live Capture tab does 1–3 in one press: it shoots a
frame, stops the preview, opens it, and waits for you to draw the label's box —
then the region editor opens on it by itself.

The capture the artwork was flattened from is marked **◆ REFERENCE** in the
image list, because redefining regions from a different shot silently moves
every region on the label.

That flattened crop is a *better* reference than vendor artwork anyway: it is
this label, under this line's lighting, through this lens.

Regions are stored as **fractions of the label**, `[x, y, w, h]` each 0–1:

```json
{ "role": "serial", "symbology": "datamatrix", "policy": "must_decode",
  "region": [0.66, 0.117, 0.28, 0.467] }
```

Fractions, not millimetres, because the mapping is pure proportion. On any
other image, drawing the label's four corners is all the positioning they need
— `Ctrl+R` places every region by homography, at any angle, any distance, any
resolution. **Nothing is measured and nothing is calibrated.** Hand-adjusted
regions are never overwritten.

That is also the answer to text that changes per unit. The artwork around a
serial never moves; the serial does. The anchor region says what matching
scores against, and a text region says where to read the part that changes.

### What the reference image is for

Honestly, three things in this order:

1. **It is the surface you draw regions on** — and it comes from a capture, so
   there is nothing to go and find.
2. Human documentation: which artwork this label id refers to.
3. Input to artwork matching, *once that is built*. Nothing reads its pixels at
   runtime today (see §10), so until then identity comes from barcode decoding
   plus the operator's choice.

It is optional on the Add Label page for exactly that reason: a label needs an
id, a name and a family, and nothing else until you have an image of it.

---

## 6. Layout

```text
label_detections/
├── core/                  stdlib only -- no Qt, no OpenCV in the logic
│   ├── labels.py          the label library + read-regions  [new]
│   ├── annotations.py     nested sidecars (boxes -> regions) [new]
│   ├── review.py          markers + the per-image gate      [rewritten]
│   ├── dataset.py         group-aware split, coverage       [new]
│   ├── yolo_export.py     per-label export                  [rewritten]
│   ├── dataset_health.py  readiness tallies                 [rewritten]
│   ├── active_learning.py queue scoring                     [rewritten]
│   ├── storage.py         data root + per-label paths       [ported + new]
│   ├── geometry.py        OBB maths, homography             [superset]
│   ├── persistence.py     atomic library/sidecar IO         [new]
│   ├── wizard.py          questionnaire framework           [new]
│   ├── label_wizard.py    the add-a-label question set      [new]
│   ├── imageio.py         capture writing, image import     [ported]
│   ├── camera.py          Basler/Pylon + V4L2               [ported as-is]
│   ├── training.py, evaluation.py, export_report.py, relabel.py, class_stats.py
│   └──                                                       [ported as-is]
└── ui/
    ├── main_window.py     the app                           [migrated]
    ├── canvas.py          OBB canvas, now identity-aware    [ported + extended]
    ├── region_editor.py   draw read-regions on artwork      [new]
    ├── flow_dialog.py     generic renderer for any Flow     [new]
    └── wizards.py         the add-a-label entry point       [new]
```

`core/` has no Qt and no OpenCV anywhere in it. Not tidiness — it is what lets
the decision that rejects an image be tested exhaustively instead of observed
on a conveyor.

**248 tests.** The core suite needs nothing but pytest; the UI tests build the
real `MainWindow` under the offscreen platform plugin, which is what caught the
lost `@staticmethod` decorators and the canvas dropping `label_id` during this
migration.

---

## 7. The split bug worth knowing about

`core/dataset.py` splits by **group**, never by image.

Label images arrive in bursts: several frames of the same physical label from
one fixture, a batch from one print run. Those frames share lighting, wear and
print drift. Shuffle images individually — which is what bunglabel's
`yolo_export._split_entries` did, correctly, for its single-camera case — and
siblings land on both sides. Validation then measures memorisation and reports
it as accuracy.

Entries carry `session` and `source`; one with neither becomes its own group,
degrading to the old behaviour rather than silently doing something surprising.
The seed is explicit, so two training runs are comparable. Any label missing
from validation is repaired by moving the smallest group containing it. Every
export ships a `split_report.txt` so a reviewer can check all of that before
trusting a number.

---

## 8. Running it

```bash
pip install -r requirements.txt
python main.py
```

Without PySide6 the questionnaire still prints:

```bash
python -m label_detections.preview labels
```

Tests:

```bash
pip install pytest
QT_QPA_PLATFORM=offscreen python -m pytest tests -q
```

Data lives under `data/`, resolved in this order: `LABELVISION_DATA_DIR`, the
operator-chosen folder, then beside the app. A configured network share that is
offline falls back to the default rather than making the app unlaunchable.

```text
data/
├── captures/<label_id>/*.jpg     one dataset per label
├── labels/<label_id>/*.json      sidecars, matched by stem
├── library/labels.json           every label definition
├── library/classes.json          the detector families
└── exports/
```

---

## 9. Two rules inherited, and worth keeping

**Only a review marker this tool wrote counts as reviewed.** Runtime and
third-party JSON is full of generic `reviewed: true` and `review_status: ok`
fields that mean something else. Letting those through is how unchecked data
ends up teaching the model.

**Editing is not approving.** If an image no longer carries the label it was
collected for, saving clears the marker. A stale approval — reviewed, then
edited, then saved — is the bug that cost bunglabel a release, and
`test_ui_label_workflow.py` pins it.

---

## 10. Not built yet

- Decode-on-draw: run the barcode decoder the moment a code region exists, so
  labeling is self-verifying. The regions and their symbologies are already
  there; the decoder call is the missing piece.
- Reference matching and OCR — the identity half of stage two.
- Synthetic sample generation from artwork.
- Auto-identify on draw (the matcher pre-filling `label_id`).
