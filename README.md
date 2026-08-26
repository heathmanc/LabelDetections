# LabelVision Studio

Battery side-label inspection: a label library, per-label training datasets,
and the recipes the vision program runs.

Built alongside [BungVision Label Studio](https://github.com/heathmanc/bunglabel),
which inspects bungs on battery lids. The data-root resolution, the review-marker
discipline and the reviewed-only export rule are ported from it. Everything
about the *data model* is different, for the reasons below.

---

## 1. The three pieces, and why they are separate

| Piece | Answers | Changes when |
|---|---|---|
| **Label library** | What does this label look like? What does it carry? | A new label SKU or artwork revision |
| **Per-label dataset** | What does this label look like in real life? | You gather and label more images of it |
| **Recipe** | Which labels must be on this battery, and where? | A product's bill of labels changes |

A label is trained **one at a time**, against its own dataset, on its own
schedule. A recipe only assembles labels that already exist, so a new recipe
costs minutes and needs no model work at all. Adding a label SKU is a library
row plus a dataset — never a retrain of everything else.

The thing this arrangement is designed to avoid: making every individual label
artwork its own detector class. That is the obvious first move and it is the
one that kills these projects, because every new SKU and every artwork revision
then means re-labeling and retraining the whole model.

### Two-stage identity

`label` / `class_id` on a box is the coarse **detector family** the model was
trained to find (`spec_plate`, `warning_label`, `cert_mark`, `trace_tag`, …).
`label_id` is the **library identity** — which exact label this is, resolved
after detection by decoding its code and matching its reference artwork.

They are separate fields that never learn about each other. Conflating them is
what forces the retrain.

---

## 2. Recipes and ROIs

A recipe is one view per camera. Each view carries a bill of labels, and each
line of the bill has an **ROI**: a normalised `[x, y, w, h]` rectangle in that
camera's frame, every value 0..1.

Normalised rather than pixels on purpose — a camera swap or a resolution change
re-scales every ROI for free, where a pixel rect silently starts pointing at the
wrong part of the battery.

The ROI does two jobs at once:

* **scopes the search** — the runtime looks for the spec plate only where the
  spec plate belongs;
* **locates the result** — a label found in the wrong place is a placement
  failure, not a pass.

```json
{
  "group": "AGM", "model": "31-AGM-950", "revision": "C", "constrained": true,
  "views": [
    { "view": "side_a", "camera": "cam1", "frame_size": [2592, 1944],
      "unexpected_severity": "warn",
      "labels": [
        { "label_id": "spec_plate_31agm", "roi": [0.05, 0.10, 0.30, 0.40],
          "count": 1, "severity": "fail", "roi_tol": 0.02 },
        { "label_id": "warning_en", "roi": [0.50, 0.10, 0.20, 0.30],
          "count": 1, "severity": "fail", "roi_tol": 0.02 }
      ],
      "forbidden": ["spec_plate_27agm"] },
    { "view": "side_b", "camera": "cam2", "frame_size": [2592, 1944],
      "labels": [
        { "label_id": "trace_tag", "roi": [0.10, 0.10, 0.40, 0.40],
          "count": 1, "severity": "fail" } ] }
  ],
  "cross_checks": [
    { "type": "equal",
      "left":  "side_a.spec_plate_31agm.serial",
      "right": "side_b.trace_tag.serial", "severity": "fail" }
  ]
}
```

Three things in there are worth more than they look.

**The forbidden list.** Most wrong-label escapes are not a *missing* label —
they are the neighbouring model's label, which is present, correct-looking, and
completely wrong. Nothing in the required bill notices it.

**Cross-checks.** With one camera per side, no single image ever sees two labels
that must agree. The check has to live at the battery, which is why it sits
above the views rather than inside one.

**`constrained: false`.** Turns the bill off entirely — free-form, everything
passes. Ported from BungVision's escape hatch, for background captures and
anything that is not an inspection.

---

## 3. Detection can only report what *is* there

"Missing spec plate" is not a detection. It is the absence of one, and the only
thing that can name that absence is the recipe.

So `core/compare.py` loops over **requirements**, not over detections;
detections that no requirement claimed are swept up afterwards as `unexpected`
or `unidentified`. An unidentified detection is never silently dropped — it is
either a new SKU nobody added to the library or a genuine wrong-label defect,
and both need eyes.

Reason codes are stable strings, because they end up in production logs and get
counted in Pareto charts six months later:

```
missing  wrong_count  forbidden  unexpected  unidentified  out_of_roi
rotated  wrong_shape  code_missing  code_unreadable  code_pattern
cross_check  no_frame_size  not_in_library  low_confidence
```

---

## 4. Adding a label: what gets asked, and why

`python -m label_detections.preview labels` prints the whole questionnaire.
The questions that earn their keep:

| Question | Why |
|---|---|
| **Physical size (mm)** | Gives every detection a real-world scale. A box that swallowed two labels has a badly wrong aspect ratio and is caught as a misdetection instead of reported as a defect on the battery. |
| **Variable data + anchor region** | A label carrying a per-unit serial matches against its unchanging artwork only. Without this, every unit looks like a mismatch. |
| **Code region on the artwork** | The runtime crops straight to the barcode from the full-resolution frame instead of searching for it. This is the difference between decoding a 10-mil DataMatrix and not. |
| **X-dimension** | Turns "is the camera sharp enough?" into a number, quoted back in the wizard, before anyone runs a trial. |
| **Surface (matte/gloss/foil)** | Decides whether reference matching works at all, and whether the line needs cross-polarised lighting. |
| **Rotation policy + tolerance** | `fixed`, `flip_ok` or `any`, so an upside-down label is a defect on the labels where that matters and noise on the ones where it does not. |
| **Looks like these labels** | The look-alikes. Their images become hard negatives when this label trains — and they are exactly what belongs on a recipe's forbidden list. |
| **Severity if missing** | Asked once, here, rather than re-litigated in every recipe. |

---

## 5. Repository layout

```text
label_detections/
├── core/                  stdlib only -- no Qt, no OpenCV, no filesystem in the logic
│   ├── geometry.py        OBB maths, homography, reference-artwork placement
│   ├── labels.py          the label library schema
│   ├── recipes.py         bill of labels, ROIs, cross-checks
│   ├── annotations.py     nested sidecar model (boxes -> regions)
│   ├── compare.py         the inspection engine
│   ├── review.py          review markers and the per-image gate
│   ├── dataset.py         group-aware train/val split, coverage reporting
│   ├── storage.py         data-root resolution and per-label paths
│   ├── persistence.py     atomic reads/writes for library, recipes, sidecars
│   ├── wizard.py          declarative questionnaire framework
│   ├── label_wizard.py    the add-a-label question set
│   └── recipe_wizard.py   the build-a-recipe question set
├── ui/
│   ├── flow_dialog.py     generic Qt renderer for any Flow
│   ├── wizards.py         the two entry points
│   └── launcher.py        home window: labels, recipes, both wizards
├── app.py / preview.py
tests/                     159 tests, stdlib + pytest only
```

`core/` has no Qt and no OpenCV anywhere in it. That is not tidiness — it is
what lets the decision that fails a battery be tested exhaustively instead of
observed on a conveyor.

The wizards are **data**. A page is a list of questions; a question has a kind,
a validator and a `visible_when`. Adding a question is a one-line change in
`core/label_wizard.py` or `core/recipe_wizard.py`, and nothing in `ui/` moves.
Conditional questions matter more than they look: asking every question every
time is how operators learn to click Next without reading.

---

## 6. Running it

```bash
pip install -r requirements.txt
python main.py
```

Without PySide6, the questionnaires still print:

```bash
python -m label_detections.preview labels
python -m label_detections.preview recipe
```

Tests need nothing but pytest:

```bash
pip install pytest
python -m pytest tests -q
```

Data lives under `data/`, resolved in this order: `LABELVISION_DATA_DIR`, then
the operator-chosen folder, then beside the app. A configured network share
that is offline falls back to the default rather than making the app
unlaunchable.

```text
data/
├── captures/<label_id>/*.jpg     one dataset per label
├── labels/<label_id>/*.json      annotation sidecars, matched by stem
├── library/labels.json           every label definition
├── recipes/<recipe>.json         the vision program's bills of labels
└── exports/
```

---

## 7. Two rules carried over from BungVision

**Only a review marker this tool wrote counts as reviewed.** Runtime and
third-party JSON is full of generic `reviewed: true` and `review_status: ok`
fields that mean something else entirely. Letting those through is how
unchecked data ends up teaching the model.

**Editing is not approving.** If labels change after an image was approved, the
marker is cleared. A stale approval — reviewed, then edited into a mismatch,
then saved — is the bug that cost BungVision a release.

Force Review is kept, and now requires a `defect_reason`
(`missing_label`, `wrong_label`, `wrong_revision`, `rotated`, `misplaced`,
`torn_or_wrinkled`, `smeared_code`, `unreadable_code`, `duplicate_label`,
`other`). Deliberate defect examples are some of the most valuable training data
there is, and "show me every wrong_revision example" is unanswerable if every
forced image just says "mismatch".

---

## 8. The split bug worth knowing about

`core/dataset.py` splits by **group**, never by image.

Label images arrive in bursts: several frames of the same physical label from
one fixture, a handful off one pallet, a batch from one print run. Those frames
share lighting, wear, print drift and placement. Shuffle images individually —
which is what BungVision's `yolo_export._split_entries` does, and it was correct
for its single-camera case — and siblings land on both sides of the split.
Validation then measures memorisation and reports it as accuracy.

Entries carry `session` and `source`; an entry with neither becomes its own
group, which degrades to the old behaviour rather than silently doing something
surprising. The seed is explicit, so the same images give the same split and two
training runs are actually comparable. Any label missing from validation is
repaired by moving the smallest group that contains it — a model validated
without a single example of a label has said nothing about that label.

---

## 9. Not built yet

- The labeling canvas: draw a label OBB, decode-on-draw, reference-anchored
  sub-regions (`core/annotations.py::apply_reference_regions` is the logic —
  it already places a barcode box from the artwork the moment four corners
  exist).
- Camera capture. `bunglabel/core/camera.py` ports over close to as-is.
- YOLO export, training and evaluation. `core/dataset.py` produces the split
  and the coverage report; the writer is still to come.
- Reference matching and OCR.
- Synthetic sample generation from artwork.
