"""Decode a saved crop and say exactly what happened, rung by rung.

    python scripts/decode_check.py data/code_reads/*.png

The Test Read dialog reports which attempt succeeded, but only for the part
currently under the camera. This runs the same ladder against a PNG that was
saved earlier, which makes a barcode that would not read reproducible: the same
pixels, as many times as it takes, while a setting or a lens is changed.

It reports every rung rather than stopping at the first success, because "read,
but only on the last attempt" and "read immediately" are different answers about
whether the setup is sound.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    import cv2

    from label_detections.core import code_reader as cr

    ok, reason = cr.available()
    if not ok:
        print(f"No decoder: {reason}")
        return 1
    module, _ = cr.backend()

    status = 1
    for name in argv:
        path = Path(name)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"{path}: could not be read as an image")
            continue

        height, width = image.shape[:2]
        print(f"\n{path.name}  {width}x{height} px")
        print("-" * (len(path.name) + 20))
        found = False
        for how, candidate, options in cr._ladder(module, image):
            reads = cr._read_once(module, candidate, **options)
            shape = getattr(candidate, "shape", ())
            size = f"{shape[1]}x{shape[0]}" if len(shape) >= 2 else "?"
            if reads:
                found = True
                for read in reads:
                    spellings = read.candidates()
                    extra = (f"   (also matches {spellings[1]})"
                             if len(spellings) > 1 else "")
                    print(f"  {how:<32} {size:>10}  {read.text} "
                          f"[{read.symbology}]{extra}")
            else:
                print(f"  {how:<32} {size:>10}  --")
        if found:
            status = 0
        else:
            print("\n  Nothing read on any attempt. If the bars look clean at "
                  "full size, measure them: a UPC-A needs about 190 px across "
                  "the symbol, and more once a warp has softened it.")
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
