"""Decode a saved crop and say exactly what happened, rung by rung.

    python scripts/decode_check.py                  # every saved crop
    python scripts/decode_check.py data/code_reads   # a folder
    python scripts/decode_check.py one_crop.png      # or just one

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


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def resolve(argv: list[str]) -> list[Path]:
    """Every image the arguments name.

    Globs are expanded here rather than left to the shell. cmd.exe does not
    expand them, so ``*.png`` arrives as a literal filename and the whole thing
    fails with "could not be read as an image" -- which reads as a broken file
    rather than a shell difference.

    With no arguments at all, the folder Save Crops writes to, since that is
    where the interesting failures already are.
    """
    from label_detections.core.storage import DATA_DIR

    if not argv:
        argv = [str(DATA_DIR / "code_reads")]

    found: list[Path] = []
    for name in argv:
        path = Path(name)
        if path.is_dir():
            found += sorted(p for p in path.iterdir()
                            if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.exists():
            found.append(path)
        else:
            # A glob the shell left alone, relative to wherever it points.
            parent = path.parent if str(path.parent) != "" else Path(".")
            found += sorted(parent.glob(path.name))
    # Same file named twice is a wasted run and a confusing report.
    seen, unique = set(), []
    for path in found:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def main(argv: list[str]) -> int:
    import cv2

    paths = resolve(argv)
    if not paths:
        print(__doc__)
        print(f"No images found in: {', '.join(argv) if argv else 'data/code_reads'}")
        print("Run Test Read in the app and press Save crops to make some.")
        return 2

    from label_detections.core import code_reader as cr

    ok, reason = cr.available()
    if not ok:
        print(f"No decoder: {reason}")
        return 1
    module, _ = cr.backend()

    status = 1
    for path in paths:
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
            print("\n  Nothing read on any attempt.")
            # Without the label definition this cannot know the region, so it
            # offers the arithmetic for the two plausible readings of the crop
            # rather than pretending to one answer.
            from label_detections.core import codes as cd
            print(f"     If this crop is the code alone, that is "
                  f"{cd.px_per_module('upca', width):.2f} px per module for a "
                  f"UPC-A.")
            print(f"     If the code is about 80% of it, "
                  f"{cd.px_per_module('upca', width * 0.8):.2f}.")
            print(f"     Under {cd.MIN_PX_PER_MODULE:.0f} is marginal for a "
                  f"photographed, rectified code -- it reads on a good frame "
                  f"and not on the next one, and no decoder setting fixes it.")
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
