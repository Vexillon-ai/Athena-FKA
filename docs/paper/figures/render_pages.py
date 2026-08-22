"""Rasterise the compiled paper to PNGs so the layout can be INSPECTED, not assumed.

LaTeX's overfull/underfull warnings say a box is wrong; they do not say whether it *looks* wrong.
This renders pages at review resolution so figures, tables and diagrams can be checked by eye.

Run: ``python docs/paper/figures/render_pages.py [page ...]``  (1-indexed; default: all)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pypdfium2 as pdfium

HERE = Path(__file__).resolve().parent
PDF = HERE.parent / "factored_knowledge_architecture_paper.pdf"
OUT = HERE.parent / "_review"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    doc = pdfium.PdfDocument(PDF)
    wanted = [int(a) for a in sys.argv[1:]] or list(range(1, len(doc) + 1))
    for p in wanted:
        page = doc[p - 1]
        img = page.render(scale=2.0).to_pil()
        path = OUT / f"page_{p:02d}.png"
        img.save(path)
        print(f"  {path.name}  {img.width}x{img.height}")
    print(f"{len(wanted)} page(s) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
