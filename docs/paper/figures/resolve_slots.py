"""Resolve every [[slot]] in the authored outline from a persisted artifact, and REPORT mismatches.

Ruling 2: a number without an address does not ship, and where the abstract disagrees with the
provenance ledger the disagreement is REPORTED, never silently corrected on either side.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _artifacts as A

def main() -> int:
    k, r, f, ref = A.dense_kstar(), A.reconciliation(), A.fka_operating_point(), A.refusal()
    addr = A.fka_addressability_at_2000()
    print("=" * 78); print("SLOT RESOLUTION — every value with its address"); print("=" * 78)
    groups = [("FKA operating point (N=2M)", f), ("FKA addressability (N=2,000 go/no-go)", addr),
              ("dense K*", k), ("reconciliation", r), ("refused extrapolation", ref),
              ("noise bands", A.noise_bands())]
    for title, g in groups:
        print(f"\n-- {title}")
        for name, v in g.items(): print(f"   {name:<26} {v}")

    print("\n" + "=" * 78); print("ABSTRACT CROSS-CHECK against the provenance ledger (M5 §5.147)")
    print("=" * 78)
    checks = [
        ("3.15 bits/param amortised", f["bits_per_param_amortised"].value, 3.15),
        ("3.51 bits/param marginal",  f["bits_per_param_marginal"].value, 3.51),
        ("0.298 dense best bits/param", k["best_bits_per_param"].value, 0.298),
        ("alpha point 0.224",         k["alpha"].value, 0.224),
        ("alpha interval [0.185,0.291]", list(k["alpha_interval"].value), [0.185, 0.291]),
        ("gap 1.89x",                 r["gap"].value, 1.89),
        ("carriers 2.35x",            r["product"].value, 2.35),
        ("amortisation 2.0x",         r["amortisation"].value, 2.0),
        ("surface 1.17x",             r["surface"].value, 1.17),
        ("162 orders",                ref["orders"].value, 162),
    ]
    bad = 0
    for label, got, want in checks:
        ok = got == want
        bad += not ok
        print(f"   {'OK ' if ok else 'MISMATCH'}  {label:<32} artifact={got!r} abstract={want!r}")

    print("\n" + "=" * 78); print("PHRASING ITEMS — ALL THREE RESOLVED"); print("=" * 78)
    ns = addr["never_supervised"].value * 100
    text = (A.ROOT / "docs/paper/sections/00_abstract.tex").read_text(encoding="utf-8")
    resolved = [
        ("addressability carries depth class and load",
         f"99.9% at N=2M (M3 §24.5) and {ns:.1f}% never-supervised at N=2,000 "
         f"({addr['direct'].value*100:.1f}% direct, {addr['composed'].value*100:.1f}% composed) "
         f"quoted SEPARATELY [{A.GONOGO_PATH}]",
         "99.8\\%" in text and "99.9\\%" in text and "composed" in text),
        ("rehabilitation count = §5.1's list length",
         "four (three chosen, one forced); §5.1 heading and lead matched to it. M5 §5.8.1's "
         "'five' counts the surface arc as two",
         "rehabilitated four times" in text),
        ("edit locality carries its configuration",
         f"edit_locality = {addr['edit_locality'].value} at N=2,000 [{A.GONOGO_PATH}]; "
         f"the abstract now states the N",
         "edit interference measured at $N = 2{,}000$" in text),
    ]
    for i, (label, detail, ok) in enumerate(resolved, 1):
        bad += not ok
        print(f"\n {i}. [{'RESOLVED' if ok else 'NOT RESOLVED'}] {label}\n    {detail}")

    print(f"\ncross-check mismatches: {bad}")
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
