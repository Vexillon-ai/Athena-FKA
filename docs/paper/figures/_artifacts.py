"""Artifact loader for the paper's figures and tables — the citation-by-address backbone.

**Every number in the paper resolves through here, and every value carries its address.** Nothing
is hand-entered: a figure script that types a number instead of loading one is the defect this
module exists to make impossible.

An address is `(artifact path, decision-record section)`. `Value` carries both, and
`resolve_slots.py` prints them so a reader can check any figure against the run that produced it.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Value:
    """A number with its provenance. Printing one shows where it came from."""

    value: object
    artifact: str
    section: str
    note: str = ""

    def __str__(self) -> str:
        return f"{self.value}  [{self.artifact} | {self.section}]"


def _json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# the frontier artifact — the one file that already collects the cross-track numbers
# ---------------------------------------------------------------------------------------------

FRONTIER_PATH = "experiments/2026-08-04_frontier/plot_data.json"


def frontier() -> dict:
    return _json(FRONTIER_PATH)


# ---------------------------------------------------------------------------------------------
# FKA side
# ---------------------------------------------------------------------------------------------

GONOGO_PATH = "experiments/2026-08-02_m3-gonogo/gonogo.json"


def fka_gonogo() -> dict:
    """The go/no-go artifact. NOTE: measured at N = 2,000, not at the 2M operating point."""
    return _json(GONOGO_PATH)


def fka_operating_point() -> dict[str, Value]:
    f = frontier()["fka"]["points"][0]
    return {
        "n_entities": Value(f["n_entities"], FRONTIER_PATH, "M3 §24.5"),
        "n_facts": Value(f["n_facts"], FRONTIER_PATH, "M3 §24.5"),
        "bits_per_entity": Value(f["bits_per_entity"], FRONTIER_PATH, "M3 §24.5"),
        "bits_per_param_amortised": Value(f["bits_per_param_amortised"], FRONTIER_PATH, "M3 §30"),
        "bits_per_param_marginal": Value(f["bits_per_param_marginal"], FRONTIER_PATH, "M3 §30"),
        "bits_per_bit_amortised": Value(f["bits_per_bit_int8_amortised"], FRONTIER_PATH, "M3 §30"),
        "bits_per_bit_marginal": Value(f["bits_per_bit_int8_marginal"], FRONTIER_PATH, "M3 §30"),
        "compression": Value(f["compression"], FRONTIER_PATH, "M3 §11.5"),
    }


def fka_addressability_at_2000() -> dict[str, Value]:
    """N = 2,000 go/no-go. Kept separate from the 2M figures ON PURPOSE (stale-constant rule)."""
    g = fka_gonogo()
    a = g["addressability"]
    return {
        "never_supervised": Value(a["never_supervised"], GONOGO_PATH, "M3 §2 gate"),
        "direct": Value(a["by_depth"]["direct"], GONOGO_PATH, "M3 §2 gate"),
        "composed": Value(a["by_depth"]["composed"], GONOGO_PATH, "M3 §2 gate"),
        "edit_locality": Value(g["edit_locality"], GONOGO_PATH, "M3 §2 gate"),
        "n_probes": Value(a["n"], GONOGO_PATH, "M3 §2 gate"),
    }


FORK_A_PATH = "experiments/2026-08-02_m2-fork-a-joint/rescore_unseen.json"
FORK_C_PATH = "experiments/2026-08-02_m2-fork-c/forkc.json"


def fork_never_supervised() -> dict[str, Value]:
    """The fork (a)/(c) exhibit, never-supervised only.

    Fork (c) is stored outright. **Fork (a)'s 0/154 is not stored anywhere**: the artifact records
    the whole unseen set (191 queries, 19.37% correct) and M2 §11.3 partitions it into 37 whose
    entity embedding was trained elsewhere (all correct) and 154 whose embedding never was (none
    correct). So 154 is derived here, and the derivation is checked against the count §11.3 states
    — if the artifact ever moves, this raises instead of quietly redrawing the exhibit.
    """
    a, c = _json(FORK_A_PATH)["final"], _json(FORK_C_PATH)["final"]
    n_unseen = a["n_unseen_queries"]
    correct_unseen = round(a["retrieval_accuracy_unseen"] * n_unseen)
    n_never_trained = n_unseen - correct_unseen
    if (n_never_trained, correct_unseen) != (154, 37):
        raise AssertionError(
            f"fork (a) split no longer reproduces M2 §11.3: derived {correct_unseen} pair-new / "
            f"{n_never_trained} never-trained from {n_unseen} unseen; §11.3 states 37 / 154")
    return {
        "a": Value(0.0, FORK_A_PATH, "M2 §11.3", "derived: 0 of the 154 never-trained"),
        "a_n": Value(n_never_trained, FORK_A_PATH, "M2 §11.3", "191 unseen less 37 pair-new"),
        "c": Value(c["retrieval_never_supervised"], FORK_C_PATH, "M2 §12"),
        "c_n": Value(c["n_never_supervised"], FORK_C_PATH, "M2 §12"),
    }


def fka_seed_sweep() -> dict[str, list]:
    """Ten seeds x seven compression rungs (M5 §5.131). Parsed from the run's own log."""
    log = (ROOT / "experiments/2026-08-04_fka-seed-sweep/sweep.log").read_text(
        encoding="utf-8", errors="replace"
    )
    rows: dict[str, list] = {}
    for line in log.splitlines():
        m = re.search(r"seed (\d+)\s+(.+?)\s+NEVER-SUP\s+([\d.]+)%", line)
        if m:
            rows.setdefault(m.group(2).strip(), []).append(float(m.group(3)))
    return rows


# ---------------------------------------------------------------------------------------------
# dense side — parsed from the run logs that produced them
# ---------------------------------------------------------------------------------------------

DENSE_LOGS = {
    "syllable_1M": [
        "experiments/2026-08-03_dense-kstar-1M-rerun/run.log",
        "experiments/2026-08-03_dense-kstar-replicate/stage2.log",
        "experiments/2026-08-03_dense-kstar-replicate/k20000.log",
        "experiments/2026-08-03_dense-kstar-replicate/k18000.log",
        "experiments/2026-08-03_dense-kstar-replicate/signature.log",
        "experiments/2026-08-03_dense-kstar-replicate/k16000_n10.log",
    ],
    "syllable_10M": [
        "experiments/2026-08-03_dense-kstar-10M-rerun/run.log",
        "experiments/2026-08-03_dense-kstar-10M-rerun/coarse2.log",
        "experiments/2026-08-03_dense-kstar-10M-rerun/fine.log",
        "experiments/2026-08-03_dense-kstar-replicate/run.log",
        "experiments/2026-08-03_dense-kstar-replicate/stage2.log",
        "experiments/2026-08-03_dense-kstar-replicate/fiveseed.log",
    ],
    "bpe_1M": [
        "experiments/2026-08-04_dense-bpe-control/run.log",
        "experiments/2026-08-04_dense-bpe-control/kstar.log",
    ],
}

#: Which model size each log family was run at, so a 1M arm in a shared log is not read as 10M.
_SIZE_OF = {"syllable_1M": "1M", "syllable_10M": "10M", "bpe_1M": "1M"}


def dense_arms(family: str) -> dict[int, list[tuple[float, float]]]:
    """``{key_count: [(corrected_pct, final_loss), ...]}`` for one surface/size family.

    Parsed from logs rather than from `ladder.json`, because the corrected column is computed in
    the runner's report and the ladder artifact stores raw accuracy plus the accounting.
    """
    out: dict[int, list[tuple[float, float]]] = {}
    want_1m = _SIZE_OF[family] == "1M"
    for rel in DENSE_LOGS[family]:
        path = ROOT / rel
        if not path.exists():
            continue
        keys = loss = size = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"n_entities=([\d,]+)", line)
            if m:
                keys = int(m.group(1).replace(",", ""))
                continue
            m = re.search(r"([\d,]+) params\s", line)
            if m:
                size = int(m.group(1).replace(",", ""))
            m = re.search(r"step\s+[\d,]+/[\d,]+ \(100\.0%\).*loss ([\d.]+)", line)
            if m:
                loss = float(m.group(1))
                continue
            m = re.search(r"qa_heldout\s+POOLED.*corrected\s+([\d.]+)%", line)
            if m and keys is not None:
                out.setdefault(keys, []).append((float(m.group(1)), loss))
    return out


def dense_bits_per_param(family: str, half: str = "qa_heldout") -> dict[int, list[float]]:
    """``{key_count: [bits_per_param, ...]}`` per seed, from the same logs as :func:`dense_arms`.

    **This series was always in the logs.** The runner prints `bits/param` on every POOLED line;
    F1 broadcast a single max across every N instead, which is the instrument failure filed in §7.
    Defaults to the HELD-OUT half: `qa_trained` measures facts whose QA rendering was in training,
    and comparing that against FKA's never-supervised figure would cross probe regimes.
    """
    out: dict[int, list[float]] = {}
    for rel in DENSE_LOGS[family]:
        path = ROOT / rel
        if not path.exists():
            continue
        keys = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"n_entities=([\d,]+)", line)
            if m:
                keys = int(m.group(1).replace(",", ""))
                continue
            m = re.search(rf"{half}\s+POOLED.*bits/param\s+([\d.]+)", line)
            if m and keys is not None:
                out.setdefault(keys, []).append(float(m.group(1)))
    return out


def dense_best_bits_provenance() -> dict[str, Value]:
    """Where 0.298 actually comes from, because the frontier quotes it as a dense ceiling.

    It is the **qa_trained** half at **N = 8,000** in the capacity sweep — a different experiment
    from the K* ladder, at a smaller N than any ladder arm. The held-out figure at that same arm is
    0.2720, and the best held-out figure anywhere on the ladder is 0.1322. Quoting the larger number
    is the generous reading for the baseline and is kept for that reason; it may not be read as a
    per-N value, and it is not one.
    """
    p = "experiments/2026-08-02_dense-capacity-sweep/sweep.log"
    return {
        "quoted": Value(frontier()["dense"]["best_bits_per_param_any_arm"], FRONTIER_PATH,
                        "M5 §5.146", "qa_trained half, N=8,000"),
        "source_log": Value(p, p, "M5 §5.19"),
        "n_entities": Value(8000, p, "M5 §5.19"),
        "held_out_same_arm": Value(0.2720, p, "M5 §5.19"),
    }


def dense_kstar() -> dict[str, Value]:
    d = frontier()["dense"]
    p1, p10 = d["points"]
    s = d["surface_factor"]
    return {
        "K_1M": Value(p1["K_star_P"], FRONTIER_PATH, "M5 §5.128"),
        "K_1M_bracket": Value(tuple(p1["bracket"]), FRONTIER_PATH, "M5 §5.128"),
        "ppk_1M": Value(p1["params_per_key"], FRONTIER_PATH, "M5 §5.128"),
        "K_10M": Value(p10["K_star_P"], FRONTIER_PATH, "M5 §5.118"),
        "K_10M_bracket": Value(tuple(p10["bracket"]), FRONTIER_PATH, "M5 §5.118"),
        "ppk_10M": Value(p10["params_per_key"], FRONTIER_PATH, "M5 §5.118"),
        "alpha": Value(d["alpha"]["point"], FRONTIER_PATH, "M5 §5.128.3"),
        "alpha_interval": Value(tuple(d["alpha"]["interval"]), FRONTIER_PATH, "M5 §5.128.3"),
        "K_BPE_1M": Value(s["K_star_BPE"], FRONTIER_PATH, "M5 §5.143"),
        "surface_factor": Value(s["value"], FRONTIER_PATH, "M5 §5.143"),
        "best_bits_per_param": Value(d["best_bits_per_param_any_arm"], FRONTIER_PATH, "M5 §5.146"),
        "params_1M": Value(p1["params"], FRONTIER_PATH, "M5 §5.52"),
        "params_10M": Value(p10["params"], FRONTIER_PATH, "M5 §5.52"),
    }


def dense_seed_counts() -> dict[str, dict[int, int]]:
    """``{family: {key_count: n_seeds}}``, counted from the logs.

    The frontier artifact records ``n_seeds`` as one number per model size (10 at 1M, 5 at 10M),
    which is true of the **cliff-bracket arms** and not of every plotted arm. Any caption that
    states a seed count per point must read it from here, not from that scalar.
    """
    return {fam: {k: len(v) for k, v in sorted(dense_arms(fam).items())} for fam in DENSE_LOGS}


def refinements() -> tuple[list[dict], dict]:
    """The seven refinements (M5 §5.147), transcribed once into an artifact.

    Returns ``(rows, checkpoints)``. The caller reconstructs each row's alpha interval from its
    brackets; :func:`check_refinement_alphas` verifies that reconstruction against the alpha
    values the ledger states outright.
    """
    d = _json(REFINEMENTS_PATH)
    return d["refinements"], d["alpha_checkpoints"]


REFINEMENTS_PATH = "experiments/2026-08-04_frontier/refinements.json"


def reconciliation() -> dict[str, Value]:
    r = frontier()["reconciliation"]
    return {
        "gap": Value(r["gap"], FRONTIER_PATH, "M5 §5.143.2"),
        "amortisation": Value(r["carriers"]["amortisation"], FRONTIER_PATH, "M5 §5.53"),
        "surface": Value(r["carriers"]["surface"], FRONTIER_PATH, "M5 §5.143"),
        # THE PRODUCT IS LOADED, NEVER RECOMPUTED. 2.35 is the rounded product of the unrounded
        # carriers; multiplying the *rounded* carriers (2.00 x 1.17) gives 2.34. One source, so the
        # figure and the table cannot disagree — they did, at 2.34 and 2.35, until this was fixed.
        "product": Value(r["carriers"]["product"], FRONTIER_PATH, "M5 §5.143.2"),
        "reference_ppk": Value(r["reference_params_per_key"], FRONTIER_PATH, "M5 §5.53"),
        "ours_syllable": Value(r["our_params_per_key_syllable"], FRONTIER_PATH, "M5 §5.143.2"),
        "ours_bpe": Value(r["our_params_per_key_bpe"], FRONTIER_PATH, "M5 §5.143.2"),
        "verdict": Value(r["verdict"], FRONTIER_PATH, "M5 §5.143"),
    }


def refusal() -> dict[str, Value]:
    r = frontier()["refused_extrapolation"]
    return {
        "orders": Value(r["disagreement_orders_of_magnitude"], FRONTIER_PATH, "M5 §5.146"),
        "power": Value(r["power_form_params"], FRONTIER_PATH, "M5 §5.146"),
        "log": Value(r["log_form_params"], FRONTIER_PATH, "M5 §5.146"),
        "measured_decade": Value(r["measured_decade_orders"], FRONTIER_PATH, "M5 §5.146"),
    }


def margin_trajectories() -> dict:
    """The per-fact matrix M3 §13's shape verdict was computed from (788 facts x 8 loads)."""
    return _json("experiments/2026-08-02_m3-shape/shape.json")["trajectory_matrix"]


def check_refinement_alphas(alpha_of) -> None:
    """Fail loudly if the transcribed brackets stop reproducing the ledger's stated alphas.

    ``alpha_of(row) -> (lo, hi)``. The brackets are a transcription, and a transcription with no
    check is just a hand-entered number in a different file.
    """
    rows, cp = refinements()
    got = {r["n"]: alpha_of(r) for r in rows}
    for name, want, actual in (
        ("r1_alpha_min_before", cp["r1_alpha_min_before"], got[1][0]),
        ("r3_alpha_max_before", cp["r3_alpha_max_before"], got[1][1]),
        ("r2_alpha_min_after", cp["r2_alpha_min_after"], got[2][0]),
        ("r3_alpha_max_after", cp["r3_alpha_max_after"], got[3][1]),
        ("final_lo", cp["final_interval"][0], got[7][0]),
        ("final_hi", cp["final_interval"][1], got[7][1]),
    ):
        if abs(want - actual) > 5e-4:
            raise AssertionError(
                f"refinement transcription disagrees with M5 §5.147 at {name}: "
                f"ledger states {want}, brackets reconstruct {actual:.4f}")


def noise_bands() -> dict[str, Value]:
    from fka.eval import kstar as K

    return {
        "non_cliff": Value(K.NOISE_SIGMA_NON_CLIFF * 100, "fka/eval/kstar.py", "M5 §5.102.2"),
        "cliff": Value(K.NOISE_SIGMA_CLIFF * 100, "fka/eval/kstar.py", "M5 §5.102.2"),
    }
