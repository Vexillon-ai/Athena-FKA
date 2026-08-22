"""Phase 5's dense baseline: probe, exposure ladder, and the sized run (M5 §4).

Three modes, in the order the program's own laws require:

* ``--mode probe``   measured step cost AND peak allocation, per size, against the real stream,
                     with a projected wall clock. Nothing may be launched before this.
* ``--mode ladder``  the exposure ladder: independent runs at increasing exposures/fact, each
                     with its own complete LR schedule, each scored by the frozen capacity
                     harness. Locates the knee before the headline run commits to a regime.
* ``--mode train``   one sized run, resumable across sessions.

Every mode takes the GPU lock (one job at a time on this box) and every non-smoke run writes a
checkpoint before exiting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fka.data.corpus_gen import generate_corpus  # noqa: E402
from fka.data.tokenizer import CharTokenizer  # noqa: E402
from fka.dense.probe import GoldStubRecall, WrongAnswerRecall, dense_capacity  # noqa: E402
from fka.dense.stream import DenseCorpusStream, DenseDataConfig  # noqa: E402
from fka.dense.surface import syllable_tokenizer  # noqa: E402
from fka.dense.train import (  # noqa: E402
    DenseTrainConfig,
    NonFiniteLoss,
    plan_run,
    probe_cost,
    recipe_for,
    train_dense,
)
from fka.eval.accounting import StorageAccount  # noqa: E402
from fka.eval.kernel_eval import most_frequent_answer_baseline  # noqa: E402
from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402


#: Generic-English merge corpus for the BPE surface: the repository's own prose. It contains no
#: entity name and no syllable spelling, which is the property that matters — corpus-trained merges
#: are permanently excluded (M5 §5.82.1) because recurring names would earn their own tokens.
_BPE_MERGE_GLOBS = ("docs/**/*.md", "CLAUDE.md")


def _bpe_tokenizer(vocab_size: int):
    """A BPE trained on generic English, with the corpus charset guaranteed present.

    ``required_chars`` is not optional: a delimiter or capitalised value name absent from English
    prose would tokenise to ``<unk>`` and the arm would read 0% (M5 §5.133).
    """
    import glob

    from fka.data.tokenizer import DEFAULT_CHARS
    from fka.dense.bpe import BPETokenizer, train_bpe

    paths = sorted({p for g in _BPE_MERGE_GLOBS for p in glob.glob(g, recursive=True)})
    text = "\n".join(Path(p).read_text(encoding="utf-8", errors="replace") for p in paths)
    required = DEFAULT_CHARS + "?|:.\n"
    specials = ("<unk>", "<eos>", "<pad>")
    merges = train_bpe(text, vocab_size, specials, required_chars=required)
    return BPETokenizer(merges=merges, specials=specials,
                        base_chars=tuple(sorted(set(text) | set(required))))


def _tokenizer_for(args):
    if args.surface == "syllable":
        return syllable_tokenizer()
    if args.surface == "bpe":
        return _bpe_tokenizer(args.bpe_vocab)
    return CharTokenizer()


def build(args) -> tuple:
    corpus = generate_corpus(
        n_entities=args.n_entities,
        seed=args.corpus_seed,
        probe_fraction=args.probe_fraction,
        # The generator demands 100x name-space headroom so per-name bits stay exact; past
        # ~160k entities the default 4096x4096 no longer supplies it.
        n_given_names=args.n_given_names,
        n_surnames=args.n_surnames,
        n_cities=args.n_cities,
        n_employers=args.n_employers,
        birth_year_range=tuple(args.birth_year_range),
        # Restricting relations holds the KEY COUNT fixed while cutting the bits — the
        # discriminator between a bits limit and a key-discrimination limit (M5 §5.15).
        **({"relations": tuple(args.relations.split(","))} if args.relations else {}),
    )
    # The syllable surface needs its inventory in the vocabulary; every other surface renders
    # inside the corpus charset (M5 §5.32).
    tokenizer = _tokenizer_for(args)
    stream = DenseCorpusStream(
        corpus,
        tokenizer,
        DenseDataConfig(
            exposures=args.exposures,
            qa_entity_fraction=args.qa_entity_fraction,
            qa_period=args.qa_period,
            surface=args.surface,
            name_units=args.name_units,
            seed=args.corpus_seed,
        ),
    )
    return corpus, tokenizer, stream


def make_cfg(args, size: str) -> DenseTrainConfig:
    return recipe_for(
        size,
        block_size=args.block_size,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        amp=not args.no_amp,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        weight_decay=args.weight_decay,
        **({"lr": args.lr} if args.lr else {}),
    )


# =======================================================================================
# gates that run before any number is admissible
# =======================================================================================


def run_gates(corpus, tokenizer, stream, *, n: int = 24) -> dict:
    """Gold stub and its red twin on the dense probe path (M5 §4.2)."""
    ids = stream.probe_fact_ids()[:n]
    # The probe surface MUST be the training surface. Omitting it silently probed a
    # syllable-trained model with verbose prompts and reported 0.00% (M5 §5.41). The gate did not
    # catch it because the stub built its own prompts on the same wrong surface — self-consistent,
    # and therefore green.
    surface = stream.surface
    gold = dense_capacity(
        GoldStubRecall(corpus, tokenizer, ids, surface=surface), tokenizer, corpus, ids,
        n_params=1, surface=surface,
    )
    wrong = dense_capacity(
        WrongAnswerRecall(corpus, tokenizer, ids, surface=surface), tokenizer, corpus, ids,
        n_params=1, surface=surface,
    )
    chance = most_frequent_answer_baseline(list(corpus.memory_pairs(ids)))
    out = {
        "n": int(ids.size),
        "gold_stub_accuracy": gold.accuracy,
        "wrong_answer_accuracy": wrong.accuracy,
        "most_frequent_answer_baseline": chance,
        "passes": gold.accuracy == 1.0 and wrong.accuracy <= chance + 1e-9,
    }
    print(
        f"   gate: gold stub {gold.accuracy:.1%}  wrong-answer stub {wrong.accuracy:.1%} "
        f"(chance {chance:.1%})  -> {'PASS' if out['passes'] else 'FAIL'}"
    )
    return out


# =======================================================================================
# modes
# =======================================================================================


def mode_probe(args, corpus, tokenizer, stream, device) -> dict:
    sizes = args.sizes.split(",")
    rows = []
    for size in sizes:
        cfg = make_cfg(args, size)
        print(f"-- probing {size}")
        try:
            row = probe_cost(stream, cfg, device=device, repeats=args.repeats)
        except torch.OutOfMemoryError as exc:  # pragma: no cover - device-dependent
            row = {"size": size, "error": f"OOM: {exc}"}
        else:
            print(
                f"   {row['n_params']:>12,} params  {row['seconds_per_step'] * 1e3:8.2f} ms/step  "
                f"{row['tokens_per_sec']:>10,.0f} tok/s  peak {row['peak_gb']:6.2f} GB  "
                f"{'stable' if row['stable'] else 'UNSTABLE'}"
            )
            print(
                f"   {row['steps_per_epoch']:,} steps/epoch x {row['exposures']} exposures = "
                f"{row['total_steps']:,} steps -> {row['projected_hours']:.2f} h"
            )
        rows.append(row)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"probes": rows, "exposure": stream.exposure_report()}


def _measure(model, tokenizer, corpus, stream, device, amp_dtype, args) -> dict:
    """Capacity on both halves of the QA split, plus the frozen storage accounting."""
    heldout = stream.probe_fact_ids()
    trained = stream.probe_fact_ids(qa_trained=True)
    n_params = model.n_params()
    out: dict = {"n_params": n_params}
    for label, ids in (("qa_heldout", heldout), ("qa_trained", trained)):
        if ids.size == 0:
            continue
        if ids.size > args.max_probes:
            # A PREFIX would be a relation-ordered slice, because fact_id = relation*N + entity:
            # capping 13,149 probes at 4,000 would have measured birth_year and a little
            # birth_city. Subsample uniformly instead, and check the mix survived.
            rng = np.random.default_rng(args.seed ^ 0xC0DE)
            ids = np.sort(rng.choice(ids, size=args.max_probes, replace=False))
        present = {p.key[1] for p in corpus.memory_pairs(ids)}
        missing = set(corpus.relations) - present
        if missing:
            raise ValueError(f"probe sample for {label} is missing relations {sorted(missing)}")
        report = dense_capacity(
            model, tokenizer, corpus, ids, device=device, amp_dtype=amp_dtype,
            batch_size=args.probe_batch, n_params=n_params, surface=stream.surface,
        )
        out[label] = report.to_dict()
        # Per-class BEFORE the aggregate (CLAUDE.md): four relations of very different difficulty
        # share one readout, so the pooled figure describes the relation mix as much as the model.
        for name, r in report.per_relation.items():
            print(f"   {label:<12} {name:<12} n={r.n_probes:>6,}  acc {r.accuracy:7.2%}  "
                  f"corrected {r.corrected_accuracy:7.2%}  bits {r.stored_bits:>12,.0f}")
        print(f"   {label:<12} {'POOLED':<12} n={report.n_probes:>6,}  acc {report.accuracy:7.2%}"
              f"  corrected {report.corrected_fraction_of_entropy:7.2%}  "
              f"bits/param {report.bits_per_param:.4f}")

    head = out.get("qa_heldout")
    if head:
        # Entropy at the TRAINED N, by construction: corpus.total_bits is this corpus's own.
        account = StorageAccount(
            n_facts=int(corpus.n_facts),
            per_fact_bits=0.0,  # a dense LM stores nothing per fact; the weights are the store
            shared_params=n_params,
            knowledge_bits=head["estimated_corpus_bits_corrected"],
            breakdown={
                "design": "dense LM (plain, firewall off)",
                "n_entities": corpus.n_entities,
                "corpus_bits_per_fact": corpus.total_bits / corpus.n_facts,
                "extrapolated_from": "qa_heldout probe half",
            },
        )
        out["account"] = account.to_dict()
        print(f"   accounting  {account}")
    return out


def mode_ladder(args, corpus, tokenizer, stream, device) -> dict:
    exposures = [int(e) for e in args.ladder.split(",")]
    amp_dtype = torch.bfloat16 if (not args.no_amp and device.type == "cuda") else None
    rows = []
    for e in exposures:
        stream.cfg = DenseDataConfig(
            exposures=e,
            qa_entity_fraction=args.qa_entity_fraction,
            qa_period=args.qa_period,
            surface=args.surface,
            name_units=args.name_units,
            seed=args.corpus_seed,
        )
        cfg = make_cfg(args, args.sizes.split(",")[0])
        plan = plan_run(stream, cfg)
        # Warmup is a fraction of the schedule, not a constant: at the bottom of the ladder a
        # fixed 200 steps would be most of the run, and the rung would measure the warmup.
        cfg.warmup_steps = min(cfg.warmup_steps, max(1, plan["total_steps"] // 20))
        print(f"\n== exposures={e}  {plan['total_steps']:,} steps "
              f"({plan['tokens_total'] / 1e9:.2f}G tokens)")
        t0 = time.perf_counter()
        try:
            model, state = train_dense(
                stream, cfg, device=device,
                out_dir=Path(args.out) / f"E{e}" if args.out else None,
            )
        except NonFiniteLoss as exc:
            print(f"   NON-FINITE: {exc}")
            rows.append({"exposures": e, "error": str(exc)})
            continue
        row = {
            "exposures": e,
            "steps": state.step,
            "final_loss": state.losses[-1],
            "seconds": state.seconds,
            "tokens_per_sec": state.tokens_per_sec,
            "peak_gb": state.peak_bytes / 1e9,
            **plan,
            **_measure(model, tokenizer, corpus, stream, device, amp_dtype, args),
        }
        row["wall_clock_minutes"] = (time.perf_counter() - t0) / 60
        rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"ladder": rows}


def mode_train(args, corpus, tokenizer, stream, device) -> dict:
    cfg = make_cfg(args, args.sizes.split(",")[0])
    plan = plan_run(stream, cfg)
    amp_dtype = torch.bfloat16 if (not args.no_amp and device.type == "cuda") else None
    print(f"== {plan['total_steps']:,} steps, {plan['tokens_total'] / 1e9:.2f}G tokens")
    model, state = train_dense(
        stream, cfg, device=device, out_dir=args.out, resume=args.resume,
        run_steps=args.run_steps,
    )
    return {
        "train": state.to_dict(),
        "plan": plan,
        **_measure(model, tokenizer, corpus, stream, device, amp_dtype, args),
    }


# =======================================================================================
# CLI
# =======================================================================================


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("probe", "ladder", "train"), default="probe")
    p.add_argument("--sizes", default="10M", help="comma-separated; ladder/train use the first")
    p.add_argument("--n-entities", type=int, default=20_000)
    p.add_argument("--exposures", type=int, default=64)
    p.add_argument("--ladder", default="1,4,16,64")
    p.add_argument("--qa-entity-fraction", type=float, default=0.5)
    p.add_argument("--qa-period", type=int, default=5)
    p.add_argument("--surface", default="verbose",
                   choices=("verbose", "terse", "terse_named", "syllable", "bpe"))
    p.add_argument("--bpe-vocab", type=int, default=735,
                   help="BPE vocabulary size; set by the <=10%% embedding-share constraint "
                        "(735 at 1M, 3193 at 10M — M5 §5.79)")
    p.add_argument("--name-units", type=int, default=None,
                   help="syllables per name (syllable surface only); default = minimum that "
                        "addresses the corpus. M5 §5.57's saturation discriminator.")
    p.add_argument("--probe-fraction", type=float, default=0.05)
    p.add_argument("--relations", default=None, help="comma-separated subset; default all four")
    # Value-space sizes: the only way to hold value-load fixed while the key count moves.
    p.add_argument("--n-cities", type=int, default=512)
    p.add_argument("--n-employers", type=int, default=1024)
    p.add_argument("--birth-year-range", type=int, nargs=2, default=(1900, 2000))
    p.add_argument("--n-given-names", type=int, default=4096)
    p.add_argument("--n-surnames", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=None)
    # Weight decay is an audit axis at high load, not a constant: it penalises exactly the large
    # weights memorisation needs, so a value that is harmless with capacity to spare can bind once
    # the corpus fills the model (M5 §5.14).
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corpus-seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument("--max-probes", type=int, default=4000)
    p.add_argument("--probe-batch", type=int, default=256)
    p.add_argument("--run-steps", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--force-gpu-lock", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    if args.smoke:
        args.n_entities, args.exposures, args.ladder = 200, 2, "1,2"
        args.sizes, args.block_size, args.batch_size = "tiny", 128, 4
        args.device, args.warmup_steps, args.max_probes = "cpu", 2, 16
        args.probe_batch, args.repeats, args.probe_fraction = 8, 2, 0.2

    seed_everything(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    corpus, tokenizer, stream = build(args)
    print(f"== {corpus!r}  fingerprint {corpus.fingerprint()[:16]}")
    report = stream.exposure_report()
    print(f"   surface {report['surface']}  world {report['world_fingerprint'][:12]}  "
          f"surface-hash {report['surface_fingerprint'][:12]}  "
          f"{report['tokens_per_fact']:.1f} tok/fact  {report['bits_per_token']:.4f} bits/token")
    print(f"   {report['tokens_per_epoch']:,} tokens/epoch, "
          f"{report['corpus_bits_per_fact']:.2f} bits/fact at N={corpus.n_entities:,}")
    print(f"   probe facts: {report['n_probe_facts_heldout_qa']:,} QA-heldout / "
          f"{report['n_probe_facts_qa_trained']:,} QA-trained")

    gates = run_gates(corpus, tokenizer, stream)
    if not gates["passes"]:
        print("== ABORT: the dense probe path failed its gold stub / red twin")
        return 1

    runner = {"probe": mode_probe, "ladder": mode_ladder, "train": mode_train}[args.mode]
    with gpu_lock(force=args.force_gpu_lock) if device.type == "cuda" else _null():
        result = runner(args, corpus, tokenizer, stream, device)

    payload = {
        "mode": args.mode,
        "args": vars(args),
        "corpus": corpus.summary(),
        "exposure": report,
        "gates": gates,
        **result,
    }
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{args.mode}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nwrote {out / f'{args.mode}.json'}")
    return 0


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
