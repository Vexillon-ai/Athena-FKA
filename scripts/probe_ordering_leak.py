"""Is the 10M model storing FACTS, or memorising the ORDER of the token stream? (M5 §5.62.3)

The 10M model reached **2.2x lower loss** and **7.4x lower recall** than the 1M model on an
identical corpus, surface, recipe and step count. `DenseCorpusStream` shuffles line order **per
variant**, and there are only five variants, so at E = 535 each of five orderings is presented
~107 times byte-identical. A model with the capacity to memorise the stream can drive loss down
without learning anything retrievable.

The discriminator is a **two-axis, teacher-forced loss decomposition** — no generation, so it cannot
fail for any reason generation fails for:

    token role  x  ordering
    ----------     --------
    subject        the TRAINED five permutations
    value          a FRESH unseen permutation of the same lines
    other

Registered predictions (M5 §5.62.3):

* **ordering memorisation** — the 10M advantage sits in **subject** tokens and collapses on a fresh
  ordering; the 1M model is ordering-insensitive and its advantage sits in **value** tokens;
* **recipe failure** — re-ordering changes little and the 10M model's *value* loss is no better than
  the 1M model's;
* **readout failure** — the 10M model's value loss IS better and is ordering-robust, so the facts are
  stored and generation is the broken link.

The instrument asserts its own manipulation took effect before reporting anything, per the
revision-test rule: a fresh ordering that silently did not apply would produce a null that looks
exactly like ordering-insensitivity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from fka.data.corpus_gen import generate_corpus  # noqa: E402
from fka.dense.stream import DenseCorpusStream, DenseDataConfig  # noqa: E402
from fka.dense.surface import syllable_tokenizer  # noqa: E402
from fka.kernel.gpu_lock import gpu_lock  # noqa: E402
from fka.kernel.model import KernelConfig, ReasoningKernel  # noqa: E402
from fka.kernel.seeding import seed_everything  # noqa: E402

SEP = "\n"


def build_records(stream: DenseCorpusStream, variant: int) -> list[tuple[str, str, str]]:
    """``(line, subject, value)`` per fact, mirroring ``DenseCorpusStream._documents`` exactly.

    Roles CANNOT be parsed out of the rendered line. The terse grammar has five delimiter variants
    (``.`` ``|`` space ``:``) and **one of them renders the value BEFORE the subject** — the same
    template `probe_dense_readout` has to exclude. Anything keyed on "text before the first bar" is
    silently wrong on a fifth of the corpus, in a direction that would inflate the subject role.
    So the spans are located by searching for the strings we already know.
    """
    corpus, cfg = stream.corpus, stream.cfg
    n = corpus.n_entities
    qa_period = cfg.qa_period
    surface = stream.surface
    out: list[tuple[str, str, str]] = []
    for r_index, relation in enumerate(corpus.relations):
        period = stream._n_variants[relation]
        offsets = corpus._variant_offset[r_index]
        subjects = stream._subjects[relation]
        values = stream._values[relation]
        base = r_index * n
        for e in range(n):
            if not stream._train_mask[base + e]:
                continue
            phase = int(offsets[e]) + variant
            subject, value = subjects[e], values[e]
            if qa_period and stream.qa_train_entities[e] and phase % qa_period == 0:
                line = surface.qa_line(relation, subject, value)
            else:
                line = surface.statement(relation, subject, value, phase % period)
            out.append((line, subject, value))
    return out


def _spans(line: str, subject: str, value: str) -> np.ndarray:
    """Per-character role: 0 subject, 1 value, 2 other. Spans found, never parsed."""
    roles = np.full(len(line), 2, dtype=np.int8)
    start = line.find(subject)
    if start >= 0:
        roles[start : start + len(subject)] = 0
    # The value may legitimately appear before the subject; take the first occurrence that does
    # not collide with the subject span.
    at = line.find(value)
    while at >= 0 and (roles[at : at + len(value)] == 0).any():
        at = line.find(value, at + 1)
    if at >= 0:
        roles[at : at + len(value)] = 1
    return roles


def tokenise(records, order: np.ndarray, tokenizer) -> tuple[np.ndarray, np.ndarray]:
    ids: list[int] = []
    roles: list[int] = []
    stoi = tokenizer.stoi
    unk = tokenizer.unk_id
    for i in order:
        line, subject, value = records[i]
        line_roles = _spans(line, subject, value)
        for ch, role in zip(line, line_roles):
            ids.append(stoi.get(ch, unk))
            roles.append(int(role))
        ids.append(tokenizer.eos_id)
        roles.append(2)
    return np.asarray(ids, dtype=np.int64), np.asarray(roles, dtype=np.int8)


@torch.no_grad()
def role_loss(model, ids: np.ndarray, roles: np.ndarray, device, block: int, batch: int,
              amp_dtype) -> dict[str, float]:
    """Mean NLL per token, split by role, over contiguous blocks. Teacher-forced throughout."""
    n_blocks = (len(ids) - 1) // block
    totals = np.zeros(3)
    counts = np.zeros(3)
    model.eval()
    for start in range(0, n_blocks, batch):
        stop = min(start + batch, n_blocks)
        idx = np.arange(start, stop) * block
        x = np.stack([ids[i : i + block] for i in idx])
        y = np.stack([ids[i + 1 : i + 1 + block] for i in idx])
        r = np.stack([roles[i + 1 : i + 1 + block] for i in idx])
        xb = torch.from_numpy(x).to(device)
        yb = torch.from_numpy(y).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, _ = model(xb)
        nll = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), yb.reshape(-1), reduction="none"
        ).reshape(yb.shape).cpu().numpy()
        for role in (0, 1, 2):
            mask = r == role
            totals[role] += float(nll[mask].sum())
            counts[role] += int(mask.sum())
    out = {"subject": totals[0] / max(counts[0], 1), "value": totals[1] / max(counts[1], 1),
           "other": totals[2] / max(counts[2], 1)}
    out["all"] = float(totals.sum() / max(counts.sum(), 1))
    return out


def load(path: Path, device):
    """Rebuild from the checkpoint's OWN model_config — never from a recomputed one.

    A probe that reconstructs the architecture from its own arguments is the schedule-horizon
    incident in miniature: it would silently evaluate a different model than the one trained.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg = KernelConfig(**blob["model_config"])
    model = ReasoningKernel(model_cfg)
    model.load_state_dict(blob["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    return model.to(device), blob["train_config"], n_params


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True, help="label=path pairs")
    p.add_argument("--n-entities", type=int, default=16_000)
    p.add_argument("--relations", default="birth_year,birth_city,employer")
    p.add_argument("--n-cities", type=int, default=23)
    p.add_argument("--n-employers", type=int, default=32)
    p.add_argument("--birth-year-range", type=int, nargs=2, default=(1990, 2000))
    p.add_argument("--n-given-names", type=int, default=4096)
    p.add_argument("--n-surnames", type=int, default=4096)
    p.add_argument("--variant", type=int, default=0)
    p.add_argument("--block", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-blocks", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None

    corpus = generate_corpus(
        n_entities=args.n_entities, seed=0, probe_fraction=0.15,
        n_given_names=args.n_given_names, n_surnames=args.n_surnames,
        n_cities=args.n_cities, n_employers=args.n_employers,
        birth_year_range=tuple(args.birth_year_range),
        relations=tuple(args.relations.split(",")),
    )
    tokenizer = syllable_tokenizer()
    stream = DenseCorpusStream(
        corpus, tokenizer,
        DenseDataConfig(exposures=1, surface="syllable", seed=0),
    )
    records = build_records(stream, args.variant)

    trained_order = np.random.default_rng((0, args.variant, 0xD0C)).permutation(len(records))
    fresh_order = np.random.default_rng(0xFEED_BEEF).permutation(len(records))

    # The revision-test rule: assert the manipulation ALTERED the stream before reading any null.
    assert not np.array_equal(trained_order, fresh_order), "fresh ordering did not move"
    keep = args.max_blocks * args.block + 1
    t_ids, t_roles = tokenise(records, trained_order, tokenizer)
    f_ids, f_roles = tokenise(records, fresh_order, tokenizer)
    assert not np.array_equal(t_ids[:keep], f_ids[:keep]), "orderings tokenised identically"
    t_ids, t_roles = t_ids[:keep], t_roles[:keep]
    f_ids, f_roles = f_ids[:keep], f_roles[:keep]

    counts = np.bincount(t_roles, minlength=3)
    print(f"tokens: subject {counts[0]:,}  value {counts[1]:,}  other {counts[2]:,}")
    print(f"{'model':<10} {'ordering':<9} {'subject':>9} {'value':>9} {'other':>9} {'all':>9}")

    results = {}
    import contextlib
    holder = gpu_lock() if device.type == "cuda" else contextlib.nullcontext()
    with holder:
        for spec in args.checkpoints:
            label, path = spec.split("=", 1)
            model, _, n_params = load(Path(path), device)
            print(f"  [{label}] {n_params:,} params from {path}")
            for name, (ids, roles) in (
                ("trained", (t_ids, t_roles)),
                ("fresh", (f_ids, f_roles)),
            ):
                out = role_loss(model, ids, roles, device, args.block, args.batch, amp_dtype)
                results[(label, name)] = out
                print(
                    f"{label:<10} {name:<9} {out['subject']:>9.4f} {out['value']:>9.4f} "
                    f"{out['other']:>9.4f} {out['all']:>9.4f}"
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\n=== ordering sensitivity (fresh minus trained; positive = memorised the order) ===")
    for label in {k[0] for k in results}:
        t, f = results[(label, "trained")], results[(label, "fresh")]
        print(
            f"  {label:<10} subject {f['subject'] - t['subject']:+.4f}   "
            f"value {f['value'] - t['value']:+.4f}   all {f['all'] - t['all']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
