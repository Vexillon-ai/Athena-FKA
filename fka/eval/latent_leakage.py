"""Latent-side leakage diagnostics for D3.

The text-level leakage test cannot see a kernel that bypasses retrieval through the *latent*
channel — there is no text to leak through. These three probes cover that gap, in the order they
should be run and trusted.

**(a) Decode split.** End-to-end accuracy on training-entity subjects versus entity-held-out
subjects. Localises the gap; does not explain it.

**(b) Latent substitution — the decisive one, and the permanent gate.** On *training* entities,
retrieve normally and then swap the final hop's latent for a different value's code before the
readout sees it. Addressing and queries are untouched; only the value on the read channel changes.

    answer follows the substituted latent  ->  the read channel really carries the answer
    answer sticks with the memory's value  ->  the kernel is answering from somewhere else

This is a stronger test than disabling memory, because disabling changes the *distribution* the
kernel sees (a zero latent is off-manifold) and a degraded answer could be blamed on that.
Substitution keeps everything in-distribution and swaps only the content, so a kernel that tracks
the channel has no excuse to fail and one that memorised has no way to pass.

**(c) Subject-path ablation.** Blank the injected subject code. Interpret last and with the
confound stated up front: the subject code *legitimately* feeds hop-1 query formation, so a drop
here is expected even for an honest kernel. Only an *absence* of a drop is strongly informative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fka.data.tokenizer import EOS, CharTokenizer
from fka.eval.capacity import answers_match
from fka.kernel.latent_episodes import PackedEpisodes
from fka.kernel.latent_train import _to_device

#: The gate is on **stick rate**, not follow rate.
#:
#: Originally specified as "follow rate >= 0.90", which was wrong in a way that inverted the
#: verdict (2026-08-02). Follow rate is capped by the codec's own accuracy: a substituted latent
#: the codec cannot decode yields "neither", not "follow". The first run reported FAIL at 68%
#: follow while decode accuracy on the same probes was 69% — i.e. the answer followed the
#: substitute essentially every time it could be decoded at all, which *refutes* the shortcut the
#: gate exists to detect.
#:
#: Stick rate has no such ceiling: a kernel reciting a memorised answer sticks regardless of codec
#: quality. It is the quantity that actually discriminates.
SUBSTITUTION_MAX_STICK = 0.05

#: Reported alongside, conditional on the codec being able to decode at all.
SUBSTITUTION_MIN_CONDITIONAL_FOLLOW = 0.90


def _collect(packed: PackedEpisodes, predictions: dict[int, str]) -> float:
    """Score by id join. This is the only place a prediction meets a gold answer.

    The two assertions are the point of the whole refactor: a positional scorer cannot notice
    that it has paired the wrong rows, whereas a join notices a *missing* or *extra* id
    immediately, and pairs the rest correctly regardless of what order anything arrived in.
    """
    ids = set(packed.episode_id.tolist())
    got = set(predictions)
    if got != ids:
        raise AssertionError(
            f"scoring boundary: {len(ids - got)} episodes never scored, "
            f"{len(got - ids)} predictions belong to no episode in this set"
        )
    correct = 0
    for eid, pred in predictions.items():
        g = packed.gold_for(eid)
        correct += int(answers_match(pred, g.answer, g.relation))
    return correct / len(ids) if ids else 0.0


@torch.no_grad()
def _decode(
    model, memory, packed: PackedEpisodes, tokenizer: CharTokenizer, sl: np.ndarray,
    device, amp_dtype, *, override=None, zero_subject=False, max_tokens=24,
) -> dict[int, str]:
    """Greedy-decode one batch, keyed by episode id, optionally overriding the final latent."""
    batch = _to_device(packed, sl, device)
    subject_code = memory.codebook.entity[batch["subject_ids"]]
    prompt_len = int(np.flatnonzero(packed.answer_mask[sl[0]])[0] + 1)
    ids = batch["tokens"][:, :prompt_len].clone()

    def forward(seq):
        kwargs = dict(
            hard_read=True, override_last_latent=override, zero_subject=zero_subject
        )
        if amp_dtype is None:
            return model(seq, subject_code, batch["subj_pos"], batch["qvec_pos"], memory, **kwargs)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            return model(seq, subject_code, batch["subj_pos"], batch["qvec_pos"], memory, **kwargs)

    logits, _, _ = forward(ids)
    out = [[] for _ in range(len(sl))]
    done = torch.zeros(len(sl), dtype=torch.bool, device=device)
    for _ in range(max_tokens):
        nxt = logits[:, -1, :].float().argmax(dim=-1)
        for i, tok in enumerate(nxt.tolist()):
            if not done[i]:
                out[i].append(tok)
        done |= nxt == tokenizer.eos_id
        if bool(done.all()):
            break
        ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
        logits, _, _ = forward(ids)
    return {
        int(packed.episode_id[row]): (
            tokenizer.decode([t for t in toks if t != tokenizer.eos_id]).split(EOS)[0].strip()
        )
        for row, toks in zip(sl, out, strict=True)
    }


def decode_accuracy(
    model, memory, packed: PackedEpisodes, tokenizer, *,
    device, amp_dtype=None, batch_size: int = 64,
) -> float:
    predictions: dict[int, str] = {}
    for sl in packed.batches_by_answer_start(batch_size):
        predictions.update(_decode(model, memory, packed, tokenizer, sl, device, amp_dtype))
    return _collect(packed, predictions)


@dataclass
class SubstitutionResult:
    """Does the answer track the read channel, or the memorised value?"""

    n: int
    n_followed_substitute: int
    n_kept_original: int
    n_neither: int

    @property
    def follow_rate(self) -> float:
        return self.n_followed_substitute / self.n if self.n else 0.0

    @property
    def stick_rate(self) -> float:
        return self.n_kept_original / self.n if self.n else 0.0

    def conditional_follow_rate(self, decode_accuracy: float | None = None) -> float | None:
        """Follow rate among probes the codec could decode at all — needs a decode baseline."""
        if not decode_accuracy:
            return None
        return min(1.0, self.follow_rate / decode_accuracy)

    @property
    def passes(self) -> bool:
        """Gate on stick rate: a memorising kernel sticks regardless of codec quality."""
        return self.stick_rate <= SUBSTITUTION_MAX_STICK

    def to_dict(self, decode_accuracy: float | None = None) -> dict:
        return {
            "n": self.n,
            "n_followed_substitute": self.n_followed_substitute,
            "n_kept_original": self.n_kept_original,
            "n_neither": self.n_neither,
            "follow_rate": self.follow_rate,
            "stick_rate": self.stick_rate,
            "conditional_follow_rate": self.conditional_follow_rate(decode_accuracy),
            "max_stick_threshold": SUBSTITUTION_MAX_STICK,
            "passes": self.passes,
        }

    def __str__(self) -> str:
        verdict = (
            "read channel is real" if self.passes else "SHORTCUT: answer ignores the channel"
        )
        return (
            f"substitution [{'PASS' if self.passes else 'FAIL'}] stick {self.stick_rate:.1%} "
            f"(max {SUBSTITUTION_MAX_STICK:.0%}), follow {self.follow_rate:.1%}, neither "
            f"{self.n_neither / self.n if self.n else 0:.1%} — {verdict}"
        )


def latent_substitution_test(
    model, memory, packed: PackedEpisodes, tokenizer, *,
    device, amp_dtype=None, seed: int = 0, batch_size: int = 64,
) -> SubstitutionResult:
    """Swap the final retrieved latent for another value's code and see what the answer does.

    Run on **training** entities: the shortcut only exists where the kernel had the opportunity to
    memorise, so testing held-out subjects would prove nothing either way.
    """
    rng = np.random.default_rng(seed)
    followed = kept = neither = 0
    scored: set[int] = set()

    for sl in packed.batches_by_answer_start(batch_size):
        # For each probe pick a different value from the same relation's space. Keyed by episode
        # id, so the substituted value cannot drift away from the row it was chosen for.
        subs = []
        sub_values: dict[int, str] = {}
        for row in sl:
            eid = int(packed.episode_id[row])
            g = packed.gold_for(eid)
            space = memory.codebook.value[g.relation]
            values = memory.corpus.spaces[g.relation].values
            while True:
                j = int(rng.integers(0, len(values)))
                if values[j] != g.answer:
                    break
            subs.append(space[j])
            sub_values[eid] = values[j]
        override = torch.stack(subs).to(device)

        preds = _decode(
            model, memory, packed, tokenizer, sl, device, amp_dtype, override=override
        )
        for eid, pred in preds.items():
            g = packed.gold_for(eid)
            scored.add(eid)
            if answers_match(pred, sub_values[eid], g.relation):
                followed += 1
            elif answers_match(pred, g.answer, g.relation):
                kept += 1
            else:
                neither += 1

    if scored != set(packed.episode_id.tolist()):
        raise AssertionError("substitution test did not score every episode exactly once")

    return SubstitutionResult(
        n=len(packed.tokens),
        n_followed_substitute=followed,
        n_kept_original=kept,
        n_neither=neither,
    )


@dataclass
class CodecDecodeResult:
    """Can the deployed readout turn a value latent into its string, given no other context?"""

    n: int
    n_correct: int
    per_relation: dict[str, float]

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "per_relation": self.per_relation,
        }

    def __str__(self) -> str:
        worst = min(self.per_relation.items(), key=lambda kv: kv[1]) if self.per_relation else None
        tail = f", worst relation {worst[0]} {worst[1]:.1%}" if worst else ""
        return f"codec decode {self.accuracy:.1%} ({self.n_correct}/{self.n}){tail}"


@torch.no_grad()
def codec_decode_accuracy(
    model, memory, tokenizer, *, device, amp_dtype=None, block_size: int = 128,
    n_per_relation: int = 200, seed: int = 0, batch_size: int = 64, max_tokens: int = 24,
) -> CodecDecodeResult:
    """Decode value latents through the deployed head under the constant codec prompt.

    This is the codec's *own* accuracy, isolated from addressing: the value code is injected
    directly with ``override_last_latent``, so retrieval never runs, and the prompt is identical
    for every probe — the only thing that varies across a batch is the latent.

    It lives here rather than in an analysis script on purpose. The last version of this
    measurement was a throwaway that filtered ``<eos>`` out of the generated tokens instead of
    truncating at it, and reported a phantom 1.6% against a true 100% (CLAUDE.md, "the free-running
    phantom"). Ad-hoc analysis code gets the same defects as library code and none of the tests.
    """
    from fka.kernel.latent_train import CodecBatcher  # local: latent_train imports nothing here

    batcher = CodecBatcher(memory, tokenizer, block_size)
    rng = np.random.default_rng(seed)

    # Enumerate deterministically rather than sampling: a relation with few values should be
    # covered exhaustively, not hit at random with replacement.
    probes: list[tuple[str, int]] = []
    for relation in batcher.relations:
        values = memory.corpus.spaces[relation].values
        take = min(n_per_relation, len(values))
        for j in rng.choice(len(values), size=take, replace=False):
            probes.append((relation, int(j)))

    prompt = torch.tensor(batcher._prompt_ids, dtype=torch.long, device=device)
    n_correct = 0
    hits: dict[str, list[int]] = {r: [] for r in batcher.relations}

    for k in range(0, len(probes), batch_size):
        chunk = probes[k : k + batch_size]
        b = len(chunk)
        codes = torch.stack(
            [memory.codebook.value[r][j] for r, j in chunk]
        ).to(device)
        zeros = torch.zeros(b, memory.codebook.dim, device=device, dtype=codes.dtype)
        ids = prompt.unsqueeze(0).expand(b, -1).clone()
        subj_pos = torch.full((b,), batcher.subj_pos, dtype=torch.long, device=device)
        qvec_pos = torch.full((b, 1), batcher.qvec_pos, dtype=torch.long, device=device)

        def forward(seq, zeros=zeros, subj_pos=subj_pos, qvec_pos=qvec_pos, codes=codes):
            kw = dict(hard_read=True, override_last_latent=codes)
            if amp_dtype is None:
                return model(seq, zeros, subj_pos, qvec_pos, memory, **kw)
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                return model(seq, zeros, subj_pos, qvec_pos, memory, **kw)

        logits, _, _ = forward(ids)
        out = [[] for _ in range(b)]
        done = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_tokens):
            nxt = logits[:, -1, :].float().argmax(dim=-1)
            for i, tok in enumerate(nxt.tolist()):
                if not done[i]:
                    out[i].append(tok)
            done |= nxt == tokenizer.eos_id
            if bool(done.all()):
                break
            ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
            logits, _, _ = forward(ids)

        for (relation, j), toks in zip(chunk, out, strict=True):
            # Truncate at <eos>; do not filter it out. That distinction is the phantom above.
            clean = [t for t in toks if t != tokenizer.eos_id]
            pred = tokenizer.decode(clean).split(EOS)[0].strip()
            gold = memory.corpus.spaces[relation].values[j]
            ok = int(answers_match(pred, gold, relation))
            n_correct += ok
            hits[relation].append(ok)

    return CodecDecodeResult(
        n=len(probes),
        n_correct=n_correct,
        per_relation={r: (sum(v) / len(v) if v else 0.0) for r, v in hits.items()},
    )


def subject_ablation_accuracy(
    model, memory, packed: PackedEpisodes, tokenizer, *,
    device, amp_dtype=None, batch_size: int = 64,
) -> float:
    """Accuracy with the subject code blanked. Confounded — see the module docstring."""
    predictions: dict[int, str] = {}
    for sl in packed.batches_by_answer_start(batch_size):
        predictions.update(
            _decode(model, memory, packed, tokenizer, sl, device, amp_dtype, zero_subject=True)
        )
    return _collect(packed, predictions)
