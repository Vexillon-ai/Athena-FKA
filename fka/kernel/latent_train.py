"""D3 training, evaluation, and the query-space confusion diagnostic.

Loss = answer cross-entropy + retrieval cross-entropy toward the correct key.

The retrieval term is the fair analogue of D1's setup, not a concession: in D1 the query span's
*text* sits inside the trained region, so D1 receives direct supervision on what to ask. Denying
D3 the equivalent would compare a supervised interface against an unsupervised one and then blame
the interface for the difference.

Evaluation uses **hard argmax** retrieval with no teacher, and greedy decoding.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fka.data.tokenizer import EOS, CharTokenizer
from fka.eval.capacity import answers_match
from fka.kernel.latent_episodes import PackedEpisodes
from fka.kernel.latent_kernel import LatentReasoningKernel
from fka.kernel.latent_memory import OracleLatentMemory

#: The codec prompt. Constant for every value, so it carries zero episode information — the only
#: input that varies across a codec batch is the latent itself.
#:
#: This forwards through the **deployed** path: the same cross-attention modules, the same final
#: layernorm, the same tied head the episode path uses. Nothing is trained that inference does not
#: use. It is implementable without contortion, which answers the pre-registered stop condition:
#: the readout *is* reachable context-free.
#:
#: The honest caveat, recorded before the result: training the head under one *constant* context
#: teaches "decode this latent given the canonical prompt". It does not by itself force invariance
#: across *varying* contexts, which is the defect actually being fixed. If the remeasure lands in
#: quadrant (ii) or (iii), this is the first thing to suspect.
CODEC_PROMPT = "<subj><qvec> A: "


@dataclass
class D3TrainConfig:
    steps: int = 3000
    batch_size: int = 32
    lr: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    retrieval_loss_weight: float = 1.0
    #: Fallback (a). Weight of the codec loss, and how many codec batches per episode batch.
    #: 1:4 mixed (not sequential phases) so the head never gets to specialise and then drift.
    codec_loss_weight: float = 1.0
    codec_batch_ratio: float = 0.25
    amp: bool = True
    seed: int = 0
    log_every: int = 250
    #: Rolling mid-run checkpoints. A long run that dies at step 3,251 leaves nothing to inspect
    #: unless it was writing as it went, and re-reaching the failure costs the whole run again.
    #: Rolling (not cumulative) because these are for post-mortems, not for history.
    checkpoint_every: int = 0
    checkpoint_keep: int = 3

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class D3TrainState:
    step: int = 0
    losses: list[float] = field(default_factory=list)
    answer_losses: list[float] = field(default_factory=list)
    retrieval_losses: list[float] = field(default_factory=list)
    codec_losses: list[float] = field(default_factory=list)
    tokens_per_sec: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "steps": self.step,
            "final_loss": self.losses[-1] if self.losses else None,
            "final_answer_loss": self.answer_losses[-1] if self.answer_losses else None,
            "final_retrieval_loss": self.retrieval_losses[-1] if self.retrieval_losses else None,
            "loss_curve": self.losses[:: max(1, len(self.losses) // 30)],
            "tokens_per_sec": self.tokens_per_sec,
            "seconds": self.seconds,
        }


def _to_device(packed: PackedEpisodes, idx: np.ndarray, device) -> dict:
    return {
        "tokens": torch.from_numpy(packed.tokens[idx]).to(device),
        "subj_pos": torch.from_numpy(packed.subj_pos[idx]).to(device),
        "qvec_pos": torch.from_numpy(packed.qvec_pos[idx]).to(device),
        "answer_mask": torch.from_numpy(packed.answer_mask[idx]).to(device),
        "subject_ids": torch.from_numpy(packed.subject_ids[idx]).to(device),
        "hop_fact_index": torch.from_numpy(packed.hop_fact_index[idx]).to(device),
    }


class CodecBatcher:
    """Builds context-free decode batches: a constant prompt, a varying value latent.

    Fallback (a) from the M1 decision record. The value code is injected with
    ``override_last_latent``, so retrieval is bypassed entirely and the only thing the head can
    condition on that varies is the latent.
    """

    def __init__(self, memory: OracleLatentMemory, tokenizer: CharTokenizer, block_size: int):
        self.memory = memory
        self.tokenizer = tokenizer
        self.relations = sorted(memory.codebook.value)
        prompt_ids = tokenizer.encode(CODEC_PROMPT)
        self.prompt_len = len(prompt_ids)
        self.block_size = block_size
        self.subj_pos = prompt_ids.index(tokenizer.stoi["<subj>"])
        self.qvec_pos = prompt_ids.index(tokenizer.stoi["<qvec>"])
        self._prompt_ids = prompt_ids

    def batch(self, rng: np.random.Generator, batch_size: int, device) -> dict:
        tokens = np.full((batch_size, self.block_size), self.tokenizer.pad_id, dtype=np.int64)
        answer_mask = np.zeros((batch_size, self.block_size - 1), dtype=bool)
        codes = []
        for i in range(batch_size):
            relation = self.relations[int(rng.integers(0, len(self.relations)))]
            values = self.memory.corpus.spaces[relation].values
            j = int(rng.integers(0, len(values)))
            codes.append(self.memory.codebook.value[relation][j])
            ids = self._prompt_ids + self.tokenizer.encode(values[j] + "<eos>")
            tokens[i, : len(ids)] = ids
            answer_mask[i, self.prompt_len - 1 : len(ids) - 1] = True
        return {
            "tokens": torch.from_numpy(tokens).to(device),
            "answer_mask": torch.from_numpy(answer_mask).to(device),
            "subj_pos": torch.full((batch_size,), self.subj_pos, dtype=torch.long, device=device),
            "qvec_pos": torch.full(
                (batch_size, 1), self.qvec_pos, dtype=torch.long, device=device
            ),
            "value_codes": torch.stack(codes).to(device),
        }


def train_d3(
    model: LatentReasoningKernel,
    memory: OracleLatentMemory,
    datasets: Sequence[PackedEpisodes],
    cfg: D3TrainConfig,
    *,
    device: torch.device | str = "cpu",
    progress: bool = True,
    codec: CodecBatcher | None = None,
    checkpoint_dir=None,
    checkpoint_extra: dict | None = None,
) -> D3TrainState:
    """Train over one or more hop-count groups, sampling a group per step."""
    device = torch.device(device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95),
    )
    amp_dtype = torch.bfloat16 if (cfg.amp and device.type == "cuda") else None
    state = D3TrainState()
    codes = memory.codebook.entity

    rolling: list = []

    def write_rolling(step: int) -> None:
        """Keep the last `checkpoint_keep` mid-run checkpoints; delete what falls out."""
        from fka.kernel.checkpoint import save_checkpoint

        path = Path(checkpoint_dir) / f"rolling_step{step:06d}.pt"
        save_checkpoint(
            path, model, model_config=model.cfg, train_config=cfg,
            extra={**(checkpoint_extra or {}), "step": step, "rolling": True},
        )
        rolling.append(path)
        while len(rolling) > max(1, cfg.checkpoint_keep):
            old = rolling.pop(0)
            old.unlink(missing_ok=True)
            old.with_suffix(".meta.json").unlink(missing_ok=True)

    t0 = time.perf_counter()
    total_tokens = 0
    for step in range(cfg.steps):
        lr = cfg.lr * min(1.0, (step + 1) / max(1, cfg.warmup_steps))
        lr *= 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, cfg.steps))))
        for g in opt.param_groups:
            g["lr"] = lr

        packed = datasets[int(rng.integers(0, len(datasets)))]
        idx = rng.integers(0, len(packed.tokens), size=cfg.batch_size)
        batch = _to_device(packed, idx, device)
        subject_code = codes[batch["subject_ids"]]

        x, y = batch["tokens"][:, :-1], batch["tokens"][:, 1:]
        mask = batch["answer_mask"].float()

        # Loop variables bound as defaults: the closure is invoked within this iteration, but
        # binding makes that explicit rather than relying on it.
        def run(x=x, subject_code=subject_code, batch=batch, y=y, mask=mask):
            return model(
                x, subject_code, batch["subj_pos"], batch["qvec_pos"], memory,
                targets=y, loss_mask=mask,
            )

        if amp_dtype is None:
            _, answer_loss, info = run()
        else:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                _, answer_loss, info = run()

        # Retrieval supervision: each emitted query should address its correct key.
        retrieval_loss = torch.zeros((), device=device)
        for hop, q in enumerate(info["queries"]):
            scores = F.normalize(q.float(), dim=-1) @ memory.keys.T / memory.temperature
            retrieval_loss = retrieval_loss + F.cross_entropy(
                scores, batch["hop_fact_index"][:, hop]
            )
        retrieval_loss = retrieval_loss / max(1, len(info["queries"]))

        # Fallback (a): a context-free decode through the deployed head, mixed in rather than
        # run as a separate phase so the head cannot specialise and then drift.
        codec_loss = torch.zeros((), device=device)
        if codec is not None and rng.random() < cfg.codec_batch_ratio:
            cb = codec.batch(rng, cfg.batch_size, device)
            cx, cy = cb["tokens"][:, :-1], cb["tokens"][:, 1:]
            cmask = cb["answer_mask"].float()
            zeros = torch.zeros(
                cfg.batch_size, memory.codebook.dim, device=device, dtype=codes.dtype
            )

            def run_codec(cx=cx, cy=cy, cmask=cmask, cb=cb, zeros=zeros):
                return model(
                    cx, zeros, cb["subj_pos"], cb["qvec_pos"], memory,
                    targets=cy, loss_mask=cmask,
                    override_last_latent=cb["value_codes"],
                )

            if amp_dtype is None:
                _, codec_loss, _ = run_codec()
            else:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    _, codec_loss, _ = run_codec()

        loss = (
            answer_loss
            + cfg.retrieval_loss_weight * retrieval_loss
            + cfg.codec_loss_weight * codec_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        state.step = step + 1
        state.losses.append(float(loss.detach()))
        state.answer_losses.append(float(answer_loss.detach()))
        state.retrieval_losses.append(float(retrieval_loss.detach()))
        state.codec_losses.append(float(codec_loss.detach()))
        total_tokens += x.numel()
        if cfg.checkpoint_every and checkpoint_dir and (step + 1) % cfg.checkpoint_every == 0:
            write_rolling(step + 1)
        if progress and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            print(
                f"   step {step + 1:>5}/{cfg.steps}  loss {state.losses[-1]:.4f}  "
                f"answer {state.answer_losses[-1]:.4f}  retrieval "
                f"{state.retrieval_losses[-1]:.4f}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    state.seconds = time.perf_counter() - t0
    state.tokens_per_sec = total_tokens / state.seconds
    return state


# =======================================================================================
# evaluation
# =======================================================================================


@dataclass
class D3EvalResult:
    n: int
    n_correct: int
    n_routing_failures: int
    n_copy_failures: int
    n_format_failures: int
    retrieval_accuracy: float
    per_hop_retrieval: list[float]
    #: episode_id -> was it exactly right. Kept so a re-score can be compared to an earlier one
    #: episode by episode, rather than only by its total.
    per_episode: dict[int, bool] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "n_routing_failures": self.n_routing_failures,
            "n_copy_failures": self.n_copy_failures,
            "n_format_failures": self.n_format_failures,
            "retrieval_accuracy": self.retrieval_accuracy,
            "per_hop_retrieval": self.per_hop_retrieval,
        }


@torch.no_grad()
def evaluate_d3(
    model: LatentReasoningKernel,
    memory: OracleLatentMemory,
    packed: PackedEpisodes,
    tokenizer: CharTokenizer,
    *,
    device: torch.device | str = "cpu",
    amp_dtype: torch.dtype | None = None,
    max_answer_tokens: int = 24,
    batch_size: int = 64,
) -> tuple[D3EvalResult, dict]:
    """Hard-retrieval evaluation with greedy decoding, batched.

    Failures are attributed the same way as D1: *routing* if any hop addressed the wrong fact,
    *copy* if every retrieval was right but the decoded answer was wrong, *format* if nothing
    decodable came out.

    Batches are grouped by answer-start and results are joined on ``episode_id``. Both are
    load-bearing and this function had neither: it was the second home of the per-batch slicer
    (QUARANTINE.md), and the sibling in ``fka.eval.latent_leakage`` was fixed while this one —
    the evaluator that produced the sweep's headline accuracies — was not.
    """
    device = torch.device(device)
    model.eval()
    codes = memory.codebook.entity
    n = len(packed.tokens)
    eos_id = tokenizer.eos_id

    n_correct = routing = copy_fail = fmt = 0
    hop_hits = np.zeros(packed.n_hops)
    diagnostics = {"margins": [], "correct_sim": [], "best_wrong_sim": []}
    scored: set[int] = set()
    per_episode: dict[int, bool] = {}

    for sl in packed.batches_by_answer_start(batch_size):
        batch = _to_device(packed, sl, device)
        subject_code = codes[batch["subject_ids"]]

        # Answers begin after the prompt; cut the sequence there and decode greedily. Every
        # episode in this group shares the position, which is what makes one slice legitimate.
        prompt_len = int(np.flatnonzero(packed.answer_mask[sl[0]])[0] + 1)
        ids = batch["tokens"][:, :prompt_len].clone()

        def forward(seq, subject_code=subject_code, batch=batch):
            if amp_dtype is None:
                return model(seq, subject_code, batch["subj_pos"], batch["qvec_pos"],
                             memory, hard_read=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                return model(seq, subject_code, batch["subj_pos"], batch["qvec_pos"],
                             memory, hard_read=True)

        logits, _, info = forward(ids)

        # Retrieval correctness, per hop, from the actual argmax address.
        for hop, q in enumerate(info["queries"]):
            got = memory.retrieved_index(q.float())
            want = batch["hop_fact_index"][:, hop]
            hop_hits[hop] += int((got == want).sum())
            sims = F.normalize(q.float(), dim=-1) @ memory.keys.T
            correct = sims.gather(1, want.unsqueeze(1)).squeeze(1)
            masked = sims.scatter(1, want.unsqueeze(1), float("-inf"))
            best_wrong = masked.max(dim=1).values
            diagnostics["correct_sim"].extend(correct.tolist())
            diagnostics["best_wrong_sim"].extend(best_wrong.tolist())
            diagnostics["margins"].extend((correct - best_wrong).tolist())

        retrieval_ok = torch.ones(len(sl), dtype=torch.bool, device=device)
        for hop, q in enumerate(info["queries"]):
            retrieval_ok &= memory.retrieved_index(q.float()) == batch["hop_fact_index"][:, hop]

        generated = [[] for _ in range(len(sl))]
        done = torch.zeros(len(sl), dtype=torch.bool, device=device)
        for _ in range(max_answer_tokens):
            nxt = logits[:, -1, :].float().argmax(dim=-1)
            for i, tok in enumerate(nxt.tolist()):
                if not done[i]:
                    generated[i].append(tok)
            done |= nxt == eos_id
            if bool(done.all()):
                break
            ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
            logits, _, info = forward(ids)

        for i, toks in enumerate(generated):
            eid = int(packed.episode_id[sl[i]])
            gold = packed.gold_for(eid)
            scored.add(eid)
            text = tokenizer.decode([t for t in toks if t != eos_id])
            predicted = text.split(EOS)[0].strip()
            ok = answers_match(predicted, gold.answer, gold.relation)
            per_episode[eid] = ok
            n_correct += int(ok)
            if not ok:
                if not predicted:
                    fmt += 1
                elif bool(retrieval_ok[i]):
                    copy_fail += 1
                else:
                    routing += 1

    if scored != set(packed.episode_id.tolist()):
        raise AssertionError(
            f"scoring boundary: scored {len(scored)} of {n} episodes — grouping dropped rows"
        )

    per_hop = (hop_hits / n).tolist()
    result = D3EvalResult(
        n=n,
        n_correct=n_correct,
        n_routing_failures=routing,
        n_copy_failures=copy_fail,
        n_format_failures=fmt,
        retrieval_accuracy=float(np.mean(per_hop)),
        per_hop_retrieval=per_hop,
        per_episode=per_episode,
    )
    return result, diagnostics

