"""Locate the first non-finite value in a D3 training run, per size and precision.

    python scripts/diagnose_nan.py --size 50M --steps 400

The 50M stability check went NaN at step 251 while 10M trains cleanly to 15,000 steps under an
identical recipe. This isolates *which* quantity goes non-finite first and *when* — the loss, a
gradient, or a parameter — across the bf16/fp32 and size axes.

Cheapest decisive probe first (CLAUDE.md): a few hundred steps at each setting costs minutes and
distinguishes "bf16 overflow" from "a size-dependent initialisation or LR problem", which no
amount of reasoning about the loss curve can.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fka.data.corpus_gen import CorpusConfig, generate_corpus  # noqa: E402
from fka.data.hardening import entity_split  # noqa: E402
from fka.eval.nonfinite import NonFiniteTrap, check_tensors  # noqa: E402
from fka.kernel.latent_episodes import d3_tokenizer  # noqa: E402
from fka.kernel.latent_kernel import LatentKernelConfig, LatentReasoningKernel  # noqa: E402
from fka.kernel.latent_memory import LatentCodebook, OracleLatentMemory  # noqa: E402
from fka.kernel.latent_train import CodecBatcher, D3TrainConfig, _to_device  # noqa: E402
from fka.kernel.model import KERNEL_SIZES  # noqa: E402
from scripts.run_d3_kernel import build_data  # noqa: E402


def probe(
    size: str, amp: bool, steps: int, seed: int, device, lr: float, clip: float,
    use_codec: bool = False, schedule_steps: int | None = None,
) -> dict:
    """Train briefly, reporting the first step at which anything goes non-finite."""
    corpus = generate_corpus(CorpusConfig(n_entities=2000, seed=seed, n_coworkers=1))
    tokenizer = d3_tokenizer()
    split = entity_split(corpus, fraction=0.2, seed=seed)
    codebook = LatentCodebook.build(corpus, dim=64, seed=seed)
    memory = OracleLatentMemory(corpus, codebook).to(device)
    block_size = 128
    groups, _, _ = build_data(
        corpus, tokenizer, split, [2, 3], memory.fact_index, block_size, seed
    )

    cfg = LatentKernelConfig(
        vocab_size=tokenizer.vocab_size, block_size=block_size,
        latent_dim=64, n_read_heads=1, cross_attn_every=2, **KERNEL_SIZES[size],
    )
    torch.manual_seed(seed)
    model = LatentReasoningKernel(cfg).to(device)
    # The LR schedule must match the run being reproduced, not the probe's own length.
    # cos(pi * step / steps) means a 4,000-step probe and a 15,000-step run are at completely
    # different learning rates by step 3,000 -- which is why two earlier probes "survived"
    # a failure they were never exposed to. Schedule length and probe length are separate knobs.
    tcfg = D3TrainConfig(
        steps=schedule_steps or steps, seed=seed, lr=lr, grad_clip=clip
    )

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tcfg.lr, betas=(0.9, 0.95),
    )
    amp_dtype = torch.bfloat16 if (amp and device.type == "cuda") else None
    rng = np.random.default_rng(seed)
    codes = memory.codebook.entity
    codec = CodecBatcher(memory, tokenizer, block_size) if use_codec else None

    trace = []
    for step in range(steps):
        lr_t = tcfg.lr * min(1.0, (step + 1) / max(1, tcfg.warmup_steps))
        lr_t *= 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, tcfg.steps))))
        for g in opt.param_groups:
            g["lr"] = lr_t

        packed = groups[int(rng.integers(0, len(groups)))]
        idx = rng.integers(0, len(packed.tokens), size=tcfg.batch_size)
        batch = _to_device(packed, idx, device)
        subject_code = codes[batch["subject_ids"]]
        x, y = batch["tokens"][:, :-1], batch["tokens"][:, 1:]
        mask = batch["answer_mask"].float()

        def run(x=x, subject_code=subject_code, batch=batch, y=y, mask=mask):
            return model(x, subject_code, batch["subj_pos"], batch["qvec_pos"], memory,
                         targets=y, loss_mask=mask)

        if amp_dtype is None:
            _, answer_loss, info = run()
        else:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                _, answer_loss, info = run()

        retrieval_loss = torch.zeros((), device=device)
        qmax = 0.0
        for hop, q in enumerate(info["queries"]):
            qmax = max(qmax, float(q.float().abs().max()))
            scores = F.normalize(q.float(), dim=-1) @ memory.keys.T / memory.temperature
            retrieval_loss = retrieval_loss + F.cross_entropy(
                scores, batch["hop_fact_index"][:, hop]
            )
        retrieval_loss = retrieval_loss / max(1, len(info["queries"]))

        codec_loss = torch.zeros((), device=device)
        if codec is not None and rng.random() < tcfg.codec_batch_ratio:
            cb = codec.batch(rng, tcfg.batch_size, device)
            cx, cy = cb["tokens"][:, :-1], cb["tokens"][:, 1:]
            cmask = cb["answer_mask"].float()
            zsub = torch.zeros(
                tcfg.batch_size, memory.codebook.dim, device=device, dtype=codes.dtype
            )

            def run_codec(cx=cx, cy=cy, cmask=cmask, cb=cb, zsub=zsub):
                return model(cx, zsub, cb["subj_pos"], cb["qvec_pos"], memory,
                             targets=cy, loss_mask=cmask,
                             override_last_latent=cb["value_codes"])

            if amp_dtype is None:
                _, codec_loss, _ = run_codec()
            else:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    _, codec_loss, _ = run_codec()

        loss = answer_loss + retrieval_loss + codec_loss

        # --- non-finite triage, BEFORE the optimiser moves anything -----------------------
        if not torch.isfinite(loss):
            print(f"   !! non-finite loss at step {step + 1}; replaying this exact step "
                  f"with the op trap armed")
            with NonFiniteTrap(model) as trap:
                if amp_dtype is None:
                    _, _, info2 = run()
                else:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        _, _, info2 = run()
            rep = trap.report
            print(f"      params finite on entry : {rep.params_finite_at_entry}")
            for c in rep.culprits[:6]:
                print(f"      {c}")
            print(f"      VERDICT: {rep.verdict}")
            # Loop-level ops the hooks cannot see (F.normalize lives in the retrieval term).
            qstats = {}
            for hop, q in enumerate(info2["queries"]):
                qstats[f"hop{hop}_query"] = q.detach()
            probes = check_tensors(**qstats)
            for k, v in probes.items():
                print(f"      {k}: finite={v['finite']} absmax={v['absmax']:.4g} "
                      f"min_row_norm={v['min_row_norm']:.4g}")
            return {
                "diverged_at": step + 1,
                "first_nonfinite": "loss",
                "trap": rep.to_dict(),
                "query_probes": probes,
                "trace": trace,
            }

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip))

        if not math.isfinite(gnorm):
            bad_grads = [n for n, pm in model.named_parameters()
                         if pm.grad is not None and not torch.isfinite(pm.grad).all()]
            print(f"   !! non-finite GRADIENT at step {step + 1} with a finite loss "
                  f"({float(loss):.4f}) — {len(bad_grads)} tensors, first: "
                  f"{bad_grads[0] if bad_grads else '?'}")

            # A finite forward with a non-finite backward is the signature of an op whose
            # DERIVATIVE blows up where its value does not -- F.normalize is the candidate:
            # d(x/||x||) carries 1/||x||, so a collapsing query norm gives a clean forward and
            # an exploding gradient. Attribute by backward-ing each term on its own.
            attribution = {}
            if amp_dtype is None:
                _, a2, info2 = run()
            else:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    _, a2, info2 = run()
            r2 = torch.zeros((), device=device)
            for hop, q in enumerate(info2["queries"]):
                sc = F.normalize(q.float(), dim=-1) @ memory.keys.T / memory.temperature
                r2 = r2 + F.cross_entropy(sc, batch["hop_fact_index"][:, hop])
            r2 = r2 / max(1, len(info2["queries"]))

            for tname, term in (("answer", a2), ("retrieval", r2)):
                if not term.requires_grad:
                    continue
                opt.zero_grad(set_to_none=True)
                term.backward(retain_graph=True)
                nbad = sum(
                    1 for _, pm in model.named_parameters()
                    if pm.grad is not None and not torch.isfinite(pm.grad).all()
                )
                attribution[tname] = {"loss": float(term), "n_nonfinite_grad_tensors": nbad}
                print(f"      backward({tname}) alone: loss {float(term):.4f}, "
                      f"{nbad} non-finite grad tensors")

            qprobe = check_tensors(
                **{f"hop{h}_query": q.detach() for h, q in enumerate(info2["queries"])}
            )
            for k, v in qprobe.items():
                print(f"      {k}: finite={v['finite']} absmax={v['absmax']:.4g} "
                      f"min_row_norm={v['min_row_norm']:.4g}")

            return {
                "diverged_at": step + 1, "first_nonfinite": "gradient",
                "bad_grads": bad_grads[:10], "loss_at_failure": float(loss),
                "attribution": attribution, "query_probes": qprobe,
                "codec_active_this_step": bool(float(codec_loss) != 0.0),
                "trace": trace,
            }

        opt.step()

        bad_param = next(
            (n for n, p in model.named_parameters() if not torch.isfinite(p).all()), None
        )
        rec = {
            "step": step + 1, "lr": lr_t,
            "answer": float(answer_loss), "retrieval": float(retrieval_loss),
            "grad_norm": gnorm, "query_absmax": qmax, "bad_param": bad_param,
            "codec": float(codec_loss),
        }
        if step % 10 == 0 or step < 5:
            trace.append(rec)
        bad = (
            not math.isfinite(rec["answer"]) or not math.isfinite(rec["retrieval"])
            or not math.isfinite(rec["codec"])
            or not math.isfinite(gnorm) or bad_param is not None
        )
        if bad:
            trace.append(rec)
            which = (
                "answer" if not math.isfinite(rec["answer"])
                else "codec" if not math.isfinite(rec["codec"])
                else "retrieval" if not math.isfinite(rec["retrieval"])
                else "grad_norm" if not math.isfinite(gnorm)
                else f"param:{bad_param}"
            )
            print(f"   !! first non-finite at step {step + 1}: {which}  "
                  f"(grad_norm {gnorm:.3g}, |q|max {qmax:.3g}, lr {lr_t:.3g})")
            return {"diverged_at": step + 1, "first_nonfinite": which, "trace": trace}

    peak = max(r["grad_norm"] for r in trace)
    print(f"   ok through {steps} steps  (peak grad_norm {peak:.3g}, "
          f"final answer {trace[-1]['answer']:.4f})")
    return {"diverged_at": None, "peak_grad_norm": peak, "trace": trace}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=400, help="how many steps to actually run")
    p.add_argument("--schedule-steps", type=int, default=None,
                   help="LR-schedule horizon of the run being reproduced; defaults to --steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--configs", nargs="+", default=["50M:bf16", "50M:fp32", "10M:bf16"],
                   help="size:precision[:codec], e.g. 50M:bf16:codec")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    results = {}
    for spec in args.configs:
        parts = spec.split(":")
        size, prec = parts[0], parts[1]
        use_codec = len(parts) > 2 and parts[2] == "codec"
        print(f"\n== {size} {prec}{' +codec' if use_codec else ''}  "
              f"lr {args.lr} clip {args.clip}  ({args.steps} steps)")
        results[f"{spec}|lr{args.lr}|clip{args.clip}|sched{args.schedule_steps or args.steps}"] = (
            probe(size, prec == "bf16", args.steps, args.seed, device, args.lr, args.clip,
                  use_codec, args.schedule_steps)
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
