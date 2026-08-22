"""Name the op that produced the first non-finite value, and say whether it *could* have.

The 50M divergence is size-correlated on a **Preview-tier ROCm** stack, which puts two very
different explanations on the table:

    finite inputs -> non-finite output, at an op that cannot mathematically produce one
        => DRIVER/KERNEL suspect. No amount of eps-tuning fixes this and trying would hide it.

    finite inputs -> non-finite output, at normalize / exp / div / rsqrt
        => NUMERICS. A real overflow or a 0/0, fixable at the source with an eps floor.

Distinguishing them requires checking *inputs* at the same moment as outputs, which is why this
is a hook-based trap rather than a scan of the loss.

Cost model: the trap is **armed only after** a step has already produced a non-finite value, and
the failing step is then replayed before the optimiser has moved anything. The normal path pays
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

#: Ops that can produce a non-finite value from finite inputs, by their own mathematics.
#: Anything outside this set doing so is a kernel-correctness suspect, not a numerics problem.
CAN_PRODUCE_NONFINITE = (
    "normalize", "softmax", "div", "exp", "rsqrt", "sqrt", "log", "pow",
    "layernorm", "layer_norm", "cross_entropy", "std", "var",
)


def _finite(t) -> bool | None:
    if not isinstance(t, torch.Tensor) or not t.is_floating_point():
        return None
    return bool(torch.isfinite(t).all())


def _all_finite(items) -> bool:
    return all(f for f in (_finite(t) for t in items) if f is not None)


@dataclass
class Culprit:
    """One module whose output went non-finite."""

    name: str
    kind: str
    inputs_finite: bool
    verdict: str

    def __str__(self) -> str:
        return f"{self.name} [{self.kind}] inputs_finite={self.inputs_finite} -> {self.verdict}"


@dataclass
class TrapReport:
    culprits: list[Culprit] = field(default_factory=list)
    params_finite_at_entry: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def first(self) -> Culprit | None:
        return self.culprits[0] if self.culprits else None

    @property
    def verdict(self) -> str:
        if not self.params_finite_at_entry:
            return "corruption-upstream: parameters were already non-finite before this forward"
        c = self.first
        if c is None:
            return "no module produced it — look at the loss terms or the backward pass"
        return c.verdict

    def to_dict(self) -> dict:
        return {
            "params_finite_at_entry": self.params_finite_at_entry,
            "verdict": self.verdict,
            "culprits": [
                {"name": c.name, "kind": c.kind, "inputs_finite": c.inputs_finite,
                 "verdict": c.verdict}
                for c in self.culprits
            ],
            "notes": self.notes,
        }


class NonFiniteTrap:
    """Forward hooks that record every module turning finite inputs into non-finite outputs.

    Modules are reported **innermost-first** by completion order, so the first entry is the
    deepest op that actually produced the value rather than the outermost one carrying it.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.report = TrapReport()
        self._handles: list = []

    def __enter__(self) -> NonFiniteTrap:
        bad = [n for n, p in self.model.named_parameters() if not torch.isfinite(p).all()]
        self.report.params_finite_at_entry = not bad
        if bad:
            self.report.notes.append(
                f"{len(bad)} parameter tensors already non-finite on entry, first: {bad[0]}"
            )
        for name, mod in self.model.named_modules():
            if list(mod.children()):
                continue  # leaves only: a container's output is just its child's
            self._handles.append(mod.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(mod, inputs, output):
            outs = output if isinstance(output, tuple) else (output,)
            if _all_finite(outs):
                return
            ins_finite = _all_finite(inputs)
            kind = type(mod).__name__
            lowered = kind.lower()
            if not ins_finite:
                verdict = "propagating — its input was already non-finite"
            elif any(op in lowered for op in CAN_PRODUCE_NONFINITE):
                verdict = "NUMERICS — this op can produce non-finite from finite input"
            else:
                verdict = (
                    "DRIVER SUSPECT — finite in, non-finite out, at an op that cannot "
                    "mathematically do that"
                )
            self.report.culprits.append(Culprit(name, kind, ins_finite, verdict))

        return hook


def check_tensors(**named) -> dict[str, dict]:
    """Finiteness plus magnitude for named tensors — for loop-level ops hooks cannot see.

    `F.normalize` in the retrieval loss is functional, not a module, so no forward hook reaches
    it. Its input norm is exactly the quantity that would explain a 0/0, so it is measured here.
    """
    out = {}
    for k, t in named.items():
        if not isinstance(t, torch.Tensor):
            continue
        f = t.float()
        out[k] = {
            "finite": bool(torch.isfinite(f).all()),
            "absmax": float(f.abs().max()) if f.numel() else 0.0,
            "min_row_norm": float(f.norm(dim=-1).min()) if f.dim() >= 2 else None,
        }
    return out
