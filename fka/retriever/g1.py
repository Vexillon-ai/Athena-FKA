"""G1 — latent denoiser for the retrieval path (research plan §5.3, M4 §7).

Iterative refinement of a lossy latent toward the clean one:

    x_{k+1} = normalize( x_k + g(x_k, k) )

`k` is the step index, embedded, so the network can behave differently early and late — that is
what makes the **step-count-vs-recovery curve** a real adaptive-compute measurement rather than the
same function applied repeatedly.

Trained on `(store.reconstruct(slot) at sub-knee bits, store.target(slot))` — the real noise
distribution the substrate produces (M4 §7.3). **Never synthetic Gaussian**: M3 §12–13 measured the
failure geometry and it is structured — margins hold and then flip, 60% cliff-like, 0% monotone, and
residual quantisation error is a sum of codebook offsets. A denoiser trained on white noise would
learn the wrong conditional and its recovery number would describe a distribution the substrate
never emits.

Cost enters the shared denominator like everything else (M3 §10.5), so the parameter count is a
budget, not a hyperparameter: `P_max = delta(bits/entity)/n_rel * n_facts / 8` (M4 §7.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class G1Config:
    latent_dim: int = 64
    hidden: int = 768
    n_layers: int = 2
    max_steps: int = 8
    step_embed: int = 32


class G1Denoiser(nn.Module):
    """Residual refinement with a step embedding. Small by design — the budget is the point."""

    def __init__(self, cfg: G1Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.step = nn.Embedding(cfg.max_steps, cfg.step_embed)
        nn.init.normal_(self.step.weight, std=0.02)
        dims = [cfg.latent_dim + cfg.step_embed] + [cfg.hidden] * cfg.n_layers
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(a, b), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], cfg.latent_dim)
        nn.init.zeros_(self.out.weight)  # start as the identity map: step 0 changes nothing
        nn.init.zeros_(self.out.bias)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def refine(self, x: torch.Tensor, k: int) -> torch.Tensor:
        step = self.step(torch.full((x.shape[0],), k, dtype=torch.long, device=x.device))
        return F.normalize(x + self.out(self.net(torch.cat([x, step], dim=-1))), dim=-1)

    def forward(self, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        for k in range(min(steps, self.cfg.max_steps)):
            x = self.refine(x, k)
        return x


class IdentityDenoiser(nn.Module):
    """The VERIFIED-RED NULL (M4 §7.5): must reproduce the no-denoiser ladder EXACTLY.

    If inserting this shifts the ladder by even one rung, the harness is changing the measurement
    and no G1 number is admissible. It is the strongest available check that the with- and
    without-denoiser comparisons are like for like.
    """

    n_params = 0

    def forward(self, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
        return F.normalize(x, dim=-1)


def fit_g1(
    model: G1Denoiser,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    *,
    steps: int = 4000,
    lr: float = 1e-3,
    batch: int = 4096,
    max_refine: int = 4,
    log_every: int = 1000,
) -> list[float]:
    """Cosine loss toward the clean latent, supervised at every refinement depth.

    Supervising all depths — not just the last — is what keeps the step-count curve meaningful:
    a network trained only at depth `n` has no reason to be useful at depth 1, and the curve would
    then measure the training schedule rather than adaptive compute.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = noisy.shape[0]
    losses = []
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), device=noisy.device)
        x, target = F.normalize(noisy[idx], dim=-1), F.normalize(clean[idx], dim=-1)
        loss = torch.zeros((), device=x.device)
        for k in range(max_refine):
            x = model.refine(x, k)
            loss = loss + (1 - F.cosine_similarity(x, target, dim=-1)).mean()
        loss = loss / max_refine
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % log_every == 0 or step == steps - 1:
            print(f"      step {step + 1:>5}/{steps}  cosine loss {losses[-1]:.5f}")
    return losses
