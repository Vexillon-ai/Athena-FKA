"""Phase 1 — the reasoning kernel. **Not started** (M1).

A small (10-300M param) decoder-only LM trained to be deliberately knowledge-poor but competent
at language and at composing facts supplied through a memory interface. Candidate designs D1-D3
and the leakage/composition success criteria are in ``docs/research_plan.md`` §2.

The kernel size cap of 300M params is a hard guardrail, not a default (§9 risk register).

Note that ``scripts/smoke_gpu.py`` contains its own throwaway TinyGPT. That is deliberate: the
environment check must not import this package or it would quietly pre-empt Phase 1's design
decisions before they are made.
"""
