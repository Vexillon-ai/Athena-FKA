"""Phase 3 — the knowledge substrate (attractor codes). **Not started** (M3).

Interchangeable ``KnowledgeStore`` designs sharing one API — ``write(fact_latent) -> slot_ids``,
``read(query, slot_ids) -> latent``, ``delete(slot_ids)`` — covering factorized triple codes (S1),
residual VQ (S2), linear associative writes (S3) and an energy parameterization (S4).
See ``docs/research_plan.md`` §4.

This phase carries the program's go/no-go gate: if no design clears ~3 bits/param at the capacity
knee (M3), we stop and rethink representations rather than rationalize past it. The measurement
itself already exists in ``fka.eval.capacity``.

Per CLAUDE.md, the store should live in ordinary host RAM tensors — on this unified-memory box the
GPU addresses them without a transfer penalty, and that is a design feature, not a compromise.
"""
