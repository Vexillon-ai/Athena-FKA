"""Phase 4 — diffusion retrieval. **Not started** (M4).

Conditional denoisers that settle a query into stored attractor basins: latent DDIM (G1), discrete
masked diffusion over RVQ indices (G2), and energy-based Langevin descent (G3). All share
compositional conditioning by score summation, which is what should make never-stored conjunctive
queries retrievable. See ``docs/research_plan.md`` §5.
"""
