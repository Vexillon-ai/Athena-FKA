"""Phase 2 — content-addressed sparse routing. **Not started** (M2).

Product-key memory layers giving O(sqrt(N)) top-k lookup over 10^6-10^9 slots, plus hierarchical
and Hopfield-rerank variants (R1-R3). Success criteria: recall@k >= 0.95 versus exact nearest
neighbour at 10^7 slots, log-log lookup-latency slope ~0.5, <10% dead slots after training.
See ``docs/research_plan.md`` §3.
"""
