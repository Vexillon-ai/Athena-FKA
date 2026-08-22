"""The M3 §10.5 storage metric, and the three gaming counterexamples it exists to block.

The gaming tests are the point. Each computes what a store would report if it were allowed the
exemption, and asserts the honest accounting gives a materially worse number — so the rule is
demonstrably load-bearing rather than decorative.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fka.eval.accounting import (
    DENSE_BASELINE,
    GATE_PASS,
    GATE_TARGET,
    StorageAccount,
    account_for,
    pointer_bits,
)
from fka.store.base import IdentityStore
from fka.store.s1_factorized import S1Config, S1FactorizedStore

DIM = 64


def _content(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, DIM, generator=g), dim=-1)


def test_gates_are_the_bits_per_param_gates_restated_under_int8():
    """3/8, 4/8 and 2/8 — the restatement is exact, not approximate (M3 §2.1)."""
    assert pytest.approx(3 / 8) == GATE_PASS
    assert pytest.approx(4 / 8) == GATE_TARGET
    assert pytest.approx(2 / 8) == DENSE_BASELINE


def test_verdict_names_the_under_delivery_band():
    """A design between pass and target passes AND under-delivers; it must be reported that way."""
    def verdict(b):
        return StorageAccount(n_facts=1, per_fact_bits=1 / b, shared_params=0,
                              knowledge_bits=1.0).verdict
    assert verdict(0.30) == "FAIL"
    assert verdict(0.40) == "PASS BUT UNDER-DELIVERS"
    assert verdict(0.60) == "PASS (meets target)"


def test_both_widths_are_reported_never_just_the_flattering_one():
    acc = StorageAccount(n_facts=100, per_fact_bits=32, shared_params=1000, knowledge_bits=5000)
    d = acc.to_dict()
    assert d["bits_per_bit_int8"] > d["bits_per_bit_fp32"], "int8 is the flattering direction"
    assert {"bits_per_bit_int8", "bits_per_bit_fp32"} <= set(d)


# -- the three gaming counterexamples ------------------------------------------------------


def _s1(**kw) -> S1FactorizedStore:
    cfg = S1Config(latent_dim=DIM, n_stages=4, codebook_size=256, residual_dim=4,
                   residual_bits=8, **kw)
    s = S1FactorizedStore(cfg)
    s.write(_content(1000))
    return s


def test_gaming_1_free_pointers_would_inflate_the_number():
    """Calling pointers "data, not parameters" is the exemption that lets a lookup table win."""
    store = _s1()
    honest = account_for(store, n_facts=1000, knowledge_bits=20000, key_path_params=0)

    cm = store.cost_model()
    without_pointers = cm["per_fact_storage_bits"] - cm["per_fact_detail"]["pointer_bits"]
    gamed = StorageAccount(n_facts=1000, per_fact_bits=without_pointers,
                           shared_params=cm["shared_parameters"], knowledge_bits=20000)
    assert gamed.headline > honest.headline
    assert cm["per_fact_detail"]["pointer_bits"] > 0, "the exemption must be worth something"


def test_gaming_2_a_hidden_dictionary_would_inflate_the_number():
    """Codebooks are not free infrastructure; in the limit they ARE the store."""
    store = _s1()
    honest = account_for(store, n_facts=1000, knowledge_bits=20000)
    gamed = StorageAccount(n_facts=1000, per_fact_bits=store.cost_model()["per_fact_storage_bits"],
                           shared_params=0, knowledge_bits=20000)
    assert gamed.headline > honest.headline
    assert store.cost_model()["shared_parameters"] > 0


def test_gaming_3_a_memorising_encoder_must_be_counted():
    """M2 §10.3.4 showed a learned map on this path CAN memorise, so this is not hypothetical."""
    store = _s1()
    without = account_for(store, n_facts=1000, knowledge_bits=20000, key_path_params=0)
    with_f = account_for(store, n_facts=1000, knowledge_bits=20000, key_path_params=1_000_000)
    assert with_f.headline < without.headline
    assert with_f.breakdown["key_path_parameters"] == 1_000_000


def test_a_store_that_hides_its_costs_is_refused():
    class Sneaky(IdentityStore):
        def cost_model(self):
            return {"design": "sneaky", "write_touches": "nothing"}

    s = Sneaky()
    s.write(_content(10))
    with pytest.raises(ValueError, match="per_fact_storage_bits"):
        account_for(s, n_facts=10, knowledge_bits=100)


# -- identical accounting across designs ---------------------------------------------------


def test_the_lossless_store_pays_full_price_under_the_same_rule():
    """IdentityStore is not exempt: 64 fp32 floats per fact is 2,048 bits, and it shows."""
    s = IdentityStore()
    s.write(_content(100))
    assert s.cost_model()["per_fact_storage_bits"] == DIM * 32
    acc = account_for(s, n_facts=100, knowledge_bits=100 * 20)
    assert acc.headline < DENSE_BASELINE, "a lossless store should not beat a dense transformer"


def test_pointer_bits_is_an_entropy_bound_not_a_byte_count():
    assert pointer_bits(256) == 8.0
    assert pointer_bits(100) == pytest.approx(6.6438, abs=1e-3)
