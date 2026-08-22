"""Value spaces: vocabulary determinism and the sample/bits contract.

The contract that matters: ``bits(v)`` must be ``-log2 P(v)`` under the distribution ``sample``
actually draws from. Everything downstream — every bits-per-parameter number in the project —
is wrong if these two drift apart, and nothing else would catch it.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from fka.data import vocab as V

# --- the syllable forge -----------------------------------------------------------------


@pytest.mark.parametrize(
    "builder,capacity",
    [
        (V.given_names, V.GIVEN_NAME_CAPACITY),
        (V.surnames, V.SURNAME_CAPACITY),
        (V.cities, V.CITY_CAPACITY),
        (V.employers, V.EMPLOYER_CAPACITY),
    ],
)
def test_vocabularies_are_distinct_at_full_capacity(builder, capacity):
    """Regression: syllable tables must concatenate injectively.

    `_GIVEN_CODA` once held both "n" and "na" while `_GIVEN_TAIL` held "a", so "n"+"a" and
    "na"+"" both rendered "na" — 384 duplicate given names, which would have made entity names
    non-unique and silently shrunk the value space below the size its bits() advertises.
    """
    words = builder(capacity)
    assert len(set(words)) == capacity, "distinct indices must render distinct words"
    assert all(w and not w.isspace() for w in words)
    with pytest.raises(ValueError, match="at most"):
        builder(capacity + 1)


@pytest.mark.parametrize("builder", [V.given_names, V.surnames, V.cities, V.employers])
def test_vocabularies_are_seed_independent_and_prefix_stable(builder):
    """The i-th word must never move: value spaces are a function of the schema, not the seed.

    Prefix stability also means growing a vocabulary does not renumber the values already in it.
    """
    assert builder(50) == builder(50)
    assert builder(200)[:50] == builder(50)


def test_vocabulary_rejects_empty():
    with pytest.raises(ValueError):
        V.cities(0)


def test_years_render_as_strings():
    assert V.years(1900, 1903) == ("1900", "1901", "1902")
    with pytest.raises(ValueError):
        V.years(1900, 1900)


# --- UniformSpace -----------------------------------------------------------------------


def test_uniform_bits_are_log2_of_size():
    space = V.UniformSpace(V.years(1900, 2000), "birth_year")
    assert space.size == 100
    assert space.bits("1974") == pytest.approx(math.log2(100))
    assert space.entropy == pytest.approx(math.log2(100))
    assert space.probabilities().sum() == pytest.approx(1.0)


def test_uniform_sampling_is_deterministic_and_in_range():
    space = V.UniformSpace(V.cities(64), "birth_city")
    a = space.sample_indices(np.random.default_rng(7), 500)
    b = space.sample_indices(np.random.default_rng(7), 500)
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() < space.size


def test_uniform_sample_returns_a_member():
    space = V.UniformSpace(V.cities(64), "birth_city")
    assert space.sample(np.random.default_rng(0)) in space.values


def test_duplicate_values_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        V.UniformSpace(("a", "b", "a"), "bad")


# --- ZipfSpace --------------------------------------------------------------------------


def test_zipf_bits_match_negative_log_probability():
    space = V.ZipfSpace(V.cities(256), "birth_city", s=1.0)
    p = space.probabilities()
    assert p.sum() == pytest.approx(1.0)
    for rank in (0, 1, 17, 255):
        assert space.bits(space.values[rank]) == pytest.approx(-math.log2(p[rank]))


def test_zipf_head_is_cheaper_than_tail_and_entropy_is_below_uniform():
    space = V.ZipfSpace(V.cities(256), "birth_city", s=1.0)
    assert space.bits(space.values[0]) < space.bits(space.values[-1])
    assert space.entropy < math.log2(space.size)


def test_zipf_empirical_frequencies_track_the_stated_probabilities():
    """Guards the sample/bits contract directly: if sampling ignored `p`, bits would be a lie."""
    space = V.ZipfSpace(V.cities(32), "birth_city", s=1.2)
    draws = space.sample_indices(np.random.default_rng(0), 200_000)
    counts = Counter(draws.tolist())
    empirical = np.array([counts[i] for i in range(space.size)]) / len(draws)
    assert np.allclose(empirical, space.probabilities(), atol=0.005)


def test_zipf_rejects_bad_exponent():
    with pytest.raises(ValueError):
        V.ZipfSpace(V.cities(8), "birth_city", s=0.0)


# --- UniqueNameSpace --------------------------------------------------------------------


def test_names_are_sampled_without_replacement():
    space = V.UniqueNameSpace(V.given_names(256), V.surnames(256))
    idx = space.sample_indices(np.random.default_rng(3), 5000)
    assert len(idx) == 5000
    assert len(set(idx.tolist())) == 5000, "entity names must be unique or probes are ambiguous"


def test_name_bits_are_log2_capacity_and_render_round_trips():
    space = V.UniqueNameSpace(V.given_names(128), V.surnames(64))
    assert space.size == 128 * 64
    assert space.bits("anything") == pytest.approx(math.log2(128 * 64))
    for i in (0, 1, 127, 128, 8191):
        assert space.index_of(space.render(i)) == i


def test_name_sampling_is_deterministic():
    a = V.UniqueNameSpace(V.given_names(64), V.surnames(64)).sample_indices(
        np.random.default_rng(11), 300
    )
    b = V.UniqueNameSpace(V.given_names(64), V.surnames(64)).sample_indices(
        np.random.default_rng(11), 300
    )
    assert np.array_equal(a, b)


def test_cannot_draw_more_names_than_exist():
    space = V.UniqueNameSpace(V.given_names(4), V.surnames(4))
    with pytest.raises(ValueError, match="distinct names"):
        space.sample_indices(np.random.default_rng(0), 17)


# --- CoworkerSetSpace -------------------------------------------------------------------


def test_coworker_bits_are_log2_of_the_subset_count():
    space = V.CoworkerSetSpace(n_entities=1000, k=2)
    assert space.bits() == pytest.approx(math.log2(math.comb(999, 2)))
    assert space.entropy == pytest.approx(space.bits())


def test_coworkers_exclude_self_and_have_no_repeats():
    n, k = 400, 3
    space = V.CoworkerSetSpace(n_entities=n, k=k)
    drawn = space.sample_indices(np.random.default_rng(5), n)
    assert drawn.shape == (n, k)
    assert not (drawn == np.arange(n)[:, None]).any(), "an entity cannot be its own coworker"
    assert (np.diff(np.sort(drawn, axis=1), axis=1) != 0).all(), "coworkers must be distinct"
    assert drawn.min() >= 0 and drawn.max() < n


def test_coworker_space_validates_arguments():
    with pytest.raises(ValueError):
        V.CoworkerSetSpace(n_entities=1, k=1)
    with pytest.raises(ValueError):
        V.CoworkerSetSpace(n_entities=10, k=0)
    with pytest.raises(ValueError):
        V.CoworkerSetSpace(n_entities=10, k=10)
