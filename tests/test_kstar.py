"""K* is a reported statistic, so it ships with tests proving it DISCRIMINATES (CLAUDE.md 2b).

The load-bearing cases are the ones that must come out red on a broken instrument:

* a cliff that never crosses the threshold must read **unlocated**, not extrapolated;
* a non-monotone sweep must be **flagged**, because §5.55 routes that to the confounds rather than
  to a scaling law;
* the threshold level must be shown **non-load-bearing**, because that is the entire justification
  for having frozen it blind to the 10M measurement.
"""

from __future__ import annotations

import math

import pytest

from fka.eval.kstar import (
    Arm,
    fit_log,
    fit_power,
    kstar,
    params_for_keys,
    threshold_sensitivity,
)

# The measured 1M anchor (M5 §5.50.3), syllable surface, 941,312 params.
ANCHOR_1M = [
    Arm(16_000, 0.3477),
    Arm(24_000, 0.0634),
    Arm(32_000, 0.0215),
    Arm(48_000, 0.0156),
]


def test_reproduces_the_published_1M_anchor():
    located = kstar(ANCHOR_1M, surface="syllable", n_params=941_312)
    assert located.located
    assert located.value == pytest.approx(19_752, rel=1e-3)
    assert located.bracket == (16_000, 24_000)
    assert located.params_per_key == pytest.approx(47.7, rel=1e-2)


def test_threshold_level_is_not_load_bearing():
    """The published sensitivity table (§5.50.2): every level lands in the same bracket, +-15%."""
    table = threshold_sensitivity(ANCHOR_1M, surface="syllable")
    assert set(table) == {0.10, 0.15, 0.20, 0.25, 0.30}
    for value in table.values():
        assert abs(value - 19_752) / 19_752 < 0.16
    # And it is monotone decreasing in the threshold, which a sign error would invert.
    values = [table[level] for level in sorted(table)]
    assert values == sorted(values, reverse=True)


def test_all_arms_alive_reports_a_LOWER_BOUND_not_a_number():
    """The situation §5.47 was actually in: extrapolating a cliff here is reading the instrument."""
    located = kstar([Arm(8_000, 0.646), Arm(12_000, 0.478), Arm(16_000, 0.348)])
    assert not located.located
    assert located.value is None
    assert located.bound == ">"
    assert "> 16,000" in str(located)


def test_all_arms_dead_reports_an_UPPER_BOUND_not_a_number():
    located = kstar([Arm(32_000, 0.021), Arm(96_000, 0.004), Arm(288_000, 0.002)])
    assert not located.located
    assert located.bound == "<"
    assert "< 32,000" in str(located)


def test_non_monotone_sweep_is_FLAGGED():
    """§5.55's mis-specification clause: more keys reading better means the sweep is not a cliff."""
    located = kstar([Arm(16_000, 0.35), Arm(24_000, 0.06), Arm(32_000, 0.75)])
    assert located.non_monotone == ((24_000, 32_000),)


def test_noise_sized_non_monotonicity_is_NOT_flagged():
    """A wobble inside the MEASURED band is scatter, not a finding — the flag must not fire on it.

    RE-DERIVED 2026-08-03 (M5 §5.88.1, §5.102.2) and deliberately not deleted, so the suite records
    the threshold's history. The original wobble was 0.7 points, chosen to sit under a **guessed**
    1.5-point band. Replication measured the cliff band at 12.8-39.4 points and the non-cliff band
    at ~4.6, so the wobble is re-scaled to sit just inside the cliff band the flag now defaults to.
    """
    located = kstar([Arm(16_000, 0.35), Arm(24_000, 0.021), Arm(32_000, 0.021 + 0.50)])
    assert located.non_monotone == ()
    # ...and the ORIGINAL 0.7-point wobble, which the old constant was built around, still passes.
    assert kstar([Arm(16_000, 0.35), Arm(24_000, 0.021), Arm(32_000, 0.028)]).non_monotone == ()


def test_a_wobble_just_OUTSIDE_the_measured_band_IS_flagged():
    """The re-derived threshold must still discriminate — a band that never fires is decoration."""
    located = kstar([Arm(16_000, 0.35), Arm(24_000, 0.021), Arm(32_000, 0.021 + 0.60)])
    assert located.non_monotone == ((24_000, 32_000),)


def test_the_28000_anomaly_does_NOT_flag_under_the_measured_cliff_band():
    """The audit's call 6, pinned: 12.93 points is ~1 sigma at the cliff and must not fire.

    This is the observation that closed the sequence — the flag was right, its zero point was not.
    """
    located = kstar([Arm(24_000, 0.8414), Arm(28_000, 0.0085), Arm(32_000, 0.1378)])
    assert located.non_monotone == ()


def test_interpolation_is_log_linear_not_linear():
    """Verified red against the linear-in-K instrument: the two disagree by ~4% on real spacing."""
    arms = [Arm(10_000, 0.30), Arm(40_000, 0.10)]
    located = kstar(arms, threshold=0.20)
    frac = (0.30 - 0.20) / (0.30 - 0.10)
    log_linear = math.exp(math.log(10_000) + frac * (math.log(40_000) - math.log(10_000)))
    plain_linear = 10_000 + frac * (40_000 - 10_000)
    assert located.value == pytest.approx(log_linear)
    assert located.value != pytest.approx(plain_linear, rel=0.02)


def test_two_points_cannot_discriminate_a_functional_form():
    """Functional-form restraint, enforced in code: a 2-parameter fit on 2 points has no residual."""
    points = [(941_312, 19_752.0), (10_220_160, 60_000.0)]
    power, log = fit_power(points), fit_log(points)
    assert power.rms == pytest.approx(0.0, abs=1e-6)
    assert log.rms == pytest.approx(0.0, abs=1e-6)
    assert not power.determined and not log.determined


def test_three_points_do_produce_residuals():
    points = [(941_312, 19_752.0), (2_887_296, 33_000.0), (10_220_160, 60_000.0)]
    assert fit_power(points).determined
    assert fit_power(points).rms > 0


def test_inverting_a_fit_recovers_its_own_points():
    points = [(941_312, 19_752.0), (10_220_160, 60_000.0)]
    fit = fit_power(points)
    for n_params, keys in points:
        assert params_for_keys(fit, keys) == pytest.approx(n_params, rel=1e-4)


def test_inverting_a_FLAT_law_reports_unreachable_rather_than_a_huge_number():
    """A flat K* means 2M keys is not reachable at ANY parameter count. Say so, don't return 1e17."""
    fit = fit_power([(941_312, 19_752.0), (10_220_160, 19_752.0)])
    assert params_for_keys(fit, 2_000_000) is None


def test_a_single_arm_is_refused():
    with pytest.raises(ValueError):
        kstar([Arm(16_000, 0.35)])


# --- the summariser's free instrument check ------------------------------------------------------


def test_recovered_chance_matches_the_value_space_and_FIRES_when_it_does_not():
    """Inverting corrected accuracy recovers the chance floor, which the value space already fixes.

    This is a check the run gets for nothing, and it discriminates: a corrected-accuracy denominator
    error or a sign flip moves the recovered floor off the value space it must equal.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from summarise_kstar import recovered_chance

    # Measured 1M anchor, 16,000 keys: three relations, three different value spaces.
    assert recovered_chance(0.4022, 0.3358) == pytest.approx(1 / 10, abs=0.005)   # 10 years
    assert recovered_chance(0.3673, 0.3385) == pytest.approx(1 / 23, abs=0.005)   # 23 cities
    assert recovered_chance(0.3841, 0.3643) == pytest.approx(1 / 32, abs=0.005)   # 32 employers

    # A corrected column computed against the wrong floor no longer matches its value space.
    assert recovered_chance(0.4022, 0.2500) != pytest.approx(1 / 10, abs=0.005)


def test_recovered_chance_is_SILENT_when_corrected_is_clamped_at_zero():
    """A check that fires on its own clamp cries wolf on every dead arm — most of a K* sweep.

    Below-chance accuracy reports corrected = 0, and the inversion then returns the raw accuracy
    rather than the floor. Two real arms (96,000 and 288,000 keys at 10M) hit exactly this and read
    as chance-floor "mismatches" until the clamp was handled.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from summarise_kstar import recovered_chance

    assert recovered_chance(0.0832, 0.0) is None   # below chance for a 10-value space
    assert recovered_chance(0.0258, 0.0) is None
    # ...but a live arm is still checked.
    assert recovered_chance(0.1220, 0.0245) == pytest.approx(1 / 10, abs=0.005)
