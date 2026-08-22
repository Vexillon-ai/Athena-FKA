"""Timing utility: the warmup/best-of-k contract every microbenchmark in this project relies on."""

from __future__ import annotations

import pytest

from fka.eval.timing import (
    DEFAULT_REPEATS,
    DEFAULT_WARMUP,
    TimingResult,
    benchmark,
    benchmark_throughput,
    synchronize,
)


def _counter():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    return fn, calls


def test_warmup_calls_are_made_but_not_timed():
    """The whole point of the module: cold calls must run and must not reach the samples."""
    fn, calls = _counter()
    result = benchmark(fn, warmup=4, repeats=6)
    assert calls["n"] == 10, "fn should be called warmup + repeats times"
    assert result.repeats == 6, "only the post-warmup calls are recorded"
    assert result.warmup == 4


def test_defaults_include_a_warmup():
    fn, calls = _counter()
    result = benchmark(fn)
    assert result.warmup == DEFAULT_WARMUP >= 1, "timing a cold call must never be the default"
    assert result.repeats == DEFAULT_REPEATS
    assert calls["n"] == DEFAULT_WARMUP + DEFAULT_REPEATS


def test_zero_warmup_is_allowed_but_explicit():
    fn, calls = _counter()
    benchmark(fn, warmup=0, repeats=3)
    assert calls["n"] == 3


def test_statistics_are_ordered_and_consistent():
    result = TimingResult(name="t", samples=(0.5, 0.1, 0.3, 0.2), warmup=1)
    assert result.best == 0.1
    assert result.median == pytest.approx(0.25)
    assert result.mean == pytest.approx(0.275)
    assert result.best <= result.median <= max(result.samples)
    assert result.repeats == 4


def test_best_is_the_quoted_number_not_the_mean():
    """Interference is one-sided: the minimum estimates intrinsic cost, the mean measures load."""
    result = TimingResult(name="t", samples=(0.10, 0.10, 0.11, 0.90), warmup=1)
    assert result.best == pytest.approx(0.10)
    assert result.mean > 0.25, "a single stalled sample would dominate the mean"


def test_unstable_runs_are_flagged():
    steady = TimingResult(name="t", samples=(0.100, 0.101, 0.099, 0.100), warmup=1)
    noisy = TimingResult(name="t", samples=(0.10, 0.55, 0.12, 0.90), warmup=1)
    assert steady.stable
    assert not noisy.stable
    assert noisy.relative_spread > steady.relative_spread


def test_single_sample_has_zero_stdev_and_is_stable():
    result = TimingResult(name="t", samples=(0.2,), warmup=1)
    assert result.stdev == 0.0
    assert result.relative_spread == 0.0
    assert result.stable


def test_throughput_uses_the_best_sample():
    result = TimingResult(name="t", samples=(0.5, 0.25), warmup=1)
    assert result.throughput(1000) == pytest.approx(4000.0)


def test_benchmark_throughput_records_the_rate_in_metadata():
    fn, _ = _counter()
    result, rate = benchmark_throughput(fn, 4096, name="train", unit="tokens", repeats=3, warmup=1)
    assert rate == pytest.approx(result.throughput(4096))
    assert result.to_dict()["tokens_per_sec"] == pytest.approx(rate)
    assert result.to_dict()["tokens_per_call"] == 4096


def test_to_dict_carries_what_an_experiment_record_needs():
    fn, _ = _counter()
    d = benchmark(fn, name="matmul", repeats=3, warmup=1, metadata={"n": 4096}).to_dict()
    for key in ("name", "warmup", "repeats", "seconds_best", "seconds_median", "stable"):
        assert key in d
    assert d["n"] == 4096


def test_invalid_arguments_are_rejected():
    fn, _ = _counter()
    with pytest.raises(ValueError, match="repeats"):
        benchmark(fn, repeats=0)
    with pytest.raises(ValueError, match="warmup"):
        benchmark(fn, warmup=-1)


def test_synchronize_is_a_noop_without_a_device():
    synchronize(None)  # must not raise on CPU-only or torch-free environments


def test_str_is_readable_and_announces_instability():
    steady = TimingResult(name="matmul", samples=(0.1, 0.1, 0.1), warmup=2)
    noisy = TimingResult(name="matmul", samples=(0.1, 0.9, 0.5), warmup=2)
    assert "matmul" in str(steady) and "UNSTABLE" not in str(steady)
    assert "UNSTABLE" in str(noisy)
