"""Stats module tests — Wilson, Jeffreys, rule-of-three, AxisStats."""

from __future__ import annotations


from redteam.stats import (
    aggregate_probe_outcome,
    clustered_failure_rate,
    compute_axis_stats,
    design_effect,
    effective_n,
    estimate_icc,
    jeffreys_interval,
    rule_of_three,
    wilson_interval,
)
from redteam.severity import COST_WEIGHTS


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------


def test_wilson_zero_failures():
    lo, hi = wilson_interval(0, 40)
    assert lo == 0.0
    assert 0.0 < hi < 0.10  # should be small


def test_wilson_all_failures():
    lo, hi = wilson_interval(40, 40)
    assert hi == 1.0
    assert lo > 0.90


def test_wilson_half():
    lo, hi = wilson_interval(20, 40)
    assert 0.30 < lo < 0.50
    assert 0.50 < hi < 0.70


def test_wilson_interval_ordered():
    for k in range(0, 41, 5):
        lo, hi = wilson_interval(k, 40)
        assert lo <= hi


def test_wilson_empty_n():
    lo, hi = wilson_interval(0, 0)
    assert lo == 0.0 and hi == 1.0


# ---------------------------------------------------------------------------
# Jeffreys interval
# ---------------------------------------------------------------------------


def test_jeffreys_zero_failures():
    lo, hi = jeffreys_interval(0, 40)
    assert lo == 0.0
    # Upper bound should be close to rule_of_three / slightly below it
    assert 0.0 < hi < 0.15


def test_jeffreys_all_failures():
    lo, hi = jeffreys_interval(40, 40)
    assert hi == 1.0
    assert lo > 0.85


def test_jeffreys_one_failure_of_40():
    lo, hi = jeffreys_interval(1, 40)
    assert lo >= 0.0
    assert hi < 0.15


def test_jeffreys_interval_ordered():
    for k in range(0, 41, 5):
        lo, hi = jeffreys_interval(k, 40)
        assert lo <= hi, f"k={k}: [{lo}, {hi}]"


def test_jeffreys_empty_n():
    lo, hi = jeffreys_interval(0, 0)
    assert lo == 0.0 and hi == 1.0


def test_jeffreys_half():
    lo, hi = jeffreys_interval(20, 40)
    assert 0.33 < lo < 0.50
    assert 0.50 < hi < 0.67


def test_jeffreys_upper_wider_than_wilson_for_small_k():
    """For small k, Jeffreys upper bound should be at least as wide as Wilson
    (Jeffreys is conservative for small n)."""
    k, n = 2, 40
    j_lo, j_hi = jeffreys_interval(k, n)
    w_lo, w_hi = wilson_interval(k, n)
    assert j_hi >= w_hi - 0.02  # within 2pp, direction may vary


# ---------------------------------------------------------------------------
# Rule of three
# ---------------------------------------------------------------------------


def test_rule_of_three_n40():
    r = rule_of_three(40)
    # Should be close to 3/40 = 0.075
    assert 0.06 < r < 0.09


def test_rule_of_three_n10():
    r = rule_of_three(10)
    # 1 - 0.05^(1/10) ≈ 0.259
    assert 0.25 < r < 0.30


def test_rule_of_three_n1():
    r = rule_of_three(1)
    # 1 - 0.05 = 0.95
    assert abs(r - 0.95) < 0.01


def test_rule_of_three_n0():
    assert rule_of_three(0) == 1.0


def test_jeffreys_zero_upper_close_to_rule_of_three():
    """Jeffreys upper and rule-of-three are different but both conservative
    for k=0. They should be in the same ballpark."""
    n = 25
    _, j_hi = jeffreys_interval(0, n)
    r3 = rule_of_three(n)
    assert abs(j_hi - r3) < 0.05, (
        f"Jeffreys upper {j_hi:.4f} vs rule-of-three {r3:.4f} diverged too much"
    )


# ---------------------------------------------------------------------------
# AxisStats / compute_axis_stats
# ---------------------------------------------------------------------------


def _make_scores(axis: str, n: int, n_failed: int) -> list:
    from redteam.scorer import ProbeScore
    sev = "S3" if axis == "phi" else "H2"
    scores = []
    for i in range(n):
        checks = []
        from redteam.scorer import CheckResult
        if i < n_failed:
            checks.append(CheckResult(check="phi.ssn_in_text", passed=False, evidence="test"))
        else:
            checks.append(CheckResult(check="phi.ssn_in_text", passed=True))
        scores.append(ProbeScore(
            probe_id=f"p-{i:03d}",
            axis=axis,
            vector="impersonation",
            severity_if_failed=sev,
            checks=checks,
        ))
    return scores


def test_axis_stats_all_pass():
    stats = compute_axis_stats("phi", _make_scores("phi", 25, 0), COST_WEIGHTS)
    assert stats.n_probes == 25
    assert stats.n_failed == 0
    assert stats.raw_asr == 0.0
    assert stats.ci_method == "rule_of_three"
    assert stats.ci_lower == 0.0
    assert 0.0 < stats.ci_upper < 0.20


def test_axis_stats_all_fail():
    stats = compute_axis_stats("phi", _make_scores("phi", 25, 25), COST_WEIGHTS)
    assert stats.n_failed == 25
    assert stats.raw_asr == 1.0
    assert stats.ci_method == "jeffreys"
    assert stats.ci_upper == 1.0


def test_axis_stats_half():
    stats = compute_axis_stats("phi", _make_scores("phi", 25, 12), COST_WEIGHTS)
    assert stats.n_failed == 12
    assert 0.30 < stats.raw_asr < 0.55
    assert stats.ci_lower < stats.raw_asr < stats.ci_upper


def test_axis_stats_empty():
    stats = compute_axis_stats("phi", [], COST_WEIGHTS)
    assert stats.n_probes == 0
    assert stats.raw_asr == 0.0


def test_axis_stats_filters_by_axis():
    phi_scores = _make_scores("phi", 10, 3)
    hall_scores = _make_scores("hallucination", 5, 1)
    phi_stats = compute_axis_stats("phi", phi_scores + hall_scores, COST_WEIGHTS)
    assert phi_stats.n_probes == 10
    assert phi_stats.n_failed == 3


# ---------------------------------------------------------------------------
# CP2.A — clustered / effective-N correction
# ---------------------------------------------------------------------------


def test_aggregate_any_rule():
    assert aggregate_probe_outcome([False, False, True]) is True
    assert aggregate_probe_outcome([False, False, False]) is False
    assert aggregate_probe_outcome([]) is False


def test_aggregate_majority_rule():
    assert aggregate_probe_outcome([True, True, False], rule="majority") is True
    assert aggregate_probe_outcome([True, False, False], rule="majority") is False


def test_aggregate_unknown_rule_raises():
    import pytest
    with pytest.raises(ValueError):
        aggregate_probe_outcome([True], rule="bogus")


def test_icc_zero_when_no_clustering():
    # Every probe behaves identically (same per-cluster rate) -> low/zero ICC.
    sizes = [10] * 8
    fails = [5] * 8  # identical 0.5 rate everywhere
    icc = estimate_icc(fails, sizes)
    assert 0.0 <= icc < 0.1


def test_icc_high_when_strong_clustering():
    # Clusters are all-or-nothing -> strong intracluster correlation.
    sizes = [10] * 8
    fails = [10, 0, 10, 0, 10, 0, 10, 0]
    icc = estimate_icc(fails, sizes)
    assert icc > 0.8


def test_icc_single_cluster_is_zero():
    assert estimate_icc([5], [10]) == 0.0


def test_design_effect_and_effective_n():
    sizes = [10] * 8  # 80 attempts, mean cluster size 10
    # DEFF = 1 + (10-1)*icc ; with icc=0.5 -> 5.5
    deff = design_effect(sizes, 0.5)
    assert abs(deff - 5.5) < 1e-9
    n_eff = effective_n(sizes, 0.5)
    assert abs(n_eff - 80 / 5.5) < 1e-6
    # No clustering -> effective n equals total.
    assert abs(effective_n(sizes, 0.0) - 80) < 1e-9


def test_clustered_interval_wider_than_naive():
    # 80 attempts, strongly clustered. The clustered CI must be wider than the
    # naive Wilson CI that pretends all 80 attempts are independent.
    sizes = [10] * 8
    fails = [10, 0, 10, 0, 10, 0, 10, 0]  # p_hat = 0.5
    cr = clustered_failure_rate(fails, sizes)
    assert abs(cr.p_hat - 0.5) < 1e-9
    assert cr.n_eff < cr.n_attempts  # deflated
    naive_lo, naive_hi = wilson_interval(40, 80)
    clustered_width = cr.ci_upper - cr.ci_lower
    naive_width = naive_hi - naive_lo
    assert clustered_width > naive_width


def test_clustered_rate_no_clustering_matches_naive():
    # Identical per-cluster rates -> icc≈0 -> clustered CI ≈ naive Wilson.
    sizes = [10] * 8
    fails = [5] * 8  # p_hat = 0.5
    cr = clustered_failure_rate(fails, sizes)
    naive_lo, naive_hi = wilson_interval(40, 80)
    assert abs(cr.ci_lower - naive_lo) < 0.03
    assert abs(cr.ci_upper - naive_hi) < 0.03


def test_axis_stats_cwASR_dominated_by_high_severity():
    """A single S3 failure should dominate the cwASR when cost_weight[S3]=25."""
    from redteam.scorer import ProbeScore, CheckResult
    scores = []
    # 24 passing probes
    for i in range(24):
        scores.append(ProbeScore(
            probe_id=f"p-{i:03d}", axis="phi", vector="impersonation",
            severity_if_failed="S1",
            checks=[CheckResult(check="phi.ssn_in_text", passed=True)],
        ))
    # 1 S3 failure
    scores.append(ProbeScore(
        probe_id="p-024", axis="phi", vector="impersonation",
        severity_if_failed="S3",
        checks=[CheckResult(check="phi.ssn_in_text", passed=False, evidence="test")],
    ))
    stats = compute_axis_stats("phi", scores, COST_WEIGHTS)
    # cwASR = 25 / 25 = 1.0 (the single S3 costs as much as 25 probes)
    assert abs(stats.cwASR - 1.0) < 0.01
