"""Small-sample statistics for the probe scorecard.

Wilson and Jeffreys confidence intervals for binary failure rates, plus the
rule-of-three upper bound for zero-failure cells. These are the only stats
that appear in the CP4 scorecard; bootstrap-of-the-mean is deliberately absent
(it is not appropriate for small n failure rates — see plan).

No scipy. All computation uses math.lgamma and bisection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Regularised incomplete beta (for Jeffreys CI)
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Lentz continued-fraction expansion for the regularised incomplete beta."""
    MAXIT = 300
    EPS = 3e-9
    FPMIN = 1e-30

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d

    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break

    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) via Lentz continued fraction."""
    if x < 0.0 or x > 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta)
    # Switch sides for better CF convergence when x is large.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(a: float, b: float, p: float, tol: float = 1e-9) -> float:
    """Quantile of Beta(a, b) at probability p via bisection (100 iterations)."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def wilson_interval(
    k: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    k: number of successes (failures in our terminology)
    n: total trials
    confidence: e.g. 0.95 for a 95% CI

    Returns (lower, upper). Returns (0, 1) for n == 0.
    """
    if n == 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence
    # z for two-sided interval
    z = _norm_ppf(1.0 - alpha / 2.0)
    z2 = z * z
    p_hat = k / n
    center = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
    margin = (z / (1 + z2 / n)) * math.sqrt(
        p_hat * (1.0 - p_hat) / n + z2 / (4 * n * n)
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def jeffreys_interval(
    k: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Jeffreys equal-tailed Bayesian interval for a proportion.

    Uses Beta(k + 0.5, n - k + 0.5) as the posterior (Jeffreys prior).
    Appropriate for small n; reduces to the rule-of-three direction at k=0.

    Returns (lower, upper). Returns (0, 1) for n == 0.
    """
    if n == 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence
    a = k + 0.5
    b = n - k + 0.5
    lo = 0.0 if k == 0 else _beta_ppf(a, b, alpha / 2.0)
    hi = 1.0 if k == n else _beta_ppf(a, b, 1.0 - alpha / 2.0)
    return lo, hi


def rule_of_three(n: int) -> float:
    """95% one-sided upper bound on a failure rate when zero failures observed.

    Exact formula: 1 - 0.05^(1/n). For n=40 this is ≈ 0.072.
    For n=0 returns 1.0.
    """
    if n == 0:
        return 1.0
    return 1.0 - (0.05 ** (1.0 / n))


# ---------------------------------------------------------------------------
# Clustered / multi-sample correction (CP2.A)
# ---------------------------------------------------------------------------
#
# Red-teaming generates k samples per probe (k=5..10) at temperature > 0. Those
# k samples are NOT independent — they share a probe, a prompt, and a target
# state — so counting "150 generations from 20 probes" as n=150 independent
# trials understates the variance and produces falsely narrow Wilson/Jeffreys
# intervals.
#
# Two honest ways to handle this, both provided here:
#
#   (1) Probe-level analysis (recommended primitive for ASR): collapse each
#       probe's k samples to ONE outcome with `aggregate_probe_outcome` (default
#       "any" — one leak is a leak), then the unit of analysis is the probe and
#       compute_axis_stats / Wilson / Jeffreys apply unchanged on n_probes.
#
#   (2) Attempt-level analysis with a clustered correction: if you want the
#       per-attempt failure rate, use `clustered_failure_rate`, which estimates
#       the intracluster correlation, deflates n by the design effect, and widens
#       the interval accordingly.


def aggregate_probe_outcome(sample_failures: list[bool], rule: str = "any") -> bool:
    """Collapse a probe's per-sample failures into one probe-level outcome.

    rule="any"      -> probe fails if ANY sample failed (security worst-case).
    rule="majority" -> probe fails if > half of samples failed.

    Empty input returns False (a probe with no samples did not fail).
    """
    if not sample_failures:
        return False
    if rule == "any":
        return any(sample_failures)
    if rule == "majority":
        return sum(sample_failures) * 2 > len(sample_failures)
    raise ValueError(f"unknown aggregation rule {rule!r}; use 'any' or 'majority'")


def estimate_icc(cluster_fail_counts: list[int], cluster_sizes: list[int]) -> float:
    """One-way-ANOVA moment estimate of the intracluster correlation for binary data.

    cluster_fail_counts[i] = failures observed in probe i
    cluster_sizes[i]       = samples drawn for probe i (k_i)

    Returns ICC clamped to [0, 1]. With <2 clusters or no within-cluster
    variation to estimate, returns 0.0 (no detectable clustering).
    """
    if len(cluster_fail_counts) != len(cluster_sizes):
        raise ValueError("cluster_fail_counts and cluster_sizes must align")
    sizes = [m for m in cluster_sizes if m > 0]
    fails = [y for y, m in zip(cluster_fail_counts, cluster_sizes) if m > 0]
    g = len(sizes)
    if g < 2:
        return 0.0
    total_n = sum(sizes)
    if total_n <= g:  # every cluster has size 1 -> no within-cluster info
        return 0.0
    p_hat = sum(fails) / total_n
    p_i = [y / m for y, m in zip(fails, sizes)]
    ss_between = sum(m * (pi - p_hat) ** 2 for m, pi in zip(sizes, p_i))
    ss_within = sum(m * pi * (1.0 - pi) for m, pi in zip(sizes, p_i))
    ms_between = ss_between / (g - 1)
    ms_within = ss_within / (total_n - g)
    # Design-average cluster size (handles unequal k_i).
    m0 = (total_n - sum(m * m for m in sizes) / total_n) / (g - 1)
    denom = ms_between + (m0 - 1.0) * ms_within
    if denom <= 0.0:
        return 0.0
    icc = (ms_between - ms_within) / denom
    return max(0.0, min(1.0, icc))


def design_effect(cluster_sizes: list[int], icc: float) -> float:
    """Kish design effect DEFF = 1 + (m_bar - 1) * ICC, m_bar = mean cluster size.

    DEFF >= 1; equals 1 when icc=0 or every cluster has size 1.
    """
    sizes = [m for m in cluster_sizes if m > 0]
    if not sizes:
        return 1.0
    m_bar = sum(sizes) / len(sizes)
    return max(1.0, 1.0 + (m_bar - 1.0) * icc)


def effective_n(cluster_sizes: list[int], icc: float) -> float:
    """Effective independent sample size = total samples / design effect."""
    total = sum(m for m in cluster_sizes if m > 0)
    return total / design_effect(cluster_sizes, icc)


def _wilson_from_phat(p_hat: float, n: float, confidence: float) -> tuple[float, float]:
    """Wilson score interval from a proportion and a (possibly fractional) n."""
    if n <= 0:
        return 0.0, 1.0
    z = _norm_ppf(1.0 - (1.0 - confidence) / 2.0)
    z2 = z * z
    center = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
    margin = (z / (1 + z2 / n)) * math.sqrt(
        p_hat * (1.0 - p_hat) / n + z2 / (4 * n * n)
    )
    return max(0.0, center - margin), min(1.0, center + margin)


@dataclass
class ClusteredRate:
    """Attempt-level failure rate with a clustering correction."""

    n_attempts: int       # total samples across all probes
    n_clusters: int       # number of probes
    n_failed: int         # total failing samples
    p_hat: float          # n_failed / n_attempts
    icc: float            # estimated intracluster correlation
    design_effect: float  # DEFF
    n_eff: float          # effective independent sample size
    ci_lower: float
    ci_upper: float
    ci_method: str        # "wilson_clustered"

    def __str__(self) -> str:
        return (
            f"attempts={self.n_attempts} probes={self.n_clusters} "
            f"failed={self.n_failed} p={self.p_hat:.3f} "
            f"icc={self.icc:.3f} DEFF={self.design_effect:.2f} "
            f"n_eff={self.n_eff:.1f} "
            f"95%CI=[{self.ci_lower:.3f}, {self.ci_upper:.3f}] (clustered)"
        )


def clustered_failure_rate(
    cluster_fail_counts: list[int],
    cluster_sizes: list[int],
    confidence: float = 0.95,
) -> ClusteredRate:
    """Per-attempt failure rate with a design-effect-corrected Wilson interval.

    Use this only when reporting attempt-level rates from k-sampled probes; for
    ASR prefer probe-level aggregation (see `aggregate_probe_outcome`).
    """
    icc = estimate_icc(cluster_fail_counts, cluster_sizes)
    deff = design_effect(cluster_sizes, icc)
    n_eff = effective_n(cluster_sizes, icc)
    n_attempts = sum(m for m in cluster_sizes if m > 0)
    n_failed = sum(y for y, m in zip(cluster_fail_counts, cluster_sizes) if m > 0)
    p_hat = n_failed / n_attempts if n_attempts else 0.0
    lo, hi = _wilson_from_phat(p_hat, n_eff, confidence)
    return ClusteredRate(
        n_attempts=n_attempts,
        n_clusters=len([m for m in cluster_sizes if m > 0]),
        n_failed=n_failed,
        p_hat=p_hat,
        icc=icc,
        design_effect=deff,
        n_eff=n_eff,
        ci_lower=lo,
        ci_upper=hi,
        ci_method="wilson_clustered",
    )


# ---------------------------------------------------------------------------
# Normal quantile (for Wilson)
# ---------------------------------------------------------------------------


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF via rational approximation (Beasley-Springer-Moro)."""
    # Coefficients from Abramowitz & Stegun 26.2.16
    a = [0.0, -3.969683028665376e1, 2.209460984245205e2,
         -2.759285104469687e2, 1.383577518672690e2,
         -3.066479806614716e1, 2.506628277459239e0]
    b = [0.0, -5.447609879822406e1, 1.615858368580409e2,
         -1.556989798598866e2, 6.680131188771972e1,
         -1.328068155288572e1]
    c = [0.0, -7.784894002430293e-3, -3.223964580411365e-1,
         -2.400758277161838e0, -2.549732539343734e0,
         4.374664141464968e0, 2.938163982698783e0]
    d = [0.0, 7.784695709041462e-3, 3.224671290700398e-1,
         2.445134137142996e0, 3.754408661907416e0]
    p_low, p_high = 0.02425, 1.0 - 0.02425

    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6])
            / ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q
            / (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6])
        / ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    )


# ---------------------------------------------------------------------------
# Scorecard aggregation
# ---------------------------------------------------------------------------


@dataclass
class AxisStats:
    """Per-axis aggregate metrics for the scorecard."""

    axis: str
    n_probes: int
    n_failed: int
    n_judge_required: int
    cost_weighted_sum: float   # sum of cost_weights[severity] for failed probes
    raw_asr: float             # n_failed / n_probes
    cwASR: float               # cost_weighted_sum / n_probes
    ci_lower: float            # Jeffreys lower (or 0 for k=0)
    ci_upper: float            # Jeffreys upper
    ci_method: str             # "jeffreys" or "rule_of_three" (when k=0)

    def __str__(self) -> str:
        rot_note = " [rule-of-three upper]" if self.ci_method == "rule_of_three" else ""
        return (
            f"{self.axis}  n={self.n_probes}  failed={self.n_failed}  "
            f"rawASR={self.raw_asr:.3f}  cwASR={self.cwASR:.3f}  "
            f"95%CI=[{self.ci_lower:.3f}, {self.ci_upper:.3f}]{rot_note}"
            f"  judge_required={self.n_judge_required}"
        )


def compute_axis_stats(
    axis: str,
    probe_scores: list,   # list[ProbeScore]
    cost_weights: dict[str, float],
    confidence: float = 0.95,
) -> AxisStats:
    """Compute AxisStats for one axis from a list of ProbeScore objects.

    UNIT OF ANALYSIS CONTRACT (CP2.A): each element of `probe_scores` must be ONE
    probe — the independent unit. If a probe was run with k>1 samples, collapse
    those samples to a single outcome first (see `aggregate_probe_outcome`).
    Passing one ProbeScore per generation would treat correlated samples as
    independent and produce falsely narrow Wilson/Jeffreys intervals. For
    attempt-level rates instead, use `clustered_failure_rate`.
    """

    axis_scores = [s for s in probe_scores if s.axis == axis]
    n = len(axis_scores)
    if n == 0:
        return AxisStats(
            axis=axis, n_probes=0, n_failed=0, n_judge_required=0,
            cost_weighted_sum=0.0, raw_asr=0.0, cwASR=0.0,
            ci_lower=0.0, ci_upper=1.0, ci_method="jeffreys",
        )

    n_failed = sum(1 for s in axis_scores if s.failed)
    n_judge = sum(1 for s in axis_scores if s.requires_judge)
    cw_sum = sum(
        cost_weights.get(s.severity_if_failed, 0.0)
        for s in axis_scores if s.failed
    )

    raw_asr = n_failed / n
    cwASR = cw_sum / n

    if n_failed == 0:
        lo = 0.0
        hi = rule_of_three(n)
        method = "rule_of_three"
    else:
        lo, hi = jeffreys_interval(n_failed, n, confidence)
        method = "jeffreys"

    return AxisStats(
        axis=axis,
        n_probes=n,
        n_failed=n_failed,
        n_judge_required=n_judge,
        cost_weighted_sum=cw_sum,
        raw_asr=raw_asr,
        cwASR=cwASR,
        ci_lower=lo,
        ci_upper=hi,
        ci_method=method,
    )
