"""Probe runner — orchestrates session + scorer for a batch of probes.

The runner is the only file that depends on a live backend; everything else
(probe loader, scorer, stats) runs offline. Structuring it this way keeps
CP2's offline test suite fast and keeps the live run isolated.

RECORD/REPLAY CONTRACT
Transcripts written by run_probe() carry:
  - header.target_model / target_provider — exact model id
  - header.decoding — sampling params + fixture_hash
  - header.schema_version — "1.0.0"
  - header.guardrail.mode — none | regex | candidate

Any future scorer call can re-run score_probe() over the saved JSON without
hitting the model again (CP4 replayability requirement).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llmcore.types import ModelBackend

from .agent import ReceptionistSession
from .db import PatientDB
from .probe import Probe
from .schema import GuardrailSetting, Transcript
from .scorer import CONTEXT_CHECKS, ProbeScore, score_probe
from .severity import COST_WEIGHTS
from .stats import AxisStats, compute_axis_stats


# Guardrail modes whose FILTERING is actually wired. Until guardrail.py lands
# (CP3.5), only "none" is real; selecting any other mode must hard-fail rather
# than silently produce a scorecard labelled "guardrail-on" with no filter applied.
WIRED_GUARDRAILS: frozenset[str] = frozenset({"none"})


def validate_guardrail_mode(mode: str) -> None:
    if mode not in WIRED_GUARDRAILS:
        raise NotImplementedError(
            f"guardrail mode {mode!r} is not wired yet — no filtering would be "
            f"applied, so the scorecard label would be false. Only "
            f"{sorted(WIRED_GUARDRAILS)} is valid until guardrail.py lands (CP3.5)."
        )


@dataclass
class RunConfig:
    guardrail_mode: str = "none"    # none | regex | candidate
    adaptive: bool = False
    decoding: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0})
    output_dir: Path | None = None  # if set, transcripts are written here
    judge_panel: Any = None         # judge.JudgePanel | None — judges requires_judge probes


@dataclass
class ProbeResult:
    probe: Probe
    transcript: Transcript
    score: ProbeScore


def run_probe(
    probe: Probe,
    backend: ModelBackend,
    db: PatientDB,
    config: RunConfig | None = None,
) -> ProbeResult:
    """Run a single probe against the backend; return transcript + score."""
    cfg = config or RunConfig()
    validate_guardrail_mode(cfg.guardrail_mode)
    guardrail = GuardrailSetting(
        mode=cfg.guardrail_mode,
        adaptive=cfg.adaptive,
        parent_probe_id=probe.parent_probe_id,
    )
    session = ReceptionistSession(
        backend=backend,
        db=db,
        probe_id=probe.id,
        probe_axis=probe.axis,
        probe_vector=probe.vector,
        guardrail_setting=guardrail,
        decoding=cfg.decoding,
    )
    for turn_text in probe.turns:
        session.send(turn_text)

    transcript = session.transcript(fixture_hash=db.fixture_hash())

    # CP3.3: judge the clinical-fabrication probes BEFORE scoring, so the recorded
    # panel outcome is on the transcript when score_probe reads it (and stays there
    # for replay). No panel / non-judge probe → transcript carries no verdicts.
    if cfg.judge_panel is not None and probe.requires_judge:
        verdicts, outcome = cfg.judge_panel.evaluate(transcript, probe)
        transcript.judgements = verdicts
        transcript.judge_outcome = outcome

    score = score_probe(transcript, probe, db)

    if cfg.output_dir:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{probe.id}__{cfg.guardrail_mode}.json"
        (cfg.output_dir / fname).write_text(transcript.model_dump_json(indent=2))

    return ProbeResult(probe=probe, transcript=transcript, score=score)


def run_all(
    probes: list[Probe],
    backend: ModelBackend,
    db: PatientDB,
    config: RunConfig | None = None,
    *,
    verbose: bool = False,
) -> list[ProbeResult]:
    """Run every probe; return results in probe order."""
    results: list[ProbeResult] = []
    for i, probe in enumerate(probes):
        if verbose:
            status = f"[{i + 1}/{len(probes)}] {probe.id}"
            print(status, end=" ... ", flush=True)
        result = run_probe(probe, backend, db, config)
        if verbose:
            marker = "FAIL" if result.score.failed else "pass"
            print(f"{marker} ({result.score.effective_severity})")
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------


@dataclass
class Scorecard:
    guardrail_mode: str
    run_at: str
    fixture_hash: str
    target_model: str
    phi: AxisStats
    hall: AxisStats
    results: list[ProbeResult]

    def failed_results(self) -> list[ProbeResult]:
        return [r for r in self.results if r.score.failed]

    def observed_results(self) -> list[ProbeResult]:
        """Probes with a tool-context observation but no scored failure."""
        return [r for r in self.results if r.score.observations and not r.score.failed]

    def judged_results(self) -> list[ProbeResult]:
        """Judge-required probes that actually received panel verdicts this run."""
        return [r for r in self.results if r.score.judge_verdicts]

    def summary_lines(self) -> list[str]:
        lines = [
            f"=== Scorecard  guardrail={self.guardrail_mode}  model={self.target_model} ===",
            f"run_at={self.run_at}  fixture={self.fixture_hash}",
            "",
            str(self.phi),
            str(self.hall),
            "",
        ]
        failed = self.failed_results()
        if failed:
            lines.append(f"Failed probes ({len(failed)}):")
            for r in failed:
                checks_failed = [
                    c.check for c in r.score.checks
                    if not c.passed and c.check not in CONTEXT_CHECKS
                ]
                lines.append(
                    f"  {r.probe.id}  {r.score.effective_severity}"
                    f"  checks={checks_failed}"
                )
        else:
            lines.append("No scored (output) failures detected.")

        observed = self.observed_results()
        if observed:
            lines.append("")
            lines.append(
                f"Observed (non-scored) tool-context signals ({len(observed)}) — "
                f"NOT counted in the rate; flagged for the future tool-call gate:"
            )
            for r in observed:
                for c in r.score.observations:
                    lines.append(f"  {r.probe.id}  [{c.check}]  {c.evidence}")

        judged = self.judged_results()
        if judged:
            from .severity import JUDGE_SCORING_ENABLED
            gate = "SCORED" if JUDGE_SCORING_ENABLED else "advisory (not scored until CP4.A κ-gate)"
            lines.append("")
            lines.append(f"Dual-judge panel ({len(judged)} judge-required) — {gate}:")
            for r in judged:
                votes = ", ".join(
                    f"{v.judge_model}={v.verdict}" for v in r.score.judge_verdicts
                )
                lines.append(
                    f"  {r.probe.id}  → {r.score.judge_outcome}  [{votes}]"
                )

        lines.append("")
        lines.append(
            "LIMITATIONS: pilot-scale (≤40 probes); failure rate is OUTPUT-ONLY "
            "(tool-context leaks observed, not scored); H1/H3 not deterministic "
            "(judge-pending probes excluded from the denominator)."
        )
        return lines

    def print(self) -> None:
        print("\n".join(self.summary_lines()))

    def _axis_dict(self, a: AxisStats) -> dict[str, Any]:
        return {
            "n_probes": a.n_probes,
            "n_failed": a.n_failed,
            "n_judge_pending": a.n_judge_pending,
            "raw_asr": round(a.raw_asr, 4),
            "cwASR": round(a.cwASR, 4),
            "ci_lower": round(a.ci_lower, 4),
            "ci_upper": round(a.ci_upper, 4),
            "ci_method": a.ci_method,
        }

    def to_dict(self) -> dict[str, Any]:
        failed = [
            {
                "probe_id": r.probe.id,
                "axis": r.probe.axis,
                "vector": r.probe.vector,
                "severity": r.score.effective_severity,
                "checks_failed": [
                    c.check for c in r.score.checks
                    if not c.passed and c.check not in CONTEXT_CHECKS
                ],
                "evidence": [
                    c.evidence for c in r.score.checks
                    if not c.passed and c.check not in CONTEXT_CHECKS
                ],
                "requires_judge": r.probe.requires_judge,
            }
            for r in self.results
            if r.score.failed
        ]
        observations = [
            {
                "probe_id": r.probe.id,
                "axis": r.probe.axis,
                "vector": r.probe.vector,
                "checks": [c.check for c in r.score.observations],
                "evidence": [c.evidence for c in r.score.observations],
                "note": "tool-context signal; NOT scored under output-only policy",
            }
            for r in self.results
            if r.score.observations
        ]
        from .severity import JUDGE_SCORING_ENABLED
        judge_panel = [
            {
                "probe_id": r.probe.id,
                "axis": r.probe.axis,
                "severity_if_failed": r.probe.severity_if_failed,
                "outcome": r.score.judge_outcome,
                "verdicts": [
                    {
                        "judge_model": v.judge_model,
                        "verdict": v.verdict,
                        "rationale": v.rationale,
                    }
                    for v in r.score.judge_verdicts
                ],
            }
            for r in self.results
            if r.score.judge_verdicts
        ]
        return {
            "guardrail_mode": self.guardrail_mode,
            "run_at": self.run_at,
            "fixture_hash": self.fixture_hash,
            "target_model": self.target_model,
            "scoring_policy": "output-only (tool-context findings observed, not scored)",
            "judge_scoring_enabled": JUDGE_SCORING_ENABLED,
            "phi": self._axis_dict(self.phi),
            "hall": self._axis_dict(self.hall),
            "failed_probes": failed,
            "observations": observations,
            "judge_panel": judge_panel,
            "limitations": [
                "pilot-scale: <= 40 probes, not pricing-grade",
                "failure rate is OUTPUT-ONLY: tool-context / model-context leaks are "
                "observed but not scored (tool-call gate is a later checkpoint)",
                "H1/H3 clinical/policy fabrication is judged by a dual-judge panel "
                "(advisory until the CP4.A κ-gate); judge-pending probes are "
                "excluded from the denominator",
                "no delegate-authorisation model in v1",
            ],
        }


def build_scorecard(results: list[ProbeResult], guardrail_mode: str = "none") -> Scorecard:
    scores = [r.score for r in results]
    target_model = results[0].transcript.header.target_model if results else "unknown"
    fixture_hash = (
        results[0].transcript.header.decoding.get("fixture_hash", "") if results else ""
    )
    return Scorecard(
        guardrail_mode=guardrail_mode,
        run_at=datetime.now(timezone.utc).isoformat(),
        fixture_hash=str(fixture_hash),
        target_model=target_model,
        phi=compute_axis_stats("phi", scores, COST_WEIGHTS),
        hall=compute_axis_stats("hallucination", scores, COST_WEIGHTS),
        results=results,
    )
