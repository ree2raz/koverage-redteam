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
from .scorer import ProbeScore, score_probe
from .severity import COST_WEIGHTS
from .stats import AxisStats, compute_axis_stats


@dataclass
class RunConfig:
    guardrail_mode: str = "none"    # none | regex | candidate
    adaptive: bool = False
    decoding: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0})
    output_dir: Path | None = None  # if set, transcripts are written here


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
                checks_failed = [c.check for c in r.score.checks if not c.passed]
                lines.append(
                    f"  {r.probe.id}  {r.score.effective_severity}"
                    f"  checks={checks_failed}"
                )
        else:
            lines.append("No failures detected.")
        lines.append("")
        lines.append(
            "LIMITATIONS: pilot-scale (≤40 probes); H1/H3 not deterministic; "
            "no delegate-auth model; output-only guardrails cannot address tool misuse."
        )
        return lines

    def print(self) -> None:
        print("\n".join(self.summary_lines()))

    def to_dict(self) -> dict[str, Any]:
        failed = [
            {
                "probe_id": r.probe.id,
                "axis": r.probe.axis,
                "vector": r.probe.vector,
                "severity": r.score.effective_severity,
                "checks_failed": [c.check for c in r.score.checks if not c.passed],
                "evidence": [c.evidence for c in r.score.checks if not c.passed],
                "requires_judge": r.probe.requires_judge,
            }
            for r in self.results
            if r.score.failed
        ]
        return {
            "guardrail_mode": self.guardrail_mode,
            "run_at": self.run_at,
            "fixture_hash": self.fixture_hash,
            "target_model": self.target_model,
            "phi": {
                "n_probes": self.phi.n_probes,
                "n_failed": self.phi.n_failed,
                "raw_asr": round(self.phi.raw_asr, 4),
                "cwASR": round(self.phi.cwASR, 4),
                "ci_lower": round(self.phi.ci_lower, 4),
                "ci_upper": round(self.phi.ci_upper, 4),
                "ci_method": self.phi.ci_method,
            },
            "hall": {
                "n_probes": self.hall.n_probes,
                "n_failed": self.hall.n_failed,
                "raw_asr": round(self.hall.raw_asr, 4),
                "cwASR": round(self.hall.cwASR, 4),
                "ci_lower": round(self.hall.ci_lower, 4),
                "ci_upper": round(self.hall.ci_upper, 4),
                "ci_method": self.hall.ci_method,
            },
            "failed_probes": failed,
            "limitations": [
                "pilot-scale: <= 40 probes, not pricing-grade",
                "H1/H3 clinical/policy fabrication requires judge scoring",
                "no delegate-authorisation model in v1",
                "output-only guardrails cannot address tool-call misuse (surface 2)",
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
