"""Re-score persisted transcripts with the current scorer — zero compute.

Record/replay is a load-bearing design decision (see README): every transcript
carries its fixture hash, target model, and full wire log, so a scorer change can
be re-applied to past runs offline to re-derive the scorecard without touching
the live target. This module is that replay path.

Use it whenever the deterministic scorer changes (a fixed false positive, a new
check): point it at a ``campaign_out/transcripts/`` directory and it rebuilds the
scorecard from the saved transcripts, matching each one to its probe by id.

    uv run python -m redteam.replay campaign_out/transcripts
    uv run python -m redteam.replay campaign_out/transcripts --out campaign_out

Guard: the fixture hash in every transcript header MUST match the current fixture
(the scorer reads ground truth from the live PatientDB). A mismatch means the
fixture moved under the transcripts and the replay would score against the wrong
ground truth, so we refuse rather than emit a misleading scorecard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .db import PatientDB
from .probe import Probe, load_probes_dir
from .runner import ProbeResult, build_scorecard, validate_guardrail_mode
from .schema import Transcript
from .scorer import score_probe

_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def replay_dir(
    transcripts_dir: Path,
    probes_dir: Path,
    db: PatientDB,
    guardrail_mode: str,
) -> tuple[list[ProbeResult], list[str]]:
    """Load every transcript, match it to its probe by id, re-score. Returns
    (results, warnings). Raises on a fixture-hash mismatch — see module docstring."""
    probes_by_id: dict[str, Probe] = {p.id: p for p in load_probes_dir(probes_dir)}
    fixture_hash = db.fixture_hash()

    results: list[ProbeResult] = []
    warnings: list[str] = []
    for path in sorted(transcripts_dir.glob("*.json")):
        transcript = Transcript.model_validate_json(path.read_text())
        probe_id = transcript.header.probe_id
        probe = probes_by_id.get(probe_id)
        if probe is None:
            warnings.append(f"{path.name}: no probe with id {probe_id!r}; skipped")
            continue
        saved_hash = str(transcript.header.decoding.get("fixture_hash", ""))
        if saved_hash and saved_hash != fixture_hash:
            raise ValueError(
                f"{path.name}: fixture hash {saved_hash} != current {fixture_hash}; "
                f"the fixture moved under these transcripts — re-run, do not replay."
            )
        score = score_probe(transcript, probe, db)
        results.append(ProbeResult(probe=probe, transcript=transcript, score=score))
    return results, warnings


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Re-score saved transcripts (no compute)")
    p.add_argument("transcripts_dir", help="directory of saved transcript JSON files")
    p.add_argument("--probes-dir", default="probes", help="probe YAML directory")
    p.add_argument("--guardrail", default="none", help="guardrail mode label for the card")
    p.add_argument("--out", default=None, help="write scorecard_<mode>.json here")
    args = p.parse_args(argv)

    try:
        validate_guardrail_mode(args.guardrail)
    except NotImplementedError as exc:
        print(f"{_RED}FAIL{_RESET} {exc}")
        sys.exit(1)

    db = PatientDB.default()
    try:
        results, warnings = replay_dir(
            Path(args.transcripts_dir), Path(args.probes_dir), db, args.guardrail
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"{_RED}FAIL{_RESET} {exc}")
        sys.exit(1)
    if not results:
        print(f"{_RED}FAIL{_RESET} no transcripts scored in {args.transcripts_dir}")
        sys.exit(1)
    for w in warnings:
        print(f"{_YELLOW}warn{_RESET} {w}")

    scorecard = build_scorecard(results, guardrail_mode=args.guardrail)
    print(f"{_DIM}replayed {len(results)} transcript(s) · fixture={db.fixture_hash()}{_RESET}\n")
    scorecard.print()

    if args.out:
        card = scorecard.to_dict()
        card["n_probes_run"] = len(results)
        card["replayed"] = True
        card["judge_required_pending"] = sorted(
            r.probe.id for r in results if r.score.is_judge_pending
        )
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"scorecard_{args.guardrail}.json"
        out_path.write_text(json.dumps(card, indent=2))
        print(f"\n{_DIM}scorecard → {out_path}{_RESET}")


if __name__ == "__main__":
    main()
