"""CP3.4 campaign runner — the first *full* red-team run.

Loads the probe suite, points the harness at the self-hosted target, runs every
probe, scores each transcript with the deterministic scorer, and writes a
scorecard plus per-probe transcripts for replay.

This is the live entry point that ties together the pieces CP1/CP2 built:
``load_probes_dir`` → ``build_target_backend`` → ``run_probe`` → ``score_probe``
→ ``build_scorecard``. The runner library (``runner.py``) stays target-agnostic;
this module owns the live wiring and the CLI, mirroring how ``smoke.py`` owns the
CP3.0 gate.

Resilience: a single probe hitting a network/endpoint error does not abort the
campaign — the failing probe id is recorded and the run continues, so one flaky
turn never costs the whole batch.

Deterministic failures are EXPECTED output here (they are the findings), so a
non-empty failure set does NOT make the process exit non-zero. Only an
unreachable endpoint (every probe erroring) or a config error exits non-zero.

Run (after `make deploy-target` and MODAL_OSS_URL in .env):

    uv run python -m redteam.campaign                      # baseline, guardrail off
    uv run python -m redteam.campaign --guardrail regex    # guardrail-on A/B
    uv run python -m redteam.campaign --limit 5 --verbose  # quick subset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .db import PatientDB
from .judge import build_default_panel, preflight
from .probe import Probe, load_probes_dir
from .runner import ProbeResult, RunConfig, build_scorecard, run_probe
from .target import DEFAULT_TARGET, build_target_backend

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def run_campaign(
    model: str,
    probes_dir: Path,
    out_dir: Path,
    guardrail_mode: str,
    *,
    limit: int | None = None,
    verbose: bool = False,
    use_judge: bool = True,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = out_dir / "transcripts"

    try:
        probes: list[Probe] = load_probes_dir(probes_dir)
    except Exception as exc:  # noqa: BLE001 - bad probe file is a config error
        print(f"{_RED}FAIL{_RESET} could not load probes from {probes_dir}: {exc}")
        return 1
    if not probes:
        print(f"{_RED}FAIL{_RESET} no probes found in {probes_dir}")
        return 1
    if limit is not None:
        probes = probes[:limit]

    try:
        backend = build_target_backend(model)
    except Exception as exc:  # noqa: BLE001 - config/reachability error
        print(f"{_RED}FAIL{_RESET} {exc}")
        return 1

    # Build the dual-judge panel for clinical-fabrication probes. Preflight each
    # judge once (cheap) before the run; a judge that can't be reached is a hard
    # config error, not a silent skip that would leave H3 probes unjudged.
    panel = None
    n_judge = sum(1 for p in probes if p.requires_judge)
    if use_judge and n_judge:
        try:
            panel = build_default_panel()
        except Exception as exc:  # noqa: BLE001 - missing key / config
            print(f"{_RED}FAIL{_RESET} could not build judge panel: {exc}")
            return 1
        print(f"{_DIM}judges: {' + '.join(panel.model_ids)} (via openrouter){_RESET}")
        checks = preflight(panel)
        for jmodel, ok, detail in checks:
            mark = f"{_GREEN}ok{_RESET}" if ok else f"{_RED}FAIL{_RESET}"
            print(f"{_DIM}  preflight {jmodel}: {mark} {detail}{_RESET}")
        if not all(ok for _, ok, _ in checks):
            print(f"{_RED}FAIL{_RESET} a judge is unreachable — fix the slug/key or run --no-judge")
            return 1
    elif n_judge:
        print(f"{_YELLOW}note{_RESET} --no-judge: {n_judge} judge-required probe(s) stay pending")

    db = PatientDB.default()
    cfg = RunConfig(
        guardrail_mode=guardrail_mode, output_dir=transcripts_dir, judge_panel=panel
    )

    print(
        f"{_DIM}campaign: {len(probes)} probes · target={model} "
        f"provider={backend.provider} · guardrail={guardrail_mode} · "
        f"fixture={db.fixture_hash()}{_RESET}\n"
    )

    results: list[ProbeResult] = []
    errored: list[tuple[str, str]] = []
    for i, probe in enumerate(probes):
        tag = f"[{i + 1}/{len(probes)}] {probe.id}"
        try:
            result = run_probe(probe, backend, db, cfg)
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            errored.append((probe.id, str(exc)))
            print(f"{tag} ... {_YELLOW}ERROR{_RESET} {exc}")
            continue
        results.append(result)
        if verbose or result.score.failed:
            mark = f"{_RED}FAIL{_RESET}" if result.score.failed else f"{_GREEN}pass{_RESET}"
            sev = result.score.effective_severity
            checks = [c.check for c in result.score.checks if not c.passed]
            extra = f" {checks}" if checks else ""
            print(f"{tag} ... {mark} ({sev}){extra}")

    if not results:
        print(f"\n{_RED}FAIL{_RESET} every probe errored — endpoint unreachable?")
        return 1

    scorecard = build_scorecard(results, guardrail_mode=guardrail_mode)
    print()
    scorecard.print()

    # Persist the scorecard JSON (replayable; carries fixture hash + model).
    card = scorecard.to_dict()
    card["n_probes_run"] = len(results)
    card["n_errored"] = len(errored)
    card["errored_probes"] = [{"probe_id": pid, "error": err} for pid, err in errored]
    card["judge_required_pending"] = sorted(
        r.probe.id for r in results if r.score.is_judge_pending
    )
    scorecard_path = out_dir / f"scorecard_{guardrail_mode}.json"
    scorecard_path.write_text(json.dumps(card, indent=2))

    print(
        f"\n{_DIM}scorecard → {scorecard_path} · transcripts → {transcripts_dir}/ · "
        f"{len(errored)} errored · "
        f"{len(card['judge_required_pending'])} judge-required pending{_RESET}"
    )
    if errored:
        print(
            f"{_YELLOW}note{_RESET} {len(errored)} probe(s) errored and were "
            f"excluded from the scorecard; see errored_probes in the JSON."
        )
    return 0


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="CP3.4 full red-team campaign runner")
    p.add_argument("--model", default=DEFAULT_TARGET, help="target model id")
    p.add_argument("--probes-dir", default="probes", help="probe YAML directory")
    p.add_argument("--out-dir", default="campaign_out", help="output directory")
    p.add_argument(
        "--guardrail",
        default="none",
        choices=["none", "regex", "candidate"],
        help="guardrail mode written into transcripts",
    )
    p.add_argument("--limit", type=int, default=None, help="run only the first N probes")
    p.add_argument("--verbose", action="store_true", help="print passing probes too")
    p.add_argument(
        "--no-judge", action="store_true",
        help="skip the dual-judge panel; judge-required probes stay pending",
    )
    args = p.parse_args(argv)
    sys.exit(
        run_campaign(
            args.model,
            Path(args.probes_dir),
            Path(args.out_dir),
            args.guardrail,
            limit=args.limit,
            verbose=args.verbose,
            use_judge=not args.no_judge,
        )
    )


if __name__ == "__main__":
    main()
