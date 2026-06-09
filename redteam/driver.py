"""Hand-driver CLI for the receptionist red-team harness.

Two modes:
  Interactive  — prompts for caller input on stdin, prints every turn and
                 tool event, writes transcript JSON when the session ends.
  Batch        — reads a probe JSON file ({"turns": ["...", "..."]}) and
                 runs every turn automatically, then writes the transcript.

Usage:
  uv run python -m redteam.driver --interactive [--out SESSION_ID.json]
  uv run python -m redteam.driver probe.json   [--out SESSION_ID.json]

Optional flags:
  --model MODEL_ID      target model (default: claude/claude-sonnet-4-6)
  --probe-id ID         written into the transcript header
  --probe-axis AXIS     phi | hallucination  (default: phi)
  --probe-vector VECTOR attack vector label (default: manual)
  --temperature FLOAT   sampling temperature (default: 0.0)
  --out PATH            transcript output path (default: <session_id>.json)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llmcore.config import settings
from llmcore.providers.router import Router

from .db import PatientDB
from .agent import ReceptionistSession


_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"


def _fmt_turn_user(text: str) -> str:
    return f"{_BOLD}{_CYAN}[CALLER]{_RESET} {text}"


def _fmt_turn_agent(text: str) -> str:
    return f"{_BOLD}{_GREEN}[AGENT]{_RESET}  {text}"


def _fmt_tool_event(ev: object) -> str:
    d = ev.model_dump()
    name = d["name"]
    args = json.dumps(d["arguments"], separators=(",", ":"))
    flag = " PRIVILEGED" if d.get("privileged") else ""
    err = f" ERR={d['error']!r}" if d.get("error") else ""
    target = f" pid={d['target_patient_id']!r}" if d.get("target_patient_id") else ""
    result_snippet = (d.get("result") or "")[:80].replace("\n", " ")
    return (
        f"  {_YELLOW}[TOOL]{_RESET}{_DIM}{flag}{err}{target}{_RESET}"
        f" {name}({args})\n"
        f"    {_DIM}→ {result_snippet}{_RESET}"
    )


def _run_session(
    session: ReceptionistSession,
    db: PatientDB,
    turns: list[str] | None,
    out_path: Path,
) -> None:
    """Drive the session from a list of turns (or stdin if turns is None)."""
    try:
        if turns is not None:
            for utterance in turns:
                print(_fmt_turn_user(utterance))
                reply = session.send(utterance)
                # Print tool events from the last assistant turn
                last_assistant = next(
                    (t for t in reversed(session._turns) if t.role.value == "assistant"), None
                )
                if last_assistant:
                    for ev in last_assistant.tool_events:
                        print(_fmt_tool_event(ev))
                print(_fmt_turn_agent(reply))
                print()
        else:
            print(
                f"{_DIM}Interactive session. Type 'quit' or press Ctrl-C to end.{_RESET}\n"
            )
            while True:
                try:
                    utterance = input(_fmt_turn_user("") + " ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if utterance.lower() in ("quit", "exit", "q"):
                    break
                if not utterance:
                    continue
                reply = session.send(utterance)
                last_assistant = next(
                    (t for t in reversed(session._turns) if t.role.value == "assistant"), None
                )
                if last_assistant:
                    for ev in last_assistant.tool_events:
                        print(_fmt_tool_event(ev))
                print(_fmt_turn_agent(reply))
                print()
    finally:
        transcript = session.transcript(fixture_hash=db.fixture_hash())
        out_path.write_text(transcript.model_dump_json(indent=2))
        print(f"{_DIM}Transcript written → {out_path}{_RESET}")
        u = transcript.usage
        print(
            f"{_DIM}Usage: {u.prompt_tokens} prompt + {u.completion_tokens} completion"
            f" = {u.total_tokens} tokens{_RESET}"
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Receptionist red-team hand driver")
    p.add_argument("probe_file", nargs="?", help="Path to probe JSON (batch mode)")
    p.add_argument("--interactive", action="store_true", help="Interactive stdin mode")
    p.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    p.add_argument("--probe-id", default="interactive")
    p.add_argument("--probe-axis", default="phi", choices=["phi", "hallucination"])
    p.add_argument("--probe-vector", default="manual")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out", default=None, help="Output JSON path")
    args = p.parse_args(argv)

    if not args.interactive and not args.probe_file:
        p.error("Specify a probe file or --interactive")

    # Build backend
    router = Router(settings)
    try:
        backend = router.backend_for(args.model)
    except Exception as exc:
        print(f"ERROR: could not build backend for {args.model!r}: {exc}", file=sys.stderr)
        sys.exit(1)

    db = PatientDB.default()
    session = ReceptionistSession(
        backend=backend,
        db=db,
        probe_id=args.probe_id,
        probe_axis=args.probe_axis,
        probe_vector=args.probe_vector,
        decoding={"temperature": args.temperature},
    )

    # Load turns from probe file in batch mode
    turns: list[str] | None = None
    if args.probe_file:
        probe = json.loads(Path(args.probe_file).read_text())
        turns = probe.get("turns", [])
        if not turns:
            print("ERROR: probe file has no 'turns' list", file=sys.stderr)
            sys.exit(1)

    out_path = Path(args.out) if args.out else Path(f"{session._session_id}.json")
    _run_session(session, db, turns, out_path)


if __name__ == "__main__":
    main()
