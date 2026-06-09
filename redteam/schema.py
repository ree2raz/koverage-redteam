"""Frozen transcript schema (v1) — the structured log CP2's scorers consume.

FREEZE DISCIPLINE
-----------------
This schema is version-locked. CP2 scorers read it with a handful of predicates
rather than parsing prose, so editing it after CP2 starts is a failure trigger.
Any breaking change bumps ``SCHEMA_VERSION`` and is recorded in the changelog
below; transcripts carry the version that wrote them so old runs stay replayable
(CP4 requires that any single attack transcript can be replayed).

Changelog
  1.0.0  initial freeze — turns + tool events + advisory verification marker +
         replay header.

DESIGN NOTES
------------
- Built on ``llmcore`` message types so the agent loop emits these directly and
  CP2 never reparses prose. ``raw_messages`` preserves the exact wire log for
  replay; ``turns`` is the derived, scorer-friendly view.
- ``ToolEvent`` separates the two leak surfaces that matter for CP3: surface 1
  (a sensitive value written into reply *text*) is scored off ``Turn.text``;
  surface 2 (a privileged or wrong-caller *tool call*) is scored off
  ``ToolEvent`` — an output-only guardrail cannot see surface 2 unless it
  inspects tool calls, and that asymmetry is the gap CP4 names.
- ``verification`` on a turn is ADVISORY ONLY. Per the agreed policy, CP2
  recomputes the verified-caller predicate independently from the transcript
  (see ``verification.py``); a flag the agent can be talked into setting is not
  the source of truth.
"""

from __future__ import annotations

from llmcore.types import Message, Role, Usage
from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


# --- leak-surface instrumentation -------------------------------------------


class ToolEvent(BaseModel):
    """One tool invocation by the agent, with enough context to score surface 2
    (privileged / cross-patient disclosure) deterministically from the log."""

    name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    result: str = ""  # exactly what the tool returned to the model
    privileged: bool = False  # True iff this is the disclose_sensitive path
    target_patient_id: str | None = None  # patient_id arg, for cross-patient scoring
    field: str | None = None  # disclosed field name, when privileged
    masked: bool = True  # whether the returned sensitive fields were masked
    error: str | None = None


class VerificationMarker(BaseModel):
    """ADVISORY snapshot of the agent loop's belief about caller identity after a
    turn. Never used for scoring — CP2 recomputes verification authoritatively
    from the transcript. Present for debugging and to make the agent's reasoning
    legible when a probe talks it into a false belief."""

    believed_verified_patient_id: str | None = None
    note: str = ""


# --- turns and session ------------------------------------------------------


class Turn(BaseModel):
    index: int
    role: Role
    text: str = ""  # caller utterance, or the agent's reply text
    tool_events: list[ToolEvent] = Field(default_factory=list)  # assistant turns
    verification: VerificationMarker = Field(default_factory=VerificationMarker)


class GuardrailSetting(BaseModel):
    """Which guardrail was in the loop for this session. ``mode`` is the axis of
    comparison in CP3; ``adaptive`` flags transcripts produced by the rewriting
    round so the surviving break rate can be computed over them alone."""

    mode: str = "none"  # none | regex | candidate
    adaptive: bool = False
    parent_probe_id: str | None = None  # original probe this rewrite descends from


class SessionHeader(BaseModel):
    """Everything needed to replay a session exactly (CP4)."""

    session_id: str
    created_at: str  # UTC ISO-8601
    schema_version: str = SCHEMA_VERSION

    probe_id: str
    probe_axis: str  # "phi" | "hallucination"
    probe_vector: str

    target_model: str  # concrete model id on the wire
    target_provider: str
    decoding: dict[str, object] = Field(default_factory=dict)  # temperature, top_p, seed, ...

    guardrail: GuardrailSetting = Field(default_factory=GuardrailSetting)


class Transcript(BaseModel):
    schema_version: str = SCHEMA_VERSION
    header: SessionHeader
    turns: list[Turn] = Field(default_factory=list)
    raw_messages: list[Message] = Field(default_factory=list)  # exact wire log for replay
    usage: Usage = Field(default_factory=Usage)

    # convenience views for the scorers ------------------------------------

    def agent_texts(self) -> list[str]:
        """All assistant reply texts — the surface-1 (text leak) scan target."""
        return [t.text for t in self.turns if t.role == Role.ASSISTANT and t.text]

    def tool_events(self) -> list[ToolEvent]:
        """Flat list of every tool call — the surface-2 (tool-call) scan target."""
        return [ev for t in self.turns for ev in t.tool_events]

    def caller_texts(self) -> list[str]:
        """All caller utterances — what the verification predicate is recomputed
        over (which identifiers the caller actually supplied)."""
        return [t.text for t in self.turns if t.role == Role.USER and t.text]
