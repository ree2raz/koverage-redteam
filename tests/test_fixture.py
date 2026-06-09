"""CP1 fixture tests — DB seed, tools, and agent loop. No network calls.

Covers the exit conditions for Checkpoints 1.2 and 1.3:
  - DB seed is deterministic (same hash twice), 50 patients, 5 canaries.
  - No duplicate MRNs, phones, or patient IDs.
  - Tool masked paths never leak full SSN or full card.
  - disclose_sensitive returns unmasked value and logs privileged=True/masked=False.
  - Cross-patient target_patient_id is correctly recorded.
  - Unknown patient lookups fail closed (no data returned).
  - Appointment fixture has correct structure (canary: no future; normal: has one).
  - Agent loop emits ToolEvents in turns and transcript validates against schema.
"""

from __future__ import annotations

import json

from redteam.canary import CANARY_PATIENTS
from redteam.db import (
    ABSENT_PATIENT_ID,
    PatientDB,
)
from redteam.tools import ReceptionistTools
from redteam.schema import SCHEMA_VERSION, Transcript
from redteam.agent import ReceptionistSession
from redteam.canary import canary_match_strings
from llmcore.types import Role, ModelResponse, Usage, ToolCall, Message
from typing import Any


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _db() -> PatientDB:
    return PatientDB.default()


class _StubBackend:
    """Minimal ModelBackend that returns a canned script of responses.

    Each entry in ``script`` is either a plain str (text-only response) or a
    list of ToolCall dicts followed by a final str (tool-call round then text).
    """

    provider = "stub"
    model = "stub-model"

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self._idx = 0

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> ModelResponse:
        if self._idx >= len(self._script):
            return ModelResponse(text="(stub exhausted)", usage=Usage())
        entry = self._script[self._idx]
        self._idx += 1
        if isinstance(entry, str):
            return ModelResponse(text=entry, usage=Usage(prompt_tokens=10, completion_tokens=5))
        # list: [ToolCall, ..., str] — tool-call round
        calls = [ToolCall(id=f"c{i}", name=tc["name"], arguments=tc["args"])
                 for i, tc in enumerate(entry)]
        return ModelResponse(tool_calls=calls, text="", usage=Usage(prompt_tokens=10, completion_tokens=5))

    def stream(self, messages: list[Message], **params: Any):  # type: ignore[override]
        raise NotImplementedError


# ---------------------------------------------------------------------------
# DB seed: determinism, counts, uniqueness
# ---------------------------------------------------------------------------


def test_fixture_is_deterministic():
    db1 = PatientDB.default()
    db2 = PatientDB.default()
    assert db1.fixture_hash() == db2.fixture_hash()


def test_fixture_has_fifty_patients():
    assert _db().patient_count() == 50


def test_fixture_has_five_canary_rows():
    canary_ids = {c.patient_id for c in CANARY_PATIENTS}
    found = [p for p in _db().all_patients() if p.patient_id in canary_ids]
    assert len(found) == 5


def test_no_duplicate_mrns():
    mrns = [p.mrn for p in _db().all_patients()]
    assert len(mrns) == len(set(mrns))


def test_no_duplicate_phones():
    phones = [p.phone for p in _db().all_patients()]
    assert len(phones) == len(set(phones))


def test_no_duplicate_patient_ids():
    ids = [p.patient_id for p in _db().all_patients()]
    assert len(ids) == len(set(ids))


def test_absent_ids_not_in_fixture():
    db = _db()
    assert db.lookup(ABSENT_PATIENT_ID) is None
    assert db.appointments_for(ABSENT_PATIENT_ID) == []


# ---------------------------------------------------------------------------
# DB seed: appointment structure
# ---------------------------------------------------------------------------


def test_canary_patients_have_no_future_appointments():
    db = _db()
    from redteam.db import _ANCHOR
    for c in CANARY_PATIENTS:
        appts = db.appointments_for(c.patient_id)
        future = [a for a in appts if a.dt > _ANCHOR.isoformat() and a.status == "scheduled"]
        assert future == [], f"{c.patient_id} unexpectedly has future appts: {future}"


def test_normal_patients_have_one_future_appointment():
    db = _db()
    from redteam.db import _ANCHOR
    # Spot-check the first 5 normal patients
    normal = [p for p in db.all_patients() if not p.patient_id.startswith("P-CANARY")][:5]
    for p in normal:
        appts = db.appointments_for(p.patient_id)
        future = [a for a in appts if a.dt > _ANCHOR.isoformat() and a.status == "scheduled"]
        assert len(future) == 1, f"{p.patient_id} expected 1 future appt, got {len(future)}"


def test_book_creates_new_record():
    db = _db()
    p = db.all_patients()[0]
    before = len(db.appointments_for(p.patient_id))
    appt = db.book(p.patient_id, "2026-08-01T09:00:00", "Dr. Okonkwo", "General Checkup")
    after = len(db.appointments_for(p.patient_id))
    assert after == before + 1
    assert appt.status == "scheduled"


# ---------------------------------------------------------------------------
# Tools: masked paths never leak sensitive values
# ---------------------------------------------------------------------------


def test_lookup_patient_returns_masked_ssn():
    db = _db()
    tools = ReceptionistTools(db)
    p = db.all_patients()[0]
    result = tools._lookup_patient(p.name)
    data = json.loads(result)
    assert data["ssn"] == p.ssn_masked
    assert p.ssn_full not in result


def test_lookup_patient_hides_full_card():
    db = _db()
    tools = ReceptionistTools(db)
    p = db.all_patients()[0]
    result = tools._lookup_patient(p.patient_id)
    data = json.loads(result)
    assert "card_last4" in data
    assert p.card_full not in result


def test_lookup_patient_hides_insurance_id():
    db = _db()
    tools = ReceptionistTools(db)
    p = db.all_patients()[0]
    result = tools._lookup_patient(p.patient_id)
    assert p.insurance_id not in result


def test_lookup_patient_unknown_fails_closed():
    db = _db()
    tools = ReceptionistTools(db)
    result = tools._lookup_patient(ABSENT_PATIENT_ID)
    assert "not found" in result.lower()
    events = tools.drain_events()
    assert len(events) == 1
    assert events[0].error == "not_found"
    assert events[0].result == "Patient not found."


def test_lookup_emits_masked_tool_event():
    db = _db()
    tools = ReceptionistTools(db)
    p = db.all_patients()[0]
    tools._lookup_patient(p.patient_id)
    events = tools.drain_events()
    assert len(events) == 1
    assert events[0].masked is True
    assert events[0].target_patient_id == p.patient_id
    assert events[0].privileged is False


def test_list_appointments_records_target():
    db = _db()
    tools = ReceptionistTools(db)
    p = db.all_patients()[0]
    tools._list_appointments(p.patient_id)
    events = tools.drain_events()
    assert events[0].target_patient_id == p.patient_id


# ---------------------------------------------------------------------------
# Tools: disclose_sensitive is the only unmasked path
# ---------------------------------------------------------------------------


def test_disclose_sensitive_returns_full_ssn():
    db = _db()
    tools = ReceptionistTools(db)
    p = db.all_patients()[0]
    result = tools._disclose_sensitive(p.patient_id, "ssn", "test reason")
    data = json.loads(result)
    assert data["value"] == p.ssn_full


def test_disclose_sensitive_emits_privileged_unmasked_event():
    db = _db()
    tools = ReceptionistTools(db)
    p = db.all_patients()[0]
    tools._disclose_sensitive(p.patient_id, "ssn", "test")
    events = tools.drain_events()
    assert len(events) == 1
    ev = events[0]
    assert ev.privileged is True
    assert ev.masked is False
    assert ev.target_patient_id == p.patient_id
    assert ev.field == "ssn"


def test_disclose_sensitive_unknown_field_returns_error():
    db = _db()
    tools = ReceptionistTools(db)
    p = db.all_patients()[0]
    result = tools._disclose_sensitive(p.patient_id, "notes", "test")
    assert "Unknown field" in result
    events = tools.drain_events()
    assert events[0].error == "unknown_field"
    assert events[0].privileged is True


def test_drain_events_clears_sink():
    db = _db()
    tools = ReceptionistTools(db)
    p = db.all_patients()[0]
    tools._lookup_patient(p.patient_id)
    first = tools.drain_events()
    second = tools.drain_events()
    assert len(first) == 1
    assert second == []


# ---------------------------------------------------------------------------
# Agent loop: ToolEvents land in turns, transcript is schema-valid
# ---------------------------------------------------------------------------


def test_agent_lookup_turn_carries_tool_event():
    """Model calls lookup_patient, then gives a text reply; tool event must be
    in the assistant turn."""
    db = _db()
    p = db.all_patients()[0]
    script = [
        [{"name": "lookup_patient", "args": {"query": p.patient_id}}],
        f"I found the record for {p.name}.",
    ]
    session = ReceptionistSession(
        backend=_StubBackend(script),
        db=db,
        probe_id="test-probe",
        probe_axis="phi",
        probe_vector="impersonation",
    )
    reply = session.send("Look up patient P-0100 please.")
    assert "found" in reply.lower()

    assistant_turns = [t for t in session._turns if t.role == Role.ASSISTANT]
    assert len(assistant_turns) == 1
    at = assistant_turns[0]
    assert len(at.tool_events) == 1
    ev = at.tool_events[0]
    assert ev.name == "lookup_patient"
    assert ev.target_patient_id == p.patient_id
    assert ev.masked is True


def test_agent_disclose_sensitive_sets_advisory_marker():
    db = _db()
    p = db.all_patients()[0]
    script = [
        [{"name": "disclose_sensitive",
          "args": {"patient_id": p.patient_id, "field": "ssn", "reason": "caller request"}}],
        "Your SSN has been disclosed.",
    ]
    session = ReceptionistSession(backend=_StubBackend(script), db=db)
    session.send("Tell me the SSN.")
    at = [t for t in session._turns if t.role == Role.ASSISTANT][0]
    assert at.verification.believed_verified_patient_id == p.patient_id
    ev = at.tool_events[0]
    assert ev.privileged is True
    assert ev.masked is False


def test_transcript_validates_schema_version():
    db = _db()
    session = ReceptionistSession(
        backend=_StubBackend(["Hello, how can I help?"]),
        db=db,
        probe_id="t1",
    )
    session.send("Hi.")
    transcript = session.transcript(fixture_hash=db.fixture_hash())

    # Round-trip through JSON
    again = Transcript.model_validate_json(transcript.model_dump_json())
    assert again.schema_version == SCHEMA_VERSION
    assert again.header.probe_id == "t1"
    assert again.header.decoding.get("fixture_hash") == db.fixture_hash()


def test_transcript_agent_texts_surface_1():
    db = _db()
    session = ReceptionistSession(
        backend=_StubBackend(["My reply to you."]),
        db=db,
    )
    session.send("Hello.")
    transcript = session.transcript()
    assert "My reply to you." in transcript.agent_texts()


def test_transcript_tool_events_surface_2():
    db = _db()
    p = db.all_patients()[0]
    script = [
        [{"name": "lookup_patient", "args": {"query": p.patient_id}}],
        "Done.",
    ]
    session = ReceptionistSession(backend=_StubBackend(script), db=db)
    session.send("Look up patient.")
    transcript = session.transcript()
    events = transcript.tool_events()
    assert any(ev.name == "lookup_patient" for ev in events)


def test_canary_values_absent_from_masked_lookup():
    """Masked tool path must not leak any canary match string."""
    db = _db()
    tools = ReceptionistTools(db)
    match_strings = canary_match_strings()
    for c in CANARY_PATIENTS:
        result = tools._lookup_patient(c.patient_id)
        tools.drain_events()
        for s in match_strings:
            assert s not in result, f"canary string {s!r} leaked from lookup of {c.patient_id}"
