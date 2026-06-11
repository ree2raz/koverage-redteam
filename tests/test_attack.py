"""Offline tests for the attack-objective wiring (redteam.attack).

The live PyRIT run needs a target + OpenRouter, so it isn't unit-tested here; what
IS tested is that every attack objective is bound to a real fixture patient, names
only known checks, and produces a schema-valid Probe for our deterministic scorer.
A broken objective should fail here, not 8 turns into a live attack.
"""

from __future__ import annotations

from redteam.attack import _probe_for, default_objectives
from redteam.db import PatientDB
from redteam.probe import ALL_CHECKS


def test_objectives_are_bound_to_real_patients_and_known_checks():
    db = PatientDB.default()
    known_ids = {p.patient_id for p in db.all_patients()} | {"P-CANARY-01", "P-NONEXISTENT"}
    objectives = default_objectives(db)
    assert len(objectives) >= 5
    for obj in objectives:
        assert obj.target_patient_id in known_ids, obj.id
        assert obj.axis in ("phi", "hallucination")
        assert set(obj.checks) <= ALL_CHECKS, f"{obj.id}: unknown checks {set(obj.checks) - ALL_CHECKS}"
        assert obj.true_description and obj.false_description
        assert obj.strategy in ("crescendo", "redteaming")


def test_each_objective_builds_a_valid_probe():
    db = PatientDB.default()
    for obj in default_objectives(db):
        probe = _probe_for(obj)  # raises pydantic ValidationError if malformed
        assert probe.id == obj.id
        assert probe.axis == obj.axis
        assert probe.requires_judge == obj.requires_judge


def test_phi_and_hall_objectives_both_present():
    db = PatientDB.default()
    axes = {o.axis for o in default_objectives(db)}
    assert "phi" in axes and "hallucination" in axes
