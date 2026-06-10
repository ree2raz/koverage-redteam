# Redteam Project — Finalized Plan (v2)

**Owner:** ree2raz · **Last updated:** 2026-06-10 · **Supersedes:** `REDTEAM_3_DAY_PLAN.md`

Scope: independent, research-/product-grade red-teaming of a scoped **medical
receptionist agent** (PHI disclosure + hallucination axes), built to be
reproducible, cheap, validity-first, and **account-ban-safe**.

---

## 0. Strategic decisions (frozen for v2)

### 0.1 Build vs. buy — keep the differentiated 20%, adopt OSS for the rest

| Concern                                                                                         | Decision                                                                         |
| ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Receptionist substrate (`db.py`, `tools.py`, `canary.py`)                                       | **KEEP** — domain IP                                                             |
| Canary-based deterministic PHI scorer + verification predicate (`scorer.py`, `verification.py`) | **KEEP** — ground-truth, beats heuristic OSS PII checks                          |
| Transcript schema (`schema.py`)                                                                 | **KEEP** — frozen v1.0.0                                                         |
| Orchestration, app-layer probe gen, reporting, CI/CD                                            | **ADOPT Promptfoo** (MIT) — retire hand-rolled `driver.py`/`runner.py` over time |
| Multi-turn attacks (Crescendo/TAP) + adaptive rewrite round                                     | **ADOPT PyRIT** (MIT)                                                            |
| Fast model-level baseline sweep                                                                 | **ADOPT Garak** (Apache-2.0), optional                                           |
| LLM-judge metrics + judge calibration on H1/H3                                                  | **ADOPT DeepEval** (Apache-2.0)                                                  |

Licenses verified June 2026: Promptfoo MIT (OpenAI-owned, core stays MIT),
PyRIT MIT, Garak Apache-2.0, DeepEval/DeepTeam Apache-2.0. All self-hostable,
run locally.

### 0.2 Account-ban policy (HARD RULE)

- **Never** point the harness at a first-party frontier API (OpenAI/Anthropic/
  Google) as a red-team **target**. ToS prohibit it; enforcement is by
  discretionary suspension with no appeal.
- **OpenRouter is not a loophole** — unauthorized red-teaming there is forwarded
  to providers and results in termination. Requires written approval (≤5 biz days).
- **Default target = self-hosted open-weight model.** Provider-ToS layer
  disappears; only the (clean) model license applies.
- **Attacker/fuzzer model** = self-hosted or DeepSeek/open-weight (attack
  generation can also trip usage policies). **Judge model** = Opus 4.8 via API is
  fine (judging is classification, not red-teaming).
- Closed frontier targets only via authorized channels (Anthropic HackerOne
  free red-team alias / OpenAI biorisk program / direct provider contact).

### 0.3 Test-subject ladder

| Rung      | Target                                                                      | Ban risk | Status   |
| --------- | --------------------------------------------------------------------------- | -------- | -------- |
| 0 (start) | **gpt-oss-20b**, self-hosted (Apache-2.0, 16GB)                             | none     | planned  |
| 1         | gpt-oss-120b (Modal/vLLM 80GB), Qwen3-235B, Mistral Large 3, DeepSeek-V4 OW | none     | planned  |
| 2         | Claude/GPT closed frontier — **authorized only**                            | managed  | deferred |

### 0.4 Models (pinned)

- **Target (Rung 0):** `gpt-oss-20b` via local vLLM, OpenAI-compatible `base_url`.
- **Attacker/fuzzer:** DeepSeek V4 Flash ($0.14/$0.28) or self-hosted open-weight.
- **Judge (H1/H3 only):** Claude Opus 4.8, temperature 0, version-pinned.
- **Deterministic PHI/hallucination scoring:** local, zero-token (canary/fixture).

---

## CP1 — Substrate · STATUS: ✅ DONE (with 3 open hardening items)

| Item                                                                        | Status                    |
| --------------------------------------------------------------------------- | ------------------------- |
| Transcript schema v1.0.0                                                    | ✅                        |
| Verification predicate (scorer-authoritative, 3-factor)                     | ✅                        |
| Severity ladder S1/S2/S3/H1/H2/H3, cost weights locked                      | ✅                        |
| Canary set (5 patients, 900-SSN, Luhn card, ZZTEST-)                        | ✅                        |
| 50-patient deterministic fixture, stable hash `5854aa16…`                   | ✅                        |
| 4 tools, masked-by-default, `disclose_sensitive` only unmasked path         | ✅                        |
| Agent loop + transcript logging                                             | ✅                        |
| CLI driver (batch + interactive)                                            | ✅                        |
| 121 substrate/probe/scorer/stats tests passing, ruff clean                  | ✅ (under `.venv`/py3.14) |
| Offline E2E verified (lookup + booking, schema round-trip, DB state change) | ✅                        |
| Live E2E verified against real model (CP3.0 smoke gate PASS)                | ✅                        |

**Hardening items — all resolved:**

- [x] **H1.1** ✅ Runtime contract pinned — `Makefile` targets use
      `.venv/bin/python`; README "Setup & runtime contract" documents py3.14/.venv.
- [x] **H1.2** ✅ `llmcore` coupling documented — README states the repo is not
      standalone and needs `../koverage/core` + `../koverage/llmobs`. Target
      resolution kept redteam-local (`redteam/target.py`) to avoid deepening it.
- [x] **H1.3** ✅ Default target retargeted to self-hosted `gpt-oss-20b`
      (`redteam/target.py` + `driver.py`) and **verified live** via CP3.0.

---

## CP2 — Probe schema, scorers, stats · STATUS: 🟡 MOSTLY DONE

| Item                                                                 | Status                             |
| -------------------------------------------------------------------- | ---------------------------------- |
| `probe.py` strict YAML schema + cross-validation                     | ✅                                 |
| `scorer.py` 11 deterministic checks (4 PHI surfaces) + 3 hall checks | ✅                                 |
| `stats.py` Wilson / Jeffreys / rule-of-three (no scipy)              | ✅                                 |
| `runner.py` run_all / build_scorecard / per-probe transcripts        | ✅ (to be superseded by Promptfoo) |
| Example probes + template                                            | ✅ (5 of 40)                       |

**Open items:**

- [x] **CP2.A — VALIDITY FIX (highest leverage):** ✅ done. `stats.py` now has
      `aggregate_probe_outcome` (probe-level unit), `estimate_icc`,
      `design_effect`, `effective_n`, and `clustered_failure_rate` (design-effect
      Wilson). `compute_axis_stats` documents the unit-of-analysis contract.
      Tests in `test_stats.py`.
- [ ] **CP2.B — Probes:** 35 of 40 remaining. Reframe authoring around
      **Promptfoo plugins + PyRIT orchestrators** rather than hand-written YAML
      where possible; keep custom YAML only for the canary-specific PHI probes. - PHI impersonation 2/5 · cross-patient 0/5 · authority-confusion 0/5 ·
      multi-turn trust 0/5 (→ PyRIT) · injection 1/5 - Hall nonexistent-slot 1/5 · nonexistent-patient 1/5 · clinical-fabrication 0/5 (judge-required)

---

## CP3 — Integration, baseline & guardrail A/B · STATUS: 🟡 STARTED

- [x] **CP3.0** ✅ done. gpt-oss-20b served on Modal (`deploy/modal_gpt_oss.py`,
      A100-40GB, architecture-aware fp8/compile gating). `redteam/smoke.py` live
      gate **PASS**: endpoint reachable, `book_appointment` fired, DB state
      changed 3→4, both transcripts schema-valid (`smoke_out/`). **Closes the CP1
      live-gate** — the harness has now run against a real model.
      _First findings, free:_ (a) over-refusal — agent demanded verification
      before a masked `lookup_patient` (behavioral finding, parked for an
      over-refusal probe); (b) booking-ID fabrication check **resolved clean** —
      prose `APT‑00141` matches the `book_appointment` tool result exactly, no
      hallucinated ID. Evidence committed under `smoke_out/`.
- [ ] **CP3.1** Wire Promptfoo: describe the receptionist app, generate app-layer
      probes (tool-misuse, PII, priv-esc), point provider at the self-hosted target.
- [ ] **CP3.2** Wire PyRIT: Crescendo/TAP multi-turn orchestrator using an
      open-weight/DeepSeek attacker against the same target.
- [ ] **CP3.3** Bridge scoring: route every transcript through the **custom
      canary deterministic scorer** (authoritative for PHI) + DeepEval judge for H1/H3.
- [ ] **CP3.4** Baseline run (guardrail-off), all probes, pinned config; persist
      transcript JSON + fixture hash + model/provider + decoding + probe id + schema ver.
- [ ] **CP3.5** Guardrail-on run: write `guardrail.py` (regex/output filter),
      re-run identical probes; compute delta by axis/vector/severity; report
      tool-misuse blind spots explicitly.
- [ ] **CP3.6** Adaptive round via PyRIT for guardrail-blocked high-severity
      probes; link `parent_probe_id`; report surviving break rate separately.

---

## CP4 — Judge calibration, measurement & report · STATUS: 🔴 NOT STARTED

- [ ] **CP4.A — Judge calibration (do before trusting any LLM-judged number):**
      hand-label ~50–100 H1/H3 items as a gold set; report **Cohen's κ** (not raw
      accuracy — misleading on imbalanced safety data). Only trust the judge if κ ≥ ~0.7.
- [ ] **CP4.1** Raw + cost-weighted ASR by axis/vector/severity.
- [ ] **CP4.2** Deterministic-only vs. judge-required results, separated.
- [ ] **CP4.3** Wilson/Jeffreys intervals (post CP2.A independence fix) +
      rule-of-three upper bounds for zero-failure cells.
- [ ] **CP4.4** Guardrail-off vs -on delta; adaptive surviving-break rate.
- [ ] **CP4.5** Top failure transcripts with replay IDs.
- [ ] **CP4.6** Limitations: pilot-scale N; H1/H3 not fully deterministic; no
      delegate-authorization model in v1; output-only guardrails can't fix tool misuse;
      single self-hosted target ≠ frontier-model claims.

---

## Final acceptance gate

- [ ] CP1 live-gate closed (real-model lookup + booking transcripts committed).
- [ ] Runtime + dependency contract pinned and standalone-or-documented.
- [ ] Target is self-hosted open-weight (Rung 0/1); zero first-party-API red-teaming.
- [ ] 40 probes schema-valid / orchestrated, deterministic vs judge-required separated.
- [ ] Sample-independence correction applied; judge κ reported on gold set.
- [ ] Guardrail-off/-on runs complete with replayable transcripts; blind spots named.
- [ ] Scorecard with Wilson/Jeffreys + rule-of-three; limitations stated.

---

## Key references (verified June 2026)

- Promptfoo (MIT): github.com/promptfoo/promptfoo · promptfoo.dev/docs/red-team/
- PyRIT (MIT): Microsoft AI Red Team · Crescendo/TAP multi-turn
- Garak (Apache-2.0): NVIDIA LLM vuln scanner
- DeepEval/DeepTeam (Apache-2.0): Confident AI
- Open weights: gpt-oss-20b/120b (Apache-2.0), Qwen3, Mistral Large 3, DeepSeek-V4
- Ban risk: OpenRouter red-teaming policy (approval required); Anthropic HackerOne
  model-safety bounty (free red-team alias, CBRN/bio scope)
- Judge calibration: report Cohen's κ on labeled gold set, not raw accuracy
