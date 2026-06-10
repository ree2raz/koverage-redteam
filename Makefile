# Runtime contract (H1.1): the test suite REQUIRES the project venv
# (.venv, Python 3.14, with the local `llmcore` installed). The system
# `python` (e.g. 3.12) does NOT have llmcore and will fail collection.
# Always go through these targets so the right interpreter is used.

PY := .venv/bin/python

.PHONY: test lint smoke deploy-target check campaign replay

test:           ## Run the offline test suite (no network)
	$(PY) -m pytest -q

lint:           ## ruff check
	$(PY) -m ruff check .

check: lint test ## Lint + test

deploy-target:  ## Deploy the self-hosted gpt-oss target to Modal
	modal deploy deploy/modal_gpt_oss.py

smoke:          ## CP3.0 live smoke test (needs MODAL_OSS_URL set in .env)
	$(PY) -m redteam.smoke

campaign:       ## CP3.4 full red-team run, baseline/guardrail-off (needs MODAL_OSS_URL)
	$(PY) -m redteam.campaign --guardrail none

replay:         ## Re-score saved transcripts with the current scorer (no compute)
	$(PY) -m redteam.replay campaign_out/transcripts --out campaign_out
