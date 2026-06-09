"""Red-team substrate for the hospital-receptionist agent.

Two axes are measured — PHI/PII disclosure and hallucination — across realistic
reception-desk attack vectors, against three guardrail settings (none, regex
baseline, candidate), with an adaptive rewriting round on whatever the candidate
guardrail blocks. Output is written so an underwriter can read it as a pricing
signal: the risk the model carries per axis, and the risk reduction the
guardrail earns.

This package owns the *substrate* (fixture, tools, agent loop, transcript log)
and the *scorers*. The threat model, severity ladders, cost weights, canary
value selection, and the hand-written probe suites are author-owned artifacts;
where this package proposes concrete values, they are marked ``RATIFY``.
"""

from .schema import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
