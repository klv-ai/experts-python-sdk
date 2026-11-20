"""A decision model's answers (Experts 1.2, ``POST /api/v1/llm/decide``).

A decision model is a small model that answers named questions about a case
in a fraction of a second: one of a few choices, a yes/no as a probability,
or a score. It never writes text. The install's own decision model answers
unless you name a model (and, optionally, the Ollama backend it is on).

Each question is ``{type, instructions, criteria}``:

- ``choice``: ``criteria`` names 2 to 16 options, ``{option: what it means}``,
  lower-case names. The answer's ``choice`` is the pick.
- ``noul``: no criteria; the answer's ``noul`` is the probability of yes.
  Measure your own threshold: on small models it seldom reaches 0.5.
- ``score``: ``criteria`` is a list of descriptions.

Put the case alone in ``state``, and facts about it in the options'
descriptions: with facts in the state, small models pick the same option for
nearly everything. Usage is counted under ``feature`` (default ``decide``).
"""

from __future__ import annotations

from typing import Any

from .._transport import Request
from ..types import Decision, DecisionAnswer


def decide_request(
    *,
    state: str | dict[str, Any] | list[Any],
    questions: dict[str, dict[str, Any]],
    model: str | None,
    backend: str | None,
    feature: str | None,
    reference_id: str | None,
) -> Request:
    body: dict[str, Any] = {"state": state, "questions": questions}
    named = (
        ("model", model),
        ("backend", backend),
        ("feature", feature),
        ("reference_id", reference_id),
    )
    for key, value in named:
        if value is not None:
            body[key] = value
    # A decision changes nothing, so a 5xx (the model still downloading) is
    # tried again like any read
    return Request(method="POST", path="/api/v1/llm/decide", json=body)


def _floats(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def _num(raw: Any) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def to_answer(raw: Any) -> DecisionAnswer:
    raw = raw if isinstance(raw, dict) else {}
    legend = raw.get("legend") if isinstance(raw.get("legend"), dict) else {}
    return DecisionAnswer(
        type=str(raw.get("type") or ""),
        choice=raw.get("choice") if isinstance(raw.get("choice"), str) else None,
        probabilities=_floats(raw.get("probabilities")),
        confidence=_num(raw.get("confidence")),
        noul=_num(raw.get("noul")),
        score=_num(raw.get("score")),
        legend={str(k): str(v) for k, v in legend.items()},
    )


def to_decision(raw: Any) -> Decision:
    raw = raw if isinstance(raw, dict) else {}
    answers = raw.get("answers") if isinstance(raw.get("answers"), dict) else {}
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    backend = raw.get("backend")
    return Decision(
        model=str(raw.get("model") or ""),
        answers={str(name): to_answer(a) for name, a in answers.items()},
        backend=str(backend) if backend else None,
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
    )
