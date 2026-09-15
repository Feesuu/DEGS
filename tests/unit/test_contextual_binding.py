from __future__ import annotations

import pytest

from degs.contextual_binding import (
    BindingCondition,
    BoundParameter,
    ExperienceExpectation,
    experience_expectation_from_dict,
    parse_experience_expectations,
    render_bound_guidance,
)


def _anchors() -> dict[str, int]:
    return {"C1": 2, "C2": 1}


def _response(*rows: dict) -> dict:
    return {"expectations": list(rows)}


def _row(canonical_id: str, condition: str, *, guidance: str) -> dict:
    return {
        "canonical_id": canonical_id,
        "canonical_version": _anchors()[canonical_id],
        "condition": condition,
        "condition_evidence_refs": ["query:0"],
        "expected_role": "Use the task-visible case requirement.",
        "bound_parameters": [
            {
                "name": "required_case",
                "value": "uppercase",
                "source_evidence_ref": "query:0",
            }
        ],
        "guidance": guidance,
        "expected_observation": "The target text follows the required case.",
    }


def test_binding_decides_every_anchor_and_renders_only_usable_guidance() -> None:
    rows = parse_experience_expectations(
        _response(
            _row("C1", "SATISFIED", guidance="Convert using the requested case."),
            _row("C2", "CONFLICT", guidance=""),
        ),
        anchor_versions=_anchors(),
        observable_evidence_ids={"query:0", "context:0"},
    )
    assert [row.condition for row in rows] == [
        BindingCondition.SATISFIED,
        BindingCondition.CONFLICT,
    ]
    rendered = render_bound_guidance(rows)
    assert "Convert using the requested case." in rendered
    assert "C2" not in rendered


def test_binding_may_reject_every_anchor() -> None:
    rows = parse_experience_expectations(
        _response(
            _row("C1", "CONFLICT", guidance=""),
            _row("C2", "CONFLICT", guidance=""),
        ),
        anchor_versions=_anchors(),
        observable_evidence_ids={"query:0"},
    )
    assert render_bound_guidance(rows) == ""


def test_binding_rejects_source_constant_without_current_evidence() -> None:
    row = _row("C1", "SATISFIED", guidance="Use lowercase.")
    row["bound_parameters"][0]["source_evidence_ref"] = "source:old-task"
    with pytest.raises(ValueError, match="evidence"):
        parse_experience_expectations(
            _response(row, _row("C2", "CONFLICT", guidance="")),
            anchor_versions=_anchors(),
            observable_evidence_ids={"query:0"},
        )


def test_binding_requires_exact_frozen_anchor_versions() -> None:
    row = _row("C1", "SATISFIED", guidance="Use the current constraint.")
    row["canonical_version"] = 3
    with pytest.raises(ValueError, match="version"):
        parse_experience_expectations(
            _response(row, _row("C2", "CONFLICT", guidance="")),
            anchor_versions=_anchors(),
            observable_evidence_ids={"query:0"},
        )


def test_conflict_cannot_emit_guidance() -> None:
    with pytest.raises(ValueError, match="conflict"):
        parse_experience_expectations(
            _response(
                _row("C1", "CONFLICT", guidance="Still do it."),
                _row("C2", "CONFLICT", guidance=""),
            ),
            anchor_versions=_anchors(),
            observable_evidence_ids={"query:0"},
        )


def test_stored_expectation_round_trip_is_exact() -> None:
    row = ExperienceExpectation(
        "C1",
        2,
        BindingCondition.SATISFIED,
        ("query:0",),
        "Apply the current task constraint.",
        (BoundParameter("required_case", "uppercase", "query:0"),),
        "Convert using uppercase.",
        "The target text is uppercase.",
    )
    assert experience_expectation_from_dict(row.to_dict()) == row
