from __future__ import annotations

import pytest

from degs.canonicalize import (
    TemplateRelation,
    canonical_merge_response_schema,
    canonical_unit_candidates,
    parse_canonical_merge,
)


def experience(count: int = 1) -> dict:
    return {
        "operation": "Select records whose text begins with the supplied prefix.",
        "applicability": ["Records contain a text field."],
        "inputs": [{"type": "text", "description": f"Parameter {i}"} for i in range(count)],
        "outputs": [{"type": "indices", "description": "Matching record positions."}],
    }


def test_merge_decision_contains_the_usable_experience_in_one_response() -> None:
    decision = parse_canonical_merge({
        "relation": "SAME_TEMPLATE", "basis": "Only the prefix parameter differs.",
        "canonical_experience": experience(),
    })
    assert decision.relation is TemplateRelation.SAME_TEMPLATE
    assert decision.canonical_experience.to_dict() == experience()
    assert set(canonical_merge_response_schema()["required"]) == {
        "relation", "basis", "canonical_experience",
    }


@pytest.mark.parametrize("relation,payload", [
    ("SAME_TEMPLATE", None), ("DIFFERENT_TEMPLATE", experience()),
    ("UNCERTAIN", experience()), ("OTHER", None),
])
def test_invalid_merge_cannot_commit_only_a_same_label(relation, payload) -> None:
    with pytest.raises(ValueError):
        parse_canonical_merge({"relation": relation, "basis": "Evidence", "canonical_experience": payload})


@pytest.mark.parametrize("relation", ["DIFFERENT_TEMPLATE", "UNCERTAIN"])
def test_non_merge_retains_its_reason(relation) -> None:
    decision = parse_canonical_merge({"relation": relation, "basis": "Different output semantics.", "canonical_experience": None})
    assert decision.canonical_experience is None


def test_long_structured_arrays_are_not_truncated_or_rejected() -> None:
    value = experience(80)
    decision = parse_canonical_merge({"relation": "SAME_TEMPLATE", "basis": "Same operation", "canonical_experience": value})
    assert len(decision.canonical_experience.inputs) == 80
    assert "maxItems" not in str(canonical_merge_response_schema())


def test_candidates_deduplicate_units_before_top_k() -> None:
    vectors = {"q": (1.0, 0.0), **{f"a{i}": (1.0, 0.0) for i in range(20)}, "b": (0.8, 0.2)}
    aliases = {"Q": ("q",), "A": tuple(f"a{i}" for i in range(20)), "B": ("b",)}
    rows = canonical_unit_candidates("Q", aliases, vectors, k=2)
    assert [row.target_canonical_id for row in rows] == ["A", "B"]
    assert [row.rank for row in rows] == [1, 2]


def test_exact_alias_is_retained_outside_top_k_without_deciding_a_merge() -> None:
    rows = canonical_unit_candidates(
        "Q", {"Q": ("q",), "A": ("a",), "B": ("b",)},
        {"q": (1.0, 0.0), "a": (1.0, 0.0), "b": (0.0, 1.0)},
        exact_text_by_leaf={"q": "same", "a": "other", "b": "same"}, k=1,
    )
    assert [row.target_canonical_id for row in rows] == ["A", "B"]
    assert rows[1].exact_text_match
