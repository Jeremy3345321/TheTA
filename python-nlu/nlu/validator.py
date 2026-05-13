"""
validator.py

Intent-aware output validation for the Teacher NLU system.

Ensures that:
    1. The predicted intent is a known label.
    2. The entity dict contains only fields permitted for that intent.
    3. Confidence is computed based on how many required slots were filled.
"""

# ── Allowed entity fields per intent ─────────────────────────────────────────
# Used both for validation (reject rogue keys) and confidence scoring
# (determine which slots are expected to be filled).

INTENT_ENTITY_MAP: dict[str, list[str]] = {
    "post_activity":         ["activity_type", "section", "subject", "due_date"],
    "retrieve_student_info": ["student_name", "section"],
    "record_grades":         ["section", "subject", "quarter"],
    "view_grades":           ["section", "subject", "quarter"],
    "create_announcement":   ["section", "message"],
    "list_students":         ["section"],
    "out_of_scope":          [],
}


def validate(intent: str, entities: dict) -> tuple[bool, str]:
    """
    Validates that the entity dict only contains fields
    permitted for the given intent.

    Args:
        intent:   Classified intent label.
        entities: Extracted entity dict from extractor.py.

    Returns:
        (True, "OK") if valid.
        (False, reason_string) if invalid.
    """
    if intent not in INTENT_ENTITY_MAP:
        return False, f"Unknown intent: '{intent}'"

    allowed = set(INTENT_ENTITY_MAP[intent])
    actual  = set(entities.keys())
    rogue   = actual - allowed

    if rogue:
        return False, f"Entity fields not permitted for '{intent}': {rogue}"

    return True, "OK"


def compute_confidence(intent: str, entities: dict) -> str:
    """
    Heuristic confidence score based on how many expected
    entity slots were successfully filled.

        high   → all expected slots filled
        medium → at least half of expected slots filled
        low    → fewer than half of expected slots filled

    out_of_scope always returns "high" since there are no
    expected slots to fill.

    Args:
        intent:   Classified intent label.
        entities: Extracted entity dict (values may be None).

    Returns:
        "high", "medium", or "low"
    """
    expected_slots = INTENT_ENTITY_MAP.get(intent, [])

    if not expected_slots:
        return "high"

    filled     = sum(1 for slot in expected_slots if entities.get(slot) is not None)
    fill_ratio = filled / len(expected_slots)

    if fill_ratio == 1.0:
        return "high"
    elif fill_ratio >= 0.5:
        return "medium"
    else:
        return "low"