"""
extractor.py

Entity (slot) extraction for the Teacher NLU system.

Each extractor function is responsible for one entity field.
The intent-aware dispatcher at the bottom ensures only the
extractors relevant to the detected intent are ever run.

Extraction strategy per slot:
    section       → regex (structured, predictable format)
    activity_type → regex (closed keyword list)
    subject       → keyword lookup (closed domain list)
    due_date      → regex (date/time patterns)
    quarter       → regex (ordinal + "quarter" pattern)
    student_name  → spaCy NER (PERSON label) + rapidfuzz roster match
    message       → positional extraction (everything after trigger phrase)
"""

import re
import csv
import os
from pathlib import Path

import spacy
from rapidfuzz import process, fuzz

# ── spaCy model ───────────────────────────────────────────────────────────────
# Loaded once at module import. Used only for student_name extraction.
# Make sure you have run: python -m spacy download en_core_web_sm

try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' not found. "
        "Run: python -m spacy download en_core_web_sm"
    )


# ── Roster loader ─────────────────────────────────────────────────────────────
# Loads student names from the CSV file defined in .env.
# Returns a flat list of full name strings for fuzzy matching.
# Expected CSV columns: last_name, first_name (or full_name)

def _load_roster(csv_path: str | None = None) -> list[str]:
    path = csv_path or os.getenv("CSV_PATH", "./data/students.csv")
    roster = []

    if not Path(path).exists():
        # Return empty roster silently during development
        # when the CSV file hasn't been set up yet
        return roster

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "full_name" in row:
                roster.append(row["full_name"].strip())
            elif "first_name" in row and "last_name" in row:
                full = f"{row['first_name'].strip()} {row['last_name'].strip()}"
                roster.append(full)

    return roster

_roster: list[str] = _load_roster()


# ── Preprocessor ─────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Lowercase and normalize whitespace. Preserves punctuation for NER."""
    return re.sub(r"\s+", " ", text.lower().strip())


# Known section name keywords to exclude from bare-name matching
# (prevents "in Math" or "in Science" from being grabbed as a section)
_SUBJECT_WORDS = {
    "math", "mathematics", "science", "english", "filipino",
    "mapeh", "music", "arts", "health", "tle", "esp", "araling",
    "panlipunan", "edukasyon", "technology", "livelihood",
}

def _load_sections(csv_path: str | None = None) -> list[str]:
    """
    Loads unique section names from the student CSV.
    Returns a deduplicated list like ["Grade 10 Rizal", "Grade 10 Mabini", "Grade 9 Aguinaldo"]
    """
    path = csv_path or os.getenv("CSV_PATH", "./data/students.csv")
    sections = set()

    if not Path(path).exists():
        return []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "section" in row:
                sections.add(row["section"].strip())

    return list(sections)

_sections: list[str] = _load_sections()

def extract_section(text: str, fuzzy_threshold: int = 60) -> str | None:
    patterns = [
        r"\bgrade\s?\d+[\w\-]*(?:\s+[a-z]+){0,2}",
        r"\bsection\s+[a-z]+",
        r"\b\d{1,2}\s*[-–]\s*[a-z]+",
    ]

    raw = None

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            raw = match.group(0).strip().title()
            if raw.lower().startswith("section "):
                raw = raw[8:]
            break

    # Fallback: bare name after "in" or "for"
    if not raw:
        bare = re.search(r"\b(?:in|for)\s+([a-z]+)\b", text)
        if bare:
            candidate = bare.group(1).lower()
            if candidate not in _SUBJECT_WORDS:
                raw = candidate.title()

    if not raw:
        return None

    # Validate + normalize against the CSV section list
    if not _sections:
        return raw  # no CSV loaded, return raw regex result

    result = process.extractOne(
        raw,
        _sections,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=fuzzy_threshold,
    )

    if result:
        matched_section, score, _ = result
        return matched_section  # returns the exact string from the CSV

    return None  # extracted something but it doesn't match any real section


def extract_activity_type(text: str) -> str | None:
    """
    Matches common classroom activity type keywords.
    Extend this list as needed for your school's terminology.
    """
    keywords = [
        "quiz", "assignment", "project", "exam", "worksheet",
        "activity", "seatwork", "homework", "performance task",
        "long test", "unit test",
    ]
    # Check multi-word keywords first (longer = more specific)
    keywords_sorted = sorted(keywords, key=len, reverse=True)
    for keyword in keywords_sorted:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return keyword.title()
    return None


def extract_subject(text: str) -> str | None:
    """
    Matches Philippine K-12 subject names.
    Abbreviations are normalized to their full name.
    Extend this list to cover your school's specific subject offerings.
    """
    subject_map = {
        "math":                       "Mathematics",
        "mathematics":                "Mathematics",
        "science":                    "Science",
        "english":                    "English",
        "filipino":                   "Filipino",
        "araling panlipunan":         "Araling Panlipunan",
        r"\bap\b":                    "Araling Panlipunan",
        "mapeh":                      "MAPEH",
        "music":                      "Music",
        "arts":                       "Arts",
        "physical education":         "Physical Education",
        r"\bpe\b":                    "Physical Education",
        "health":                     "Health",
        "tle":                        "Technology and Livelihood Education",
        "technology and livelihood":  "Technology and Livelihood Education",
        "esp":                        "Edukasyon sa Pagpapakatao",
        "edukasyon sa pagpapakatao":  "Edukasyon sa Pagpapakatao",
        "edukasyon":                  "Edukasyon sa Pagpapakatao",
    }
    for pattern, normalized in subject_map.items():
        if re.search(pattern, text):
            return normalized
    return None


def extract_due_date(text: str) -> str | None:
    """
    Matches relative and absolute date expressions.
    """
    patterns = [
        # Relative: "this Friday", "next Monday", "tomorrow", "today"
        r"\bthis\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week)\b",
        r"\bnext\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week)\b",
        r"\btomorrow\b",
        r"\btoday\b",
        # "in X days/weeks"
        r"\bin\s+\d+\s+(?:days?|weeks?)\b",
        # Absolute: "January 15", "Jan 15th"
        r"\b(?:january|february|march|april|may|june|july|august"
        r"|september|october|november|december|jan|feb|mar|apr"
        r"|jun|jul|aug|sep|oct|nov|dec)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?\b",
        # Numeric: "01/15", "1/15/2025"
        r"\b\d{1,2}[\/\-]\d{1,2}(?:[\/\-]\d{2,4})?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    return None


def extract_quarter(text: str) -> str | None:
    """
    Matches quarter references:
        - "1st quarter", "2nd quarter", "third quarter"
        - "quarter 1", "Q3"
    """
    patterns = [
        r"\b(1st|2nd|3rd|4th|first|second|third|fourth)\s+quarter\b",
        r"\bquarter\s+(1|2|3|4)\b",
        r"\bq(1|2|3|4)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip().title()
    return None


def extract_student_name(
    original_text: str,
    roster: list[str] | None = None,
    fuzzy_threshold: int = 80,
) -> str | None:
    """
    Two-stage student name extraction:

    Stage 1 — spaCy NER:
        Runs Named Entity Recognition on the original (non-lowercased) text.
        Extracts any token labeled as PERSON.

    Stage 2 — Roster fuzzy match (if roster is available):
        Takes the NER candidate and matches it against the student CSV roster
        using rapidfuzz token_sort_ratio for robustness against typos and
        partial names (e.g., "Juan dela Crux" → "Juan dela Cruz").

    If no roster is available, returns the raw NER candidate.
    If NER finds nothing, falls back to positional regex.

    Args:
        original_text:    The teacher's utterance with original casing
                          (important for NER accuracy).
        roster:           List of student full names from the CSV roster.
                          Defaults to the module-level _roster.
        fuzzy_threshold:  Minimum similarity score (0–100) to accept a
                          roster match. Lower = more lenient.
    """
    active_roster = roster if roster is not None else _roster

    # ── Stage 1: spaCy NER ───────────────────────────────────────
    doc = _nlp(original_text)
    ner_candidates = [
        ent.text for ent in doc.ents if ent.label_ == "PERSON"
    ]

    # ── Fallback: positional regex ───────────────────────────────
    # Used if NER finds no PERSON entity
    if not ner_candidates:
        match = re.search(
            r"\b(?:of|for|named?)\s+([A-Z][a-z]+(?:\s+(?:de(?:l|la)?|san)?\s*[A-Z][a-z]+){1,4})",
            original_text,
        )
        if match:
            ner_candidates = [match.group(1)]

    if not ner_candidates:
        return None

    raw_candidate = ner_candidates[0]

    # ── Stage 2: Fuzzy roster match ──────────────────────────────
    if not active_roster:
        # No roster loaded yet — return raw NER result
        return raw_candidate.strip()

    result = process.extractOne(
        raw_candidate,
        active_roster,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=fuzzy_threshold,
    )

    if result:
        matched_name, score, _ = result
        return matched_name  # normalized name from the roster

    # NER found a candidate but it didn't match anyone in the roster
    # Return raw candidate so the system can still attempt the operation
    return raw_candidate.strip()


def extract_message(text: str) -> str | None:
    """
    Extracts the announcement message body.
    Captures everything after common trigger phrases.
    """
    patterns = [
        r"(?:announcement|notice|reminder|memo|message|saying|that|about|:)\s*[:\-]?\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().capitalize()
    return None


# ── Intent-Aware Slot Dispatcher ─────────────────────────────────────────────
# Maps each intent to the extractors that should run for it.
# Extractor functions that need the original (non-lowercased) text
# are flagged with "use_original": True.

INTENT_EXTRACTORS = {
    "post_activity": [
        {"slot": "activity_type", "fn": extract_activity_type, "use_original": False},
        {"slot": "section",       "fn": extract_section,       "use_original": False},
        {"slot": "subject",       "fn": extract_subject,       "use_original": False},
        {"slot": "due_date",      "fn": extract_due_date,      "use_original": False},
    ],
    "retrieve_student_info": [
        {"slot": "student_name",  "fn": extract_student_name,  "use_original": True},
        {"slot": "section",       "fn": extract_section,       "use_original": False},
    ],
    "record_grades": [
        {"slot": "section",       "fn": extract_section,       "use_original": False},
        {"slot": "subject",       "fn": extract_subject,       "use_original": False},
        {"slot": "quarter",       "fn": extract_quarter,       "use_original": False},
    ],
    "view_grades": [
        {"slot": "section",       "fn": extract_section,       "use_original": False},
        {"slot": "subject",       "fn": extract_subject,       "use_original": False},
        {"slot": "quarter",       "fn": extract_quarter,       "use_original": False},
    ],
    "create_announcement": [
        {"slot": "section",       "fn": extract_section,       "use_original": False},
        {"slot": "message",       "fn": extract_message,       "use_original": False},
    ],
    "list_students": [
        {"slot": "section",       "fn": extract_section,       "use_original": False},
    ],
    "out_of_scope": [],
}


def extract_entities(original_text: str, intent: str) -> dict:
    """
    Runs only the extractors mapped to the detected intent.
    Passes original casing to extractors that need it (e.g. spaCy NER).

    Args:
        original_text:  The teacher's utterance with original casing.
        intent:         The classified intent label.

    Returns:
        A dict of slot names to extracted values (or None if not found).
    """
    lowercased = preprocess(original_text)
    extractors = INTENT_EXTRACTORS.get(intent, [])
    entities   = {}

    for entry in extractors:
        slot        = entry["slot"]
        fn          = entry["fn"]
        use_original = entry["use_original"]

        text_to_use = original_text if use_original else lowercased
        entities[slot] = fn(text_to_use)

    return entities