"""
extractor_crf.py

Entity extraction using Conditional Random Fields (CRF).

The CRF labels each token in the utterance with a BIO tag:
    B-SECTION      Beginning of a section entity
    I-SECTION      Inside a section entity
    B-SUBJECT      Beginning of a subject entity
    I-SUBJECT      Inside a subject entity
    B-DUE_DATE     Beginning of a due date entity
    I-DUE_DATE     Inside a due date entity
    B-QUARTER      Beginning of a quarter entity
    I-QUARTER      Inside a quarter entity
    B-ACTIVITY     Beginning of an activity type entity
    I-ACTIVITY     Inside an activity type entity
    B-STUDENT      Beginning of a student name entity
    I-STUDENT      Inside a student name entity
    O              Outside any entity

Requires: pip install sklearn-crfsuite
"""

import sklearn_crfsuite

# ── Feature functions ─────────────────────────────────────────────
#
# CRF doesn't read raw text — it reads hand-crafted features
# computed for each token. The richer the features, the better
# the model can learn entity boundaries.

def word_features(tokens: list[str], i: int) -> dict:
    """
    Computes features for the token at position i.
    Features describe the token itself and its neighbors.

    Args:
        tokens: List of lowercased tokens in the utterance.
        i:      Index of the current token.

    Returns:
        Feature dict for the CRF model.
    """
    word = tokens[i]

    features = {
        # ── Current token features ────────────────────────────────
        "word.lower":      word.lower(),
        "word.isupper":    word.isupper(),
        "word.istitle":    word.istitle(),   # starts with capital
        "word.isdigit":    word.isdigit(),
        "word.prefix2":    word[:2],         # first 2 chars
        "word.prefix3":    word[:3],         # first 3 chars
        "word.suffix2":    word[-2:],        # last 2 chars
        "word.suffix3":    word[-3:],        # last 3 chars

        # ── Domain keyword flags ──────────────────────────────────
        # These act as strong signals for specific entity types
        "is_grade_word":   word.lower() == "grade",
        "is_section_word": word.lower() == "section",
        "is_quarter_word": word.lower() == "quarter",
        "is_ordinal":      word.lower() in (
                               "1st","2nd","3rd","4th",
                               "first","second","third","fourth"
                           ),
        "is_day":          word.lower() in (
                               "monday","tuesday","wednesday",
                               "thursday","friday","saturday","sunday"
                           ),
        "is_this_next":    word.lower() in ("this","next"),
        "is_activity":     word.lower() in (
                               "quiz","assignment","project",
                               "exam","worksheet","activity",
                               "seatwork","homework"
                           ),
        "is_subject":      word.lower() in (
                               "math","mathematics","science",
                               "english","filipino","mapeh",
                               "tle","esp","araling","panlipunan"
                           ),
    }

    # ── Previous token features ───────────────────────────────────
    if i > 0:
        prev = tokens[i - 1]
        features.update({
            "prev.word.lower":    prev.lower(),
            "prev.word.istitle":  prev.istitle(),
            "prev.is_grade_word": prev.lower() == "grade",
            "prev.is_for":        prev.lower() == "for",
            "prev.is_of":         prev.lower() == "of",
            "prev.is_in":         prev.lower() == "in",
        })
    else:
        features["BOS"] = True  # beginning of sentence

    # ── Next token features ───────────────────────────────────────
    if i < len(tokens) - 1:
        nxt = tokens[i + 1]
        features.update({
            "next.word.lower":    nxt.lower(),
            "next.word.istitle":  nxt.istitle(),
            "next.is_quarter":    nxt.lower() == "quarter",
        })
    else:
        features["EOS"] = True  # end of sentence

    return features


def utterance_to_features(tokens: list[str]) -> list[dict]:
    """Converts a full token list into a list of feature dicts."""
    return [word_features(tokens, i) for i in range(len(tokens))]


# ── Training data ─────────────────────────────────────────────────
# Each entry is a list of (token, BIO-tag) pairs.
# In a full implementation this would be a large labeled dataset.
# Here we provide enough to demonstrate the structure.

TRAINING_DATA = [
    # "post a quiz for Grade 10 Rizal in Math due this Friday"
    [
        ("post",   "O"),
        ("a",      "O"),
        ("quiz",   "B-ACTIVITY"),
        ("for",    "O"),
        ("Grade",  "B-SECTION"),
        ("10",     "I-SECTION"),
        ("Rizal",  "I-SECTION"),
        ("in",     "O"),
        ("Math",   "B-SUBJECT"),
        ("due",    "O"),
        ("this",   "B-DUE_DATE"),
        ("Friday", "I-DUE_DATE"),
    ],
    # "record the grades of Grade 8 Mabini in Science for the 3rd quarter"
    [
        ("record",   "O"),
        ("the",      "O"),
        ("grades",   "O"),
        ("of",       "O"),
        ("Grade",    "B-SECTION"),
        ("8",        "I-SECTION"),
        ("Mabini",   "I-SECTION"),
        ("in",       "O"),
        ("Science",  "B-SUBJECT"),
        ("for",      "O"),
        ("the",      "O"),
        ("3rd",      "B-QUARTER"),
        ("quarter",  "I-QUARTER"),
    ],
    # "get the student info of Maria Santos from Section Aguinaldo"
    [
        ("get",       "O"),
        ("the",       "O"),
        ("student",   "O"),
        ("info",      "O"),
        ("of",        "O"),
        ("Maria",     "B-STUDENT"),
        ("Santos",    "I-STUDENT"),
        ("from",      "O"),
        ("Section",   "B-SECTION"),
        ("Aguinaldo", "I-SECTION"),
    ],
    
    # "get student info of Jose Reyes"
    [
        ("get",     "O"),
        ("student", "O"),
        ("info",    "O"),
        ("of",      "O"),
        ("Jose",    "B-STUDENT"),
        ("Reyes",   "I-STUDENT"),
    ],

    # "find the record of Ana Gonzales"
    [
        ("find",     "O"),
        ("the",      "O"),
        ("record",   "O"),
        ("of",       "O"),
        ("Ana",      "B-STUDENT"),
        ("Gonzales", "I-STUDENT"),
    ],
    
    # "retrieve info of Pedro Bautista from Grade 9 Aguinaldo"
    [
        ("retrieve",  "O"),
        ("info",      "O"),
        ("of",        "O"),
        ("Pedro",     "B-STUDENT"),
        ("Bautista",  "I-STUDENT"),
        ("from",      "O"),
        ("Grade",     "B-SECTION"),
        ("9",         "I-SECTION"),
        ("Aguinaldo", "I-SECTION"),
    ],
]



# ── Build CRF training format ─────────────────────────────────────

def prepare_training(data: list) -> tuple:
    X, y = [], []
    for sentence in data:
        tokens = [token for token, _ in sentence]
        labels = [label for _, label in sentence]
        X.append(utterance_to_features(tokens))
        y.append(labels)
    return X, y


# ── Train the CRF model ───────────────────────────────────────────

_crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",      # gradient descent optimizer
    c1=0.1,                 # L1 regularization
    c2=0.1,                 # L2 regularization
    max_iterations=100,
    all_possible_transitions=True,
)

X_train, y_train = prepare_training(TRAINING_DATA)
_crf.fit(X_train, y_train)


# ── Inference: BIO tags → entity dict ────────────────────────────

def tags_to_entities(tokens: list[str], tags: list[str]) -> dict:
    """
    Converts a BIO-tagged token sequence into a flat entity dict.

    Example:
        tokens: ["Grade", "10",  "Rizal", "in",  "Math"]
        tags:   ["B-SECTION", "I-SECTION", "I-SECTION", "O", "B-SUBJECT"]
        →  { "section": "Grade 10 Rizal", "subject": "Math" }
    """
    entities  = {}
    current   = []
    current_type = None

    # Map BIO tag prefixes to entity dict keys
    TAG_TO_KEY = {
        "SECTION":   "section",
        "SUBJECT":   "subject",
        "DUE_DATE":  "due_date",
        "QUARTER":   "quarter",
        "ACTIVITY":  "activity_type",
        "STUDENT":   "student_name",
    }

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            # Save previous entity if any
            if current and current_type:
                key = TAG_TO_KEY.get(current_type)
                if key:
                    entities[key] = " ".join(current).title()

            # Start new entity
            current_type = tag[2:]   # strip "B-"
            current      = [token]

        elif tag.startswith("I-") and current_type == tag[2:]:
            current.append(token)

        else:
            # O tag — flush current entity
            if current and current_type:
                key = TAG_TO_KEY.get(current_type)
                if key:
                    entities[key] = " ".join(current).title()
            current      = []
            current_type = None

    # Flush last entity
    if current and current_type:
        key = TAG_TO_KEY.get(current_type)
        if key:
            entities[key] = " ".join(current).title()

    return entities


def extract_entities_crf(utterance: str, intent: str) -> dict:
    """
    Runs CRF tagging on the utterance and returns
    only the entity fields permitted for the given intent.

    Args:
        utterance:  Original teacher instruction.
        intent:     Classified intent label.

    Returns:
        Intent-aware entity dict.
    """
    from nlu.validator import INTENT_ENTITY_MAP

    tokens   = utterance.split()
    features = utterance_to_features(tokens)
    tags     = _crf.predict([features])[0]

    all_entities = tags_to_entities(tokens, tags)

    # Filter to only allowed fields for this intent
    allowed = INTENT_ENTITY_MAP.get(intent, [])
    return {k: all_entities.get(k) for k in allowed}