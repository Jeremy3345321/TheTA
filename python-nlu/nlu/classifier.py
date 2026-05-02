"""
classifier.py

Intent classification for the Teacher NLU system.

CURRENT IMPLEMENTATION:
    Sentence Embeddings + Cosine Similarity (prototype, no training needed).
    Uses all-MiniLM-L6-v2 as the base model.

FUTURE IMPLEMENTATION:
    SetFit fine-tuned on the benchmark dataset.
    See: classify_intent_setfit() — marked with TODO below.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ── Model ────────────────────────────────────────────────────────────────────
# Loaded once at module import. Reused for every request.

_model = SentenceTransformer("all-MiniLM-L6-v2")


# ── Intent Examples ──────────────────────────────────────────────────────────
# Each intent has a list of representative utterances.
# The classifier compares an incoming utterance against the average
# embedding (centroid) of each intent's examples.
#
# Extend these lists to improve coverage, especially for:
#   - Taglish phrasings (e.g., "I-post ang quiz para sa...")
#   - Abbreviated or shorthand instructions teachers might use
#   - Edge cases discovered during benchmark evaluation

INTENT_EXAMPLES = {
    "post_activity": [
        "post a quiz for the class",
        "upload an assignment for students",
        "create an activity due friday",
        "assign a project to the section",
        "give a test to grade 10",
        "add a worksheet for science class",
        "make an exam for grade 8 rizal",
        "send an assignment to the students due next monday",
        "create a seatwork for grade 9",
        "post homework for section mabini in math",
    ],
    "retrieve_student_info": [
        "get the student profile",
        "retrieve information of a student",
        "find the record of a learner",
        "look up student data",
        "show me the info of a student",
        "search for a student record",
        "fetch the details of a learner",
        "get the profile of a student from the section",
        "find student information from the class list",
        "retrieve the details of a pupil",
    ],
    "record_grades": [
        "record the grades of the class",
        "enter scores for the quarter",
        "encode grades in science",
        "input marks for the section",
        "save student grades",
        "log the grades for the 3rd quarter",
        "submit scores for grade 10 math",
        "record the exam results of the students",
        "encode the quarterly grades",
        "input the grades for the section in filipino",
    ],
    "view_grades": [
        "view the grades of the class",
        "show scores for the quarter",
        "display marks in math",
        "check grades of students",
        "see the grades of grade 9",
        "show me the grades for the 2nd quarter",
        "display the exam results of the section",
        "retrieve the recorded grades",
        "check the scores of students in science",
        "show grades for grade 8 aguinaldo",
    ],
    "create_announcement": [
        "post an announcement for the class",
        "send a notice to the section",
        "make a reminder for students",
        "write an update for grade 8",
        "announce something to the class",
        "notify the students of grade 10",
        "send a memo to section rizal",
        "create a notice for the class",
        "post a message to the students",
        "inform grade 7 mabini about the schedule change",
    ],
    "list_students": [
        "list the students in the section",
        "show all learners in grade 10",
        "display the class list",
        "get names of students",
        "who are the students in section aguinaldo",
        "show me the class roster",
        "display all pupils in grade 9",
        "get the student list for the section",
        "show the names of students in grade 8",
        "retrieve the class list of section mabini",
    ],
    "out_of_scope": [
        "what is the weather today",
        "tell me a joke",
        "what time is it",
        "remind me about the faculty meeting",
        "what is the capital of the philippines",
        "how do i reset my password",
        "search the internet for lesson plans",
        "open youtube",
        "what is my schedule tomorrow",
        "play music",
    ],
}


# ── Precompute Intent Centroids ───────────────────────────────────────────────
# Each intent is represented by the average embedding of its examples.
# This runs once when the module is first imported.

def _build_intent_centroids(
    examples: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    centroids = {}
    for intent, sentences in examples.items():
        embeddings = _model.encode(sentences)           # (n_examples, 384)
        centroids[intent] = np.mean(embeddings, axis=0) # (384,)
    return centroids

_intent_centroids = _build_intent_centroids(INTENT_EXAMPLES)


# ── Prototype Classifier: Cosine Similarity ───────────────────────────────────

def classify_intent(utterance: str, threshold: float = 0.30) -> str:
    """
    Classifies the intent of a teacher utterance using sentence embeddings
    and cosine similarity against precomputed intent centroids.

    Steps:
        1. Encode the utterance into a 384-dimensional embedding vector.
        2. Compute cosine similarity between the utterance vector and
           each intent centroid.
        3. Return the intent with the highest similarity score.
        4. If the best score falls below the threshold, return 'out_of_scope'.

    Args:
        utterance:  The preprocessed teacher instruction string.
        threshold:  Minimum similarity score to accept an intent match.
                    Utterances below this are classified as out_of_scope.
                    Adjust this value based on benchmark evaluation results.

    Returns:
        An intent label string from INTENT_EXAMPLES keys.
    """
    utterance_vec = _model.encode([utterance])  # shape: (1, 384)

    best_intent = "out_of_scope"
    best_score  = 0.0

    for intent, centroid in _intent_centroids.items():
        if intent == "out_of_scope":
            continue  # handled by threshold fallback below

        score = cosine_similarity(utterance_vec, [centroid])[0][0]

        if score > best_score:
            best_score  = score
            best_intent = intent

    # If the best score doesn't clear the threshold,
    # the utterance is likely out of scope
    if best_score < threshold:
        return "out_of_scope"

    return best_intent


# ── TODO: SetFit Classifier ───────────────────────────────────────────────────

def classify_intent_setfit(utterance: str) -> str:
    """
    TODO: Implement SetFit-based intent classification.

    This function will replace classify_intent() once the SetFit model
    has been trained on the benchmark dataset.

    Steps to implement:
        1. Load the trained SetFit model from MODEL_PATH in .env.
        2. Call model.predict([utterance]) to get the predicted intent label.
        3. Return the label string.

    Reference:
        Arora, Jain & Merugu (2024) — Intent Detection in the Age of LLMs.
        EMNLP 2024 Industry Track.
        https://aclanthology.org/2024.emnlp-industry.114.pdf

    Usage (once implemented):
        Swap the call in nlu_pipeline() inside main.py from:
            classify_intent(text)
        to:
            classify_intent_setfit(text)

    Args:
        utterance:  The preprocessed teacher instruction string.

    Returns:
        An intent label string matching the trained label set.
    """
    # TODO: load model
    # from setfit import SetFitModel
    # model = SetFitModel.from_pretrained(os.getenv("MODEL_PATH"))

    # TODO: run inference
    # prediction = model.predict([utterance])
    # return prediction[0]

    raise NotImplementedError(
        "SetFit classifier not yet implemented. "
        "Train the model on the benchmark dataset first, "
        "then complete this function."
    )