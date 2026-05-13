"""
classifier_lr.py

Intent classification using Logistic Regression + TF-IDF.

Training:
    Converts each example utterance into a TF-IDF vector.
    Fits a Logistic Regression model on those vectors with intent labels.

Inference:
    Vectorizes the incoming utterance using the same fitted TF-IDF.
    Passes the vector through the trained LR model.
    Returns the predicted intent label.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

# ── Training data ─────────────────────────────────────────────────
# Same examples from classifier.py — LR needs labeled (text, label) pairs

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
    ],
    "retrieve_student_info": [
        "get the student profile",
        "retrieve information of a student",
        "find the record of a learner",
        "look up student data",
        "show me the info of a student",
        "search for a student record",
        "fetch the details of a learner",
        "get student info of a learner",       
        "get info of a student",
        "student info for section rizal",
        "get student information of section rizal",
        "show student details from the section",
    ],
    "record_grades": [
        "record the grades of the class",
        "enter scores for the quarter",
        "encode grades in science",
        "input marks for the section",
        "save student grades",
        "log the grades for the 3rd quarter",
    ],
    "view_grades": [
        "view the grades of the class",
        "show scores for the quarter",
        "display marks in math",
        "check grades of students",
        "see the grades of grade 9",
    ],
    "create_announcement": [
        "post an announcement for the class",
        "send a notice to the section",
        "make a reminder for students",
        "write an update for grade 8",
        "announce something to the class",
    ],
    "list_students": [
        "list the students in the section",
        "show all learners in grade 10",
        "display the class list",
        "get names of students",
        "who are the students in section aguinaldo",
        "get student information of section rizal",  
        "show all student info in section mabini",
    ],
    "out_of_scope": [
        "what is the weather today",
        "tell me a joke",
        "what time is it",
        "remind me about the faculty meeting",
        "open youtube",
    ],
}


# ── Build training corpus ─────────────────────────────────────────

def build_corpus(examples: dict) -> tuple[list[str], list[str]]:
    """
    Flattens the intent examples dict into
    parallel lists of texts and labels.

    Returns:
        texts:  ["post a quiz...", "upload an assignment...", ...]
        labels: ["post_activity", "post_activity", ...]
    """
    texts, labels = [], []
    for intent, sentences in examples.items():
        for sentence in sentences:
            texts.append(sentence)
            labels.append(intent)
    return texts, labels


# ── Pipeline: TF-IDF → Logistic Regression ───────────────────────
#
# sklearn Pipeline chains preprocessing and model into one object.
# Calling .fit() trains both steps sequentially.
# Calling .predict() applies both steps to new inputs.
#
# TfidfVectorizer parameters:
#   ngram_range=(1,2) — considers both single words AND word pairs
#                       so "list students" is one feature, not two
#   sublinear_tf=True — applies log normalization to term frequency
#                       prevents very common words from dominating
#
# LogisticRegression parameters:
#   C=1.0             — regularization strength (lower = more regularized)
#   max_iter=1000     — enough iterations to converge on small datasets
#   multi_class='multinomial' — proper softmax for >2 classes

_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )),
    ("lr", LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
    )),
])

# ── Train on module import ────────────────────────────────────────
# Training is fast (~milliseconds) on this dataset size.
# No separate training script needed for prototype.

_texts, _labels = build_corpus(INTENT_EXAMPLES)
_pipeline.fit(_texts, _labels)


# ── Inference ─────────────────────────────────────────────────────

def classify_intent_lr(
    utterance: str,
    oos_threshold: float = 0.35,
) -> str:
    """
    Classifies intent using the trained Logistic Regression pipeline.

    How it works:
        1. TF-IDF vectorizer converts the utterance into a sparse
           numeric vector where each dimension is a word or bigram
           weighted by how informative it is across all training examples.
        2. Logistic Regression applies learned weights to that vector
           and outputs a probability for each intent class via softmax.
        3. The intent with the highest probability is returned.
        4. If the highest probability is below oos_threshold,
           the utterance is classified as out_of_scope.

    Args:
        utterance:      Preprocessed teacher instruction string.
        oos_threshold:  Minimum confidence to accept a prediction.
                        Tune this based on benchmark evaluation.

    Returns:
        Intent label string.
    """
    # predict_proba returns probability for each class
    # shape: (1, n_classes)
    proba  = _pipeline.predict_proba([utterance])[0]
    max_p  = np.max(proba)
    pred   = _pipeline.classes_[np.argmax(proba)]

    if max_p < oos_threshold:
        return "out_of_scope"

    return pred