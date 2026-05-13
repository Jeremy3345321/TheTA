"""
main.py

FastAPI microservice entry point for the Teacher NLU system.

Exposes a single endpoint:
    POST /infer
        Receives a teacher utterance.
        Returns structured JSON: intent, entities, confidence.

Start the server with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from dotenv import load_dotenv 
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nlu.classifier_lr import classify_intent_lr
from nlu.extractor import extract_entities, preprocess
from nlu.validator import validate, compute_confidence

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Teacher NLU Microservice",
    description=(
        "Classifies teacher natural language instructions into structured "
        "intent + entity JSON for downstream platform operations."
    ),
    version="0.1.0",
)

# Allow requests from the Node.js server (localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class InferRequest(BaseModel):
    utterance: str

class InferResponse(BaseModel):
    intent:     str
    entities:   dict
    confidence: str


# ── NLU pipeline ─────────────────────────────────────────────────────────────

def nlu_pipeline(utterance: str) -> dict:
    """
    Full NLU pipeline:
        1. Preprocess (lowercase, normalize whitespace)
        2. Classify intent (cosine similarity — swap to SetFit later)
        3. Extract entities (intent-aware slot extractors)
        4. Validate output (reject rogue entity fields)
        5. Compute confidence score

    To switch to SetFit once trained, replace step 2 with:
        from nlu.classifier import classify_intent_setfit
        intent = classify_intent_setfit(preprocessed)

    Args:
        utterance: Raw teacher instruction string.

    Returns:
        dict with keys: intent, entities, confidence
    """
    preprocessed = preprocess(utterance)

    # Step 2: Intent classification
    intent = classify_intent_lr(preprocessed)

    # Step 3: Entity extraction (passes original text for NER accuracy)
    entities = extract_entities(
        original_text=utterance,
        intent=intent,
    )

    # Step 4: Validate intent-entity consistency
    is_valid, reason = validate(intent, entities)
    if not is_valid:
        # This should not happen under normal operation.
        # Log and fall back to out_of_scope rather than crashing.
        print(f"[NLU Validation Error] {reason}")
        return {
            "intent":     "out_of_scope",
            "entities":   {},
            "confidence": "low",
        }

    # Step 5: Confidence scoring
    confidence = compute_confidence(intent, entities)

    return {
        "intent":     intent,
        "entities":   entities,
        "confidence": confidence,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    """
    Accepts a teacher utterance and returns the NLU output.

    Request body:
        { "utterance": "Post a quiz for Grade 10 Rizal in Math due Friday" }

    Response:
        {
            "intent": "post_activity",
            "entities": {
                "activity_type": "Quiz",
                "section": "Grade 10 Rizal",
                "subject": "Mathematics",
                "due_date": "friday"
            },
            "confidence": "high"
        }
    """
    utterance = request.utterance.strip()

    if not utterance:
        raise HTTPException(
            status_code=400,
            detail="Utterance must not be empty."
        )

    result = nlu_pipeline(utterance)
    return result


@app.get("/health")
async def health():
    """
    Health check endpoint.
    Node.js server can ping this to confirm the NLU service is running.
    """
    return {"status": "ok", "service": "teacher-nlu"}