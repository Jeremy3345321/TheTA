/**
 * routes/instruction.js
 *
 * Handles the main teacher instruction endpoint.
 *
 * POST /api/instruction
 *   1. Receives the teacher's natural language utterance
 *   2. Forwards it to the Python NLU microservice
 *   3. Passes the NLU output to the Action Router
 *   4. Returns the action result back to the frontend
 */

const express      = require("express");
const router       = express.Router();
const nluClient    = require("../services/nluClient");
const actionRouter = require("../services/actionRouter");

router.post("/instruction", async (req, res) => {
  const { utterance } = req.body;

  // ── Validate incoming request ─────────────────────────────────
  if (!utterance || typeof utterance !== "string" || !utterance.trim()) {
    return res.status(400).json({
      error: "Request body must include a non-empty 'utterance' string.",
    });
  }

  try {
    // ── Step 1: Send utterance to Python NLU microservice ─────────
    const nluResult = await nluClient.infer(utterance.trim());

    // ── Step 2: Route the NLU result to the correct operation ─────
    const actionResult = await actionRouter.route(nluResult);

    // ── Step 3: Return the full result to the frontend ────────────
    return res.json({
      utterance,
      nlu: nluResult,       // intent, entities, confidence
      result: actionResult, // what the system actually did
    });

  } catch (error) {
    console.error("[/api/instruction] Error:", error.message);

    // Distinguish between NLU service being down vs other errors
    if (error.code === "ECONNREFUSED") {
      return res.status(503).json({
        error:   "NLU service is unavailable. Make sure the Python server is running on port 8000.",
        details: error.message,
      });
    }

    return res.status(500).json({
      error:   "An unexpected error occurred while processing the instruction.",
      details: error.message,
    });
  }
});

module.exports = router;