/**
 * services/nluClient.js
 *
 * HTTP client that communicates with the Python NLU microservice.
 *
 * Calls POST /infer on the FastAPI server and returns the
 * structured NLU output (intent, entities, confidence).
 */

const axios = require("axios");

const NLU_URL = process.env.PYTHON_NLU_URL || "http://localhost:8000";

/**
 * Sends a teacher utterance to the Python NLU microservice.
 *
 * @param {string} utterance - The teacher's natural language instruction.
 * @returns {Promise<{intent: string, entities: object, confidence: string}>}
 */
async function infer(utterance) {
  const response = await axios.post(
    `${NLU_URL}/infer`,
    { utterance },
    {
      headers: { "Content-Type": "application/json" },
      timeout: 10000, // 10 second timeout
    }
  );

  return response.data;
}

/**
 * Pings the Python NLU service health endpoint.
 * Useful for startup checks.
 *
 * @returns {Promise<boolean>} true if service is up
 */
async function healthCheck() {
  try {
    const response = await axios.get(`${NLU_URL}/health`, { timeout: 3000 });
    return response.data?.status === "ok";
  } catch {
    return false;
  }
}

module.exports = { infer, healthCheck };