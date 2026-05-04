/**
 * server.js
 *
 * Express.js entry point for the Teacher Workflow System.
 *
 * Responsibilities:
 *  - Serves the frontend HTML to the browser
 *  - Exposes REST API endpoints for the frontend
 *  - Connects to the Python NLU microservice
 *
 * Start with:
 *  npm run dev   (nodemon, auto-reload)
 *  npm start     (plain node)
 */

require("dotenv").config();

const express    = require("express");
const cors       = require("cors");
const bodyParser = require("body-parser");
const path       = require("path");

const instructionRoute = require("./routes/instruction");

const app  = express();
const PORT = process.env.PORT || 3000;

// ── Middleware ────────────────────────────────────────────────────────────────

app.use(cors());
app.use(bodyParser.json());

// Serve the frontend folder as static files
app.use(express.static(path.join(__dirname, "../frontend")));

// ── Routes ────────────────────────────────────────────────────────────────────

// Main instruction endpoint — teacher input comes in here
app.use("/api", instructionRoute);

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "ok", service: "teacher-workflow-node" });
});

// ── Start server ──────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`\n Teacher Workflow System running at http://localhost:${PORT}`);
  console.log(` Python NLU service expected at  ${process.env.PYTHON_NLU_URL}`);
  console.log(" Press CTRL+C to stop\n");
});