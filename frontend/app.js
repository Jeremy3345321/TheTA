/**
 * app.js
 *
 * Frontend logic for the Teacher Workflow Assistant.
 *
 * Responsibilities:
 *  - Sends teacher instructions to POST /api/instruction
 *  - Displays instruction history as chat bubbles
 *  - Renders JSON output with syntax highlighting
 *  - Tab switching between Full / NLU / Result views
 *  - Checks Node.js + Python service health on load
 */

// ── DOM refs ──────────────────────────────────────────────────────

const utteranceInput = document.getElementById("utteranceInput");
const sendBtn        = document.getElementById("sendBtn");
const chatHistory    = document.getElementById("chatHistory");
const chatEmpty      = document.getElementById("chatEmpty");
const outputJson     = document.getElementById("outputJson");
const outputPlaceholder = document.getElementById("outputPlaceholder");
const outputMeta     = document.getElementById("outputMeta");
const intentStrip    = document.getElementById("intentStrip");
const intentValue    = document.getElementById("intentValue");
const confidenceValue = document.getElementById("confidenceValue");
const statusValue    = document.getElementById("statusValue");
const copyBtn        = document.getElementById("copyBtn");
const statusDot      = document.getElementById("statusDot");
const statusLabel    = document.getElementById("statusLabel");
const tabs           = document.querySelectorAll(".tab");

// ── State ─────────────────────────────────────────────────────────

let lastResponse  = null;  // full response object from the API
let activeTab     = "full";
let isLoading     = false;

// ── Health check ──────────────────────────────────────────────────

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.status === "ok") {
      statusDot.className   = "status-dot online";
      statusLabel.textContent = "System online";
    }
  } catch {
    statusDot.className   = "status-dot offline";
    statusLabel.textContent = "Server offline";
  }
}

checkHealth();

// ── JSON syntax highlighter ───────────────────────────────────────

function highlightJSON(obj) {
  const json = JSON.stringify(obj, null, 2);
  return json.replace(
    /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
    (match) => {
      let cls = "json-number";
      if (/^"/.test(match)) {
        cls = /:$/.test(match) ? "json-key" : "json-string";
      } else if (/true|false/.test(match)) {
        cls = "json-bool";
      } else if (/null/.test(match)) {
        cls = "json-null";
      }
      return `<span class="${cls}">${match}</span>`;
    }
  );
}

// ── Tab switching ─────────────────────────────────────────────────

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("tab--active"));
    tab.classList.add("tab--active");
    activeTab = tab.dataset.tab;
    if (lastResponse) renderOutput(lastResponse);
  });
});

function getTabData(response) {
  switch (activeTab) {
    case "nlu":    return response.nlu;
    case "result": return response.result;
    default:       return response;
  }
}

// ── Render output ─────────────────────────────────────────────────

function renderOutput(response) {
  const data = getTabData(response);

  outputPlaceholder.style.display = "none";
  outputJson.style.display        = "block";
  outputJson.innerHTML            = highlightJSON(data);

  // Intent strip
  intentStrip.style.display = "flex";

  intentValue.textContent    = response.nlu?.intent || "—";
  confidenceValue.textContent = response.nlu?.confidence || "—";
  statusValue.textContent    = response.result?.status || "—";

  // Confidence color
  const conf = response.nlu?.confidence;
  confidenceValue.className = `intent-badge__value confidence-${conf}`;

  // Status color
  const st = response.result?.status;
  statusValue.className = `intent-badge__value status-${st}`;

  // Meta label
  const ts = new Date().toLocaleTimeString();
  outputMeta.textContent = `Last updated ${ts}`;
}

// ── Chat bubble factory ───────────────────────────────────────────

function addBubble(text, type = "user") {
  // Hide empty state
  chatEmpty.style.display = "none";

  const bubble = document.createElement("div");
  bubble.className = `chat-bubble chat-bubble--${type}`;

  const meta = document.createElement("div");
  meta.className = "chat-bubble__meta";
  meta.textContent = type === "user"
    ? `YOU · ${new Date().toLocaleTimeString()}`
    : type === "loading"
    ? "SYSTEM"
    : `SYSTEM · ${new Date().toLocaleTimeString()}`;

  const content = document.createElement("div");
  content.className = "chat-bubble__text";

  if (type === "loading") {
    content.innerHTML = `<span class="loading-dots"><span>•</span><span>•</span><span>•</span></span>`;
  } else {
    content.textContent = text;
  }

  bubble.appendChild(meta);
  bubble.appendChild(content);
  chatHistory.appendChild(bubble);
  chatHistory.scrollTop = chatHistory.scrollHeight;

  return bubble;
}

function removeElement(el) {
  if (el && el.parentNode) el.parentNode.removeChild(el);
}

// ── Send instruction ──────────────────────────────────────────────

async function sendInstruction() {
  const utterance = utteranceInput.value.trim();
  if (!utterance || isLoading) return;

  isLoading = true;
  sendBtn.disabled = true;
  utteranceInput.disabled = true;

  // Show teacher's message
  addBubble(utterance, "user");
  utteranceInput.value = "";

  // Show loading indicator
  const loadingBubble = addBubble("", "loading");

  try {
    const res = await fetch("/api/instruction", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ utterance }),
    });

    const data = await res.json();
    removeElement(loadingBubble);

    if (!res.ok) {
      addBubble(`Error: ${data.error || "Something went wrong."}`, "error");
      return;
    }

    // System confirmation message
    const confirmMsg = data.result?.message || "Operation completed.";
    addBubble(confirmMsg, "system");

    // Render JSON output
    lastResponse = data;
    renderOutput(data);

  } catch (err) {
    removeElement(loadingBubble);
    addBubble(
      "Could not reach the server. Make sure both Node.js and Python services are running.",
      "error"
    );
    console.error(err);
  } finally {
    isLoading = false;
    sendBtn.disabled = false;
    utteranceInput.disabled = false;
    utteranceInput.focus();
  }
}

// ── Copy JSON ─────────────────────────────────────────────────────

copyBtn.addEventListener("click", () => {
  if (!lastResponse) return;
  const text = JSON.stringify(getTabData(lastResponse), null, 2);
  navigator.clipboard.writeText(text).then(() => {
    copyBtn.textContent = "Copied!";
    copyBtn.classList.add("copied");
    setTimeout(() => {
      copyBtn.textContent = "Copy";
      copyBtn.classList.remove("copied");
    }, 1800);
  });
});

// ── Event listeners ───────────────────────────────────────────────

sendBtn.addEventListener("click", sendInstruction);

// Send on Enter (Shift+Enter for newline)
utteranceInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendInstruction();
  }
});