/**
 * services/actionRouter.js
 *
 * Deterministic action dispatcher.
 *
 * Reads the NLU output (intent + entities) and routes the request
 * to the correct platform operation. This is pure JavaScript logic —
 * no LLM involved. Every operation is predictable and testable.
 *
 * Current operations:
 *  post_activity         → Google Classroom (stub for prototype)
 *  retrieve_student_info → CSV reader
 *  record_grades         → CSV writer (stub for prototype)
 *  view_grades           → CSV reader
 *  create_announcement   → Google Classroom (stub for prototype)
 *  list_students         → CSV reader
 *  out_of_scope          → Rejection response
 */

const csvReader = require("./csvReader");

// ── Stubs for Google Classroom operations ─────────────────────────────────────
// These return mock responses for the prototype.
// Replace the internals with real googleapis calls once OAuth is configured.

async function postActivity(entities) {
  const { activity_type, section, subject, due_date } = entities;

  // TODO: replace with real Google Classroom API call
  // const classroom = require("./classroomApi");
  // return await classroom.createCourseWork({ activity_type, section, subject, due_date });

  return {
    status:  "success",
    message: `[STUB] ${activity_type || "Activity"} posted to ${section || "the class"}`
           + `${subject ? " for " + subject : ""}`
           + `${due_date ? ", due " + due_date : ""}.`,
    stub: true,
  };
}

async function createAnnouncement(entities) {
  const { section, message } = entities;

  // TODO: replace with real Google Classroom API call
  return {
    status:  "success",
    message: `[STUB] Announcement posted to ${section || "the class"}`
           + `${message ? ': "' + message + '"' : " (no message body provided)."}`,
    stub: true,
  };
}

async function recordGrades(entities) {
  const { section, subject, quarter } = entities;

  // TODO: implement CSV write logic
  return {
    status:  "success",
    message: `[STUB] Grades recorded for ${section || "the class"}`
           + `${subject ? " in " + subject : ""}`
           + `${quarter ? ", " + quarter : ""}.`,
    stub: true,
  };
}

// ── Live operations (CSV-backed) ──────────────────────────────────────────────

async function retrieveStudentInfo(entities) {
  const { student_name, section } = entities;

  if (!student_name && !section) {
    return {
      status:  "error",
      message: "Could not retrieve student info: no student name or section provided.",
    };
  }

  const student = await csvReader.findStudent({ student_name, section });

  if (!student) {
    return {
      status:  "not_found",
      message: `No student record found for`
             + `${student_name ? " " + student_name : ""}`
             + `${section ? " in " + section : ""}.`,
    };
  }

  return {
    status:  "success",
    message: `Student record found.`,
    data:    student,
  };
}

async function viewGrades(entities) {
  const { section, subject, quarter } = entities;

  if (!section) {
    return {
      status:  "error",
      message: "Could not retrieve grades: no section specified.",
    };
  }

  const grades = await csvReader.findGrades({ section, subject, quarter });

  if (!grades || grades.length === 0) {
    return {
      status:  "not_found",
      message: `No grades found for ${section}`
             + `${subject ? " in " + subject : ""}`
             + `${quarter ? ", " + quarter : ""}.`,
    };
  }

  return {
    status:  "success",
    message: `Grades retrieved for ${section}.`,
    data:    grades,
  };
}

async function listStudents(entities) {
  const { section } = entities;

  if (!section) {
    return {
      status:  "error",
      message: "Could not list students: no section specified.",
    };
  }

  const students = await csvReader.getStudentsBySection(section);

  if (!students || students.length === 0) {
    return {
      status:  "not_found",
      message: `No students found for ${section}.`,
    };
  }

  return {
    status:  "success",
    message: `${students.length} student(s) found in ${section}.`,
    data:    students,
  };
}

// ── Out of scope ──────────────────────────────────────────────────────────────

function handleOutOfScope() {
  return {
    status:  "out_of_scope",
    message: "That instruction isn't supported. Try posting activities, "
           + "retrieving student info, recording or viewing grades, "
           + "creating announcements, or listing students.",
  };
}

// ── Main dispatcher ───────────────────────────────────────────────────────────

/**
 * Routes an NLU result to the appropriate operation.
 *
 * @param {{ intent: string, entities: object, confidence: string }} nluResult
 * @returns {Promise<object>} action result
 */
async function route(nluResult) {
  const { intent, entities, confidence } = nluResult;

  console.log(`[ActionRouter] Intent: ${intent} | Confidence: ${confidence}`);
  console.log(`[ActionRouter] Entities:`, entities);

  // Warn about low confidence but still attempt the operation
  if (confidence === "low") {
    console.warn(`[ActionRouter] Low confidence on intent '${intent}'. `
               + `Some entities may be missing.`);
  }

  switch (intent) {
    case "post_activity":
      return await postActivity(entities);

    case "retrieve_student_info":
      return await retrieveStudentInfo(entities);

    case "record_grades":
      return await recordGrades(entities);

    case "view_grades":
      return await viewGrades(entities);

    case "create_announcement":
      return await createAnnouncement(entities);

    case "list_students":
      return await listStudents(entities);

    case "out_of_scope":
    default:
      return handleOutOfScope();
  }
}

module.exports = { route };