/**
 * services/csvReader.js
 *
 * Reads student records and grades from CSV files.
 *
 * Expected CSV files (paths defined in .env):
 *
 *   students.csv columns:
 *     full_name, section, lrn, gender, contact
 *
 *   grades.csv columns:
 *     full_name, section, subject, quarter, grade
 *
 * All matching is case-insensitive and trimmed.
 */

const fs        = require("fs");
const path      = require("path");
const csvParser = require("csv-parser");

const STUDENTS_CSV = process.env.CSV_PATH
  || path.join(__dirname, "../data/students.csv");

const GRADES_CSV = process.env.GRADES_CSV_PATH
  || path.join(__dirname, "../data/grades.csv");

// ── CSV loader helper ─────────────────────────────────────────────────────────

/**
 * Reads a CSV file and returns all rows as an array of objects.
 *
 * @param {string} filePath - Absolute path to the CSV file.
 * @returns {Promise<object[]>}
 */
function loadCSV(filePath) {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(filePath)) {
      // Return empty array instead of crashing if file doesn't exist yet
      console.warn(`[csvReader] CSV file not found: ${filePath}`);
      return resolve([]);
    }

    const rows = [];
    fs.createReadStream(filePath)
      .pipe(csvParser())
      .on("data", (row) => {
        // Normalize all keys and string values to lowercase + trimmed
        const normalized = {};
        for (const [key, value] of Object.entries(row)) {
          normalized[key.trim().toLowerCase()] =
            typeof value === "string" ? value.trim() : value;
        }
        rows.push(normalized);
      })
      .on("end",   () => resolve(rows))
      .on("error", (err) => reject(err));
  });
}

// ── Matching helpers ──────────────────────────────────────────────────────────

function normalize(str) {
  return (str || "").toLowerCase().trim();
}

function matches(rowValue, queryValue) {
  if (!queryValue) return true; // no filter = match all
  return normalize(rowValue).includes(normalize(queryValue));
}

// ── Public methods ────────────────────────────────────────────────────────────

/**
 * Finds a single student by name and/or section.
 *
 * @param {{ student_name?: string, section?: string }} query
 * @returns {Promise<object|null>} First matching student row or null
 */
async function findStudent({ student_name, section } = {}) {
  const rows = await loadCSV(STUDENTS_CSV);

  const match = rows.find(
    (row) =>
      matches(row.full_name, student_name) &&
      matches(row.section,   section)
  );

  return match || null;
}

/**
 * Returns all students in a given section.
 *
 * @param {string} section - Section name to filter by.
 * @returns {Promise<object[]>}
 */
async function getStudentsBySection(section) {
  const rows = await loadCSV(STUDENTS_CSV);

  return rows.filter((row) => matches(row.section, section));
}

/**
 * Finds grade records matching the given filters.
 *
 * @param {{ section?: string, subject?: string, quarter?: string }} query
 * @returns {Promise<object[]>}
 */
async function findGrades({ section, subject, quarter } = {}) {
  const rows = await loadCSV(GRADES_CSV);

  return rows.filter(
    (row) =>
      matches(row.section, section)  &&
      matches(row.subject, subject)  &&
      matches(row.quarter, quarter)
  );
}

module.exports = {
  findStudent,
  getStudentsBySection,
  findGrades,
};