import express from 'express';
import cors from 'cors';
import pg from 'pg';
import crypto from 'crypto';
import { VertexAI } from '@google-cloud/vertexai';
import { DocumentProcessorServiceClient } from '@google-cloud/documentai';
import { BigQuery } from '@google-cloud/bigquery';

// ---------- AUTH HELPERS (scrypt; no external deps) ----------
function hashPassword(password, salt) {
  salt = salt || crypto.randomBytes(16).toString('hex');
  const hash = crypto.scryptSync(String(password), salt, 64).toString('hex');
  return { salt, hash };
}
function verifyPassword(password, salt, hash) {
  try {
    if (!salt || !hash) return false;
    const h = crypto.scryptSync(String(password), salt, 64).toString('hex');
    const a = Buffer.from(h), b = Buffer.from(hash);
    return a.length === b.length && crypto.timingSafeEqual(a, b);
  } catch (e) { return false; }
}
let _authReady = false;
async function ensureAuth() {
  if (_authReady) return;
  try {
    await pool.query(`CREATE TABLE IF NOT EXISTS app_users (
      user_id TEXT PRIMARY KEY, full_name TEXT, email TEXT, role TEXT, user_group TEXT, phone TEXT,
      active BOOLEAN DEFAULT TRUE, password_hash TEXT, password_salt TEXT, created_at TIMESTAMPTZ DEFAULT NOW()
    )`);
    await pool.query('ALTER TABLE app_users ADD COLUMN IF NOT EXISTS active BOOLEAN DEFAULT TRUE');
    await pool.query('ALTER TABLE app_users ADD COLUMN IF NOT EXISTS email TEXT');
    await pool.query('ALTER TABLE app_users ADD COLUMN IF NOT EXISTS password_hash TEXT');
    await pool.query('ALTER TABLE app_users ADD COLUMN IF NOT EXISTS password_salt TEXT');
    await pool.query('ALTER TABLE app_users ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()');
    // Seed the testing accounts (idempotent). Password only set if the account has none yet.
    const def = process.env.SEED_PASSWORD || 'Avanciers@360';
    const seed = [
      ['USR-ADMIN1', 'Parv', 'parv@avanciers.com', 'admin'],
      ['USR-HM-PARIKSHITH', 'Parikshith Upadhyaya', 'parikshith.upadhyaya@avanciers.com', 'hiring_manager'],
      ['USR-HM-PUSHPAM', 'Pushpam Singh', 'pushpam.singh@avanciers.com', 'hiring_manager'],
      ['USR-REC-NEHA', 'Neha Das', 'neha.das@avanciers.com', 'recruiter'],
      ['USR-REC-SHUBHAM', 'Shubham Swarup', 'shubham.swarup@avanciers.com', 'recruiter'],
    ];
    for (const [uid, name, email, role] of seed) {
      const { salt, hash } = hashPassword(def);
      await pool.query(
        `INSERT INTO app_users (user_id, full_name, email, role, active, password_hash, password_salt)
         VALUES ($1,$2,$3,$4,TRUE,$5,$6)
         ON CONFLICT (user_id) DO UPDATE SET full_name = EXCLUDED.full_name, email = EXCLUDED.email, role = EXCLUDED.role, active = TRUE,
           password_hash = COALESCE(app_users.password_hash, EXCLUDED.password_hash),
           password_salt = COALESCE(app_users.password_salt, EXCLUDED.password_salt)`,
        [uid, name, email, role, hash, salt]
      );
    }
    // Clean up so only the intended testers (plus anyone the admin adds) are active.
    // 1) Deactivate known legacy demo accounts.
    const legacy = ['admin@avanciers.com', 'rajesh.kumar@avanciers.com', 'riya.sharma@avanciers.com', 'amit.verma@avanciers.com', 'sofia.rossi@avanciers.com'];
    await pool.query('UPDATE app_users SET active = FALSE WHERE LOWER(email) = ANY($1)', [legacy]);
    // 2) Deactivate duplicate rows that share a seeded email but are NOT the seeded account
    //    (e.g. an old "Neha Das" row created as a hiring manager).
    const seededEmails = seed.map(s => s[2].toLowerCase());
    const seededIds = seed.map(s => s[0]);
    await pool.query('UPDATE app_users SET active = FALSE WHERE LOWER(email) = ANY($1) AND user_id <> ALL($2)', [seededEmails, seededIds]);
    _authReady = true;
    console.log('Auth schema + seed + cleanup ensured.');
  } catch (e) { console.log('ensureAuth warn:', e.message); }
}

const app = express();
app.use(cors());
app.use(express.json({ limit: '20mb' }));

const { Pool } = pg;
const pool = new Pool({
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME || 'recruit360',
  host: process.env.DB_HOST || '/cloudsql/direct-tribute-502305-q5:us-central1:recruit360-db-2',
});

const vertex = new VertexAI({ project: 'direct-tribute-502305-q5', location: 'us-central1' });
const genModel = vertex.getGenerativeModel({ model: 'gemini-2.5-flash' });

const docaiClient = new DocumentProcessorServiceClient({ apiEndpoint: 'us-documentai.googleapis.com' });
const PROCESSOR = 'projects/direct-tribute-502305-q5/locations/us/processors/22d406f8def70c29';
const bq = new BigQuery({ projectId: 'direct-tribute-502305-q5' });

app.get('/', (req, res) => res.json({ status: 'Recruit360 API running', module: 'M1-M3' }));

// OCR: PDF/image -> text (Document AI) -> fields (Gemini). Returns timing.
app.post('/jobs/ocr-extract', async (req, res) => {
  try {
    const { fileBase64, mimeType } = req.body;
    if (!fileBase64) return res.status(400).json({ error: 'File is required' });
    const t0 = Date.now();
    const [result] = await docaiClient.processDocument({
      name: PROCESSOR,
      rawDocument: { content: fileBase64, mimeType: mimeType || 'application/pdf' },
    });
    const ocrMs = Date.now() - t0;
    const docText = result.document?.text || '';
    if (!docText.trim()) return res.status(422).json({ error: 'No text found in document' });
    const t1 = Date.now();
    const prompt = 'Extract job details from the text and return ONLY valid JSON with keys: title, client, location, destination_country, openings, skills. If missing use empty string. No markdown, only JSON.\n\nTEXT:\n' + docText;
    const gen = await genModel.generateContent(prompt);
    const genMs = Date.now() - t1;
    let out = gen.response.candidates[0].content.parts[0].text.trim();
    out = out.replace(/```json/g, '').replace(/```/g, '').trim();
    const parsed = JSON.parse(out);
    return res.json({ fields: parsed, timing: { ocr_seconds: (ocrMs / 1000).toFixed(1), structuring_seconds: (genMs / 1000).toFixed(1), total_seconds: ((ocrMs + genMs) / 1000).toFixed(1) } });
  } catch (err) {
    return res.status(500).json({ error: 'Could not OCR-extract', detail: err.message });
  }
});

// Generate a job description (Gemini). Returns timing.
app.post('/jobs/generate-description', async (req, res) => {
  try {
    const { title, client, location, skills, destination_country } = req.body;
    if (!title) return res.status(400).json({ error: 'Job title is required' });
    const t0 = Date.now();
    const prompt = 'Write a concise, professional job description. Title: ' + title + '. Client: ' + (client || 'a leading company') + '. Location: ' + (location || destination_country || 'not specified') + '. Key skills: ' + (skills || 'relevant to the role') + '. Write a 2-sentence overview, then 4 key responsibilities as bullets, then 4 requirements as bullets. Under 180 words. Plain text.';
    const result = await genModel.generateContent(prompt);
    const ms = Date.now() - t0;
    const text = result.response.candidates[0].content.parts[0].text;
    return res.json({ description: text, timing: { seconds: (ms / 1000).toFixed(1) } });
  } catch (err) {
    return res.status(500).json({ error: 'Could not generate description', detail: err.message });
  }
});

// Match candidates to a job (semantic search)
app.post('/jobs/match-candidates', async (req, res) => {
  try {
    const { title, destination_country, skills, top_k } = req.body;
    if (!title) return res.status(400).json({ error: 'Job title is required' });
    let jobdesc = title;
    if (skills) jobdesc += ' with skills ' + skills;
    if (destination_country) jobdesc += ' going to ' + destination_country;
    const k = parseInt(top_k, 10) || 5;
    // Pull a wider pool, then rank by relevance. Destination is a SOFT preference:
    // candidates going to the destination rank first, others still appear (they may relocate / role may be remote).
    const pool = Math.max(k * 3, 15);
    const sql = `
      SELECT h.base.candidate_id AS candidate_id, c.full_name, c.role,
             c.destination_country, c.visa_status, c.email, h.distance
      FROM VECTOR_SEARCH(
        TABLE \`direct-tribute-502305-q5.recruit360.candidate_embeddings\`, 'embedding',
        (SELECT ml_generate_embedding_result AS embedding
         FROM ML.GENERATE_EMBEDDING(
           MODEL \`direct-tribute-502305-q5.recruit360.text_embedder\`,
           (SELECT @q AS content))),
        top_k => ${pool}) AS h
      JOIN \`direct-tribute-502305-q5.recruit360.candidates\` c
        ON c.candidate_id = h.base.candidate_id
      ORDER BY h.distance`;
    const options = { query: sql, location: 'asia-south1', params: { q: jobdesc } };
    const t0 = Date.now();
    const [rows] = await bq.query(options);
    const ms = Date.now() - t0;
    const MATCH_THRESHOLD = 0.92;
    let good = rows.filter(r => r.distance <= MATCH_THRESHOLD);
    // Compute a meaningful fit score (same approach as the Job Board), instead of raw distance.
    const cityToCountry = {
      stockholm:'Sweden', gothenburg:'Sweden', malmo:'Sweden',
      berlin:'Germany', munich:'Germany', frankfurt:'Germany', hamburg:'Germany',
      amsterdam:'Netherlands', rotterdam:'Netherlands', eindhoven:'Netherlands',
      dublin:'Ireland', cork:'Ireland', paris:'France', lyon:'France',
    };
    const destRaw = (destination_country || '').trim();
    const destWanted = (cityToCountry[destRaw.toLowerCase()] || destRaw).toLowerCase();
    const norm = (s) => (s || '').toLowerCase().trim().replace(/s$/, '');
    const roleMatches = (cand) => { const a = norm(cand), b = norm(title); return a && b && (a === b || a.includes(b) || b.includes(a)); };

    good = good.map(r => {
      const sameCountry = destWanted ? (r.destination_country || '').toLowerCase() === destWanted : false;
      const sameRole = roleMatches(r.role);
      let fit = 0;
      if (sameRole) fit += 70;
      if (sameCountry) fit += 25;
      fit += Math.round((1 - r.distance) * 5);
      if (fit > 99) fit = 99;
      if (fit < 1) fit = 1;
      return {
        ...r,
        destination_match: sameCountry,
        match: fit,
      };
    });
    // Rank: same role, then same country, then fit
    good.sort((a, b) => (b.match - a.match));
    good = good.slice(0, k);
    return res.json({ candidates: good, timing: { seconds: (ms / 1000).toFixed(1) } });
  } catch (err) {
    return res.status(500).json({ error: 'Could not match candidates', detail: err.message });
  }
});

// Create a job (with description + assigned_recruiter)
app.post('/jobs', async (req, res) => {
  try {
    const { title, client, role, location, destination_country, openings, status, description, assigned_recruiter } = req.body;
    if (!title || !title.trim()) return res.status(400).json({ error: 'Job title is required (N1)' });
    if (!client || !client.trim()) return res.status(400).json({ error: 'Client is required (N4)' });
    const openInt = parseInt(openings, 10);
    if (isNaN(openInt) || openInt <= 0) return res.status(400).json({ error: 'Openings must be positive (N2)' });
    const job_id = 'JOB' + Date.now().toString().slice(-7);
    const created_date = new Date().toISOString().slice(0, 10);
    const q = 'INSERT INTO jobs (job_id, title, client, department, location, status, openings, recruiter, created_date, description, assigned_recruiter) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) RETURNING *';
    const vals = [job_id, title, client, role || 'General', location || destination_country || '', status || 'OPEN', openInt, 'Recruiter', created_date, description || null, assigned_recruiter || null];
    const result = await pool.query(q, vals);
    return res.status(201).json({ message: 'Job created', job: result.rows[0] });
  } catch (err) {
    return res.status(500).json({ error: 'Server error', detail: err.message });
  }
});

// List jobs
app.get('/jobs', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM jobs ORDER BY created_date DESC LIMIT 100');
    return res.json({ count: result.rowCount, jobs: result.rows });
  } catch (err) {
    return res.status(500).json({ error: 'Server error', detail: err.message });
  }
});

// Get one job
app.get('/jobs/requisitions', async (req, res) => {
  try {
    const { rows } = await pool.query(
      `SELECT job_id, title, job_code, client, hiring_manager, recruiter, country, job_location,
              number_of_positions, priority, status, created_date,
              (CURRENT_DATE - created_date::date) AS ageing_days
         FROM jobs ORDER BY created_date DESC LIMIT 100`
    );
    return res.json({ jobs: rows });
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
});

app.get('/jobs/:id', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM jobs WHERE job_id=$1', [req.params.id]);
    if (result.rowCount === 0) return res.status(404).json({ error: 'Job not found' });
    return res.json({ job: result.rows[0] });
  } catch (err) {
    return res.status(500).json({ error: 'Server error', detail: err.message });
  }
});

// Update job status (publish / close)
app.patch('/jobs/:id/status', async (req, res) => {
  try {
    const { status } = req.body;
    const allowed = ['OPEN', 'POSTED', 'CLOSED', 'CANCELLED'];
    if (!allowed.includes(status)) return res.status(400).json({ error: 'Invalid status' });
    const result = await pool.query('UPDATE jobs SET status=$1 WHERE job_id=$2 RETURNING *', [status, req.params.id]);
    if (result.rowCount === 0) return res.status(404).json({ error: 'Job not found' });
    return res.json({ message: 'Status updated', job: result.rows[0] });
  } catch (err) {
    return res.status(500).json({ error: 'Server error', detail: err.message });
  }
});

// Assign a recruiter to a job
app.patch('/jobs/:id/assign', async (req, res) => {
  try {
    const { recruiter } = req.body;
    if (!recruiter) return res.status(400).json({ error: 'Recruiter is required' });
    const result = await pool.query('UPDATE jobs SET assigned_recruiter=$1 WHERE job_id=$2 RETURNING *', [recruiter, req.params.id]);
    if (result.rowCount === 0) return res.status(404).json({ error: 'Job not found' });
    return res.json({ message: 'Recruiter assigned', job: result.rows[0] });
  } catch (err) {
    return res.status(500).json({ error: 'Server error', detail: err.message });
  }
});

// ============ MODULE 3: CANDIDATES ============

// Search candidates (by name, email, role, or ID)
app.get('/candidates', async (req, res) => {
  try {
    const { q } = req.query;
    let result;
    if (q && q.trim()) {
      const term = '%' + q.trim().toLowerCase() + '%';
      result = await pool.query(
        "SELECT candidate_id, full_name, email, phone, role, destination_country, visa_status FROM candidates WHERE LOWER(full_name) LIKE $1 OR LOWER(email) LIKE $1 OR LOWER(candidate_id) LIKE $1 OR LOWER(role) LIKE $1 ORDER BY full_name LIMIT 25",
        [term]
      );
    } else {
      result = await pool.query('SELECT candidate_id, full_name, email, phone, role, destination_country, visa_status FROM candidates ORDER BY created_at DESC LIMIT 25');
    }
    return res.json({ count: result.rowCount, candidates: result.rows });
  } catch (err) {
    return res.status(500).json({ error: 'Server error', detail: err.message });
  }
});

// Resume OCR: PDF/image -> text (Document AI) -> candidate fields (Gemini)
app.post('/candidates/ocr-extract', async (req, res) => {
  try {
    const { fileBase64, mimeType } = req.body;
    if (!fileBase64) return res.status(400).json({ error: 'File is required' });
    const t0 = Date.now();
    const [result] = await docaiClient.processDocument({ name: PROCESSOR, rawDocument: { content: fileBase64, mimeType: mimeType || 'application/pdf' } });
    const ocrMs = Date.now() - t0;
    const docText = result.document?.text || '';
    if (!docText.trim()) return res.status(422).json({ error: 'No text found in document' });
    const t1 = Date.now();
    const prompt = 'Extract candidate details from this resume and return ONLY valid JSON with keys: full_name, email, phone, role, origin_city, destination_country, skills, experience_years (a number, total years of professional experience; if unclear use 0). No markdown, only JSON.\n\nRESUME:\n' + docText;
    const gen = await genModel.generateContent(prompt);
    const genMs = Date.now() - t1;
    let out = gen.response.candidates[0].content.parts[0].text.trim();
    out = out.replace(/```json/g, '').replace(/```/g, '').trim();
    const parsed = JSON.parse(out);
    return res.json({ fields: parsed, timing: { ocr_seconds: (ocrMs/1000).toFixed(1), structuring_seconds: (genMs/1000).toFixed(1), total_seconds: ((ocrMs+genMs)/1000).toFixed(1) } });
  } catch (err) {
    return res.status(500).json({ error: 'Could not read resume', detail: err.message });
  }
});

// Extract candidate details from PASTED text (e.g. an email the candidate sent) using Gemini.
app.post('/candidates/extract-text', async (req, res) => {
  try {
    const { text } = req.body || {};
    if (!text || !text.trim()) return res.status(400).json({ error: 'text is required' });
    const prompt = 'Extract candidate details from the text below (it may be an email or a pasted resume) and return ONLY valid JSON with keys: full_name, email, phone, role, origin_city, destination_country, skills, experience_years (a number; if unclear use 0). No markdown, only JSON.\n\nTEXT:\n' + String(text).slice(0, 8000);
    const gen = await genModel.generateContent(prompt);
    let out = gen.response.candidates[0].content.parts[0].text.trim().replace(/```json/g, '').replace(/```/g, '').trim();
    let parsed; try { parsed = JSON.parse(out); } catch { parsed = {}; }
    return res.json({ fields: parsed });
  } catch (err) {
    return res.status(500).json({ error: 'Could not read the text', detail: err.message });
  }
});

// Create a candidate (with duplicate check by email)
app.post('/candidates', async (req, res) => {
  try {
    const { full_name, email, phone, role, origin_city, destination_country, consent_given, recruiter, experience_years } = req.body;
    if (!full_name || !full_name.trim()) return res.status(400).json({ error: 'Full name is required' });
    if (!email || !email.trim()) return res.status(400).json({ error: 'Email is required' });
    if (!consent_given) return res.status(400).json({ error: 'Consent is required to register a candidate' });
    // duplicate check by email
    const dup = await pool.query('SELECT candidate_id FROM candidates WHERE LOWER(email)=LOWER($1) LIMIT 1', [email.trim()]);
    if (dup.rowCount > 0) {
      return res.status(409).json({ error: 'DUPLICATE', message: 'A candidate with this email already exists (' + dup.rows[0].candidate_id + ').', existing_id: dup.rows[0].candidate_id });
    }
    const candidate_id = 'C' + Date.now().toString().slice(-6);
    const q = 'INSERT INTO candidates (candidate_id, full_name, email, phone, role, origin_city, destination_country, visa_status, consent_given, recruiter, experience_years, created_at, last_updated) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,NOW(),NOW()) RETURNING candidate_id, full_name, email, role, visa_status, recruiter';
    const vals = [candidate_id, full_name, email, phone || null, role || null, origin_city || null, destination_country || null, 'INTAKE_PENDING', true, recruiter || null, experience_years != null ? parseInt(experience_years, 10) || 0 : null];
    const result = await pool.query(q, vals);
    return res.status(201).json({ message: 'Candidate registered', candidate: result.rows[0] });
  } catch (err) {
    return res.status(500).json({ error: 'Server error', detail: err.message });
  }
});

// Ask about candidates (natural language -> safe candidate search, no hallucination)
app.post('/candidates/ask', async (req, res) => {
  try {
    const { question } = req.body;
    if (!question || !question.trim()) return res.status(400).json({ error: 'Question is required' });

    // Use Gemini to extract structured search intent from the question (role, country, skills)
    const parsePrompt = 'From this recruiter question, extract search fields and return ONLY valid JSON with keys: role, destination_country, skills. Use empty string if not mentioned. No markdown.\n\nQuestion: ' + question;
    let role = '', destination_country = '', skills = '';
    try {
      const g = await genModel.generateContent(parsePrompt);
      let out = g.response.candidates[0].content.parts[0].text.trim().replace(/```json/g,'').replace(/```/g,'').trim();
      const parsed = JSON.parse(out);
      role = parsed.role || '';
      destination_country = parsed.destination_country || '';
      skills = parsed.skills || '';
    } catch (e) { /* fall through with raw question */ }

    const searchText = (role || question) + (skills ? ' with skills ' + skills : '') + (destination_country ? ' going to ' + destination_country : '');
    const THRESHOLD = 0.92;

    const sql = 'SELECT h.base.candidate_id AS candidate_id, c.full_name, c.role, c.destination_country, c.visa_status, c.email, h.distance FROM VECTOR_SEARCH(TABLE `direct-tribute-502305-q5.recruit360.candidate_embeddings`, \'embedding\', (SELECT ml_generate_embedding_result AS embedding FROM ML.GENERATE_EMBEDDING(MODEL `direct-tribute-502305-q5.recruit360.text_embedder`, (SELECT @q AS content))), top_k => 8) AS h JOIN `direct-tribute-502305-q5.recruit360.candidates` c ON c.candidate_id = h.base.candidate_id' + (destination_country ? ' WHERE LOWER(c.destination_country) = LOWER(@dest)' : '') + ' ORDER BY h.distance';

    const params = [{ name: 'q', parameterType: { type: 'STRING' }, parameterValue: { value: searchText } }];
    const queryOpts = { query: sql, location: 'asia-south1', params: { q: searchText } };
    if (destination_country) queryOpts.params.dest = destination_country;

    const t0 = Date.now();
    const [rows] = await bq.query(queryOpts);
    const ms = Date.now() - t0;

    // Apply the safety threshold - reject weak matches so we never invent
    const good = rows.filter(r => r.distance <= THRESHOLD);

    if (good.length === 0) {
      return res.json({
        answer: 'No candidates in the database match that request. I won\'t guess — there is no strong match.',
        candidates: [],
        interpreted: { role, destination_country, skills },
        timing: { seconds: (ms/1000).toFixed(1) },
      });
    }

    return res.json({
      answer: 'Found ' + good.length + ' matching candidate' + (good.length > 1 ? 's' : '') + (role ? ' for ' + role : '') + (destination_country ? ' going to ' + destination_country : '') + '.',
      candidates: good,
      interpreted: { role, destination_country, skills },
      timing: { seconds: (ms/1000).toFixed(1) },
    });
  } catch (err) {
    return res.status(500).json({ error: 'Could not answer', detail: err.message });
  }
});

// Parse pasted email/text into job fields (Gemini)
app.post('/jobs/parse-text', async (req, res) => {
  try {
    const { text } = req.body;
    if (!text || !text.trim()) return res.status(400).json({ error: 'Text is required' });
    const t0 = Date.now();
    const prompt = 'Extract job details from the following email/text and return ONLY valid JSON with keys: title, client, location, destination_country, openings, skills. If a field is missing use empty string. No markdown, only JSON.\n\nTEXT:\n' + text;
    const gen = await genModel.generateContent(prompt);
    const ms = Date.now() - t0;
    let out = gen.response.candidates[0].content.parts[0].text.trim().replace(/```json/g, '').replace(/```/g, '').trim();
    const parsed = JSON.parse(out);
    return res.json({ fields: parsed, timing: { seconds: (ms / 1000).toFixed(1) } });
  } catch (err) {
    return res.status(500).json({ error: 'Could not read the text', detail: err.message });
  }
});

// Job Feasibility Check — availability + visa-readiness + verdict (before posting)
app.post('/jobs/feasibility', async (req, res) => {
  try {
    const { title, destination_country, skills, openings } = req.body;
    if (!title) return res.status(400).json({ error: 'Job title is required' });
    let jobdesc = title;
    if (skills) jobdesc += ' with skills ' + skills;
    if (destination_country) jobdesc += ' going to ' + destination_country;
    const sql = 'SELECT h.base.candidate_id AS candidate_id, c.full_name, c.role, c.destination_country, c.visa_status, h.distance FROM VECTOR_SEARCH(TABLE `direct-tribute-502305-q5.recruit360.candidate_embeddings`, \'embedding\', (SELECT ml_generate_embedding_result AS embedding FROM ML.GENERATE_EMBEDDING(MODEL `direct-tribute-502305-q5.recruit360.text_embedder`, (SELECT @q AS content))), top_k => 50) AS h JOIN `direct-tribute-502305-q5.recruit360.candidates` c ON c.candidate_id = h.base.candidate_id ORDER BY h.distance';
    const t0 = Date.now();
    const [rows] = await bq.query({ query: sql, location: 'asia-south1', params: { q: jobdesc } });
    const ms = Date.now() - t0;

    const THRESHOLD = 0.92;
    const matches = rows.filter(r => r.distance <= THRESHOLD);
    const total = matches.length;

    // visa-ready = advanced in the journey
    const readyStatuses = ['TRAINING_COMPLETE', 'PLACEMENT_ACTIVE', 'VISA_APPROVED', 'TRAVEL_CONFIRMED', 'VISA_SUBMITTED'];
    const visaReady = matches.filter(r => readyStatuses.includes(r.visa_status)).length;

    // same-destination count
    const destWanted = (destination_country || '').trim().toLowerCase();
    const sameDest = destWanted ? matches.filter(r => (r.destination_country || '').toLowerCase() === destWanted).length : total;

    const need = parseInt(openings, 10) || 1;

    // The realistic pool: candidates who actually target this destination (if one was given).
    // If no destination given, the whole matched pool counts.
    const realisticPool = destWanted ? sameDest : total;
    // visa-ready within the realistic pool
    const visaReadyPool = destWanted
      ? matches.filter(r => (r.destination_country || '').toLowerCase() === destWanted && readyStatuses.includes(r.visa_status)).length
      : visaReady;

    // verdict based on the realistic (destination-aware) pool
    let verdict, level;
    if (realisticPool === 0) {
      verdict = destWanted
        ? 'No candidates currently target ' + destination_country + ' for this role. Consider another destination, a remote role, or external sourcing.'
        : 'No matching candidates found for this role. Consider external sourcing.';
      level = 'red';
    } else if (realisticPool >= need * 3 && visaReadyPool >= need) {
      verdict = 'Strong pipeline — you can likely fill all ' + need + ' opening' + (need > 1 ? 's' : '') + ' from existing candidates.';
      level = 'green';
    } else if (realisticPool >= need) {
      verdict = 'Moderate pipeline — ' + realisticPool + ' candidate' + (realisticPool === 1 ? '' : 's') + ' fit, but the pool is limited. You may need external sourcing too.';
      level = 'amber';
    } else {
      verdict = 'Thin pipeline — only ' + realisticPool + ' matching candidate' + (realisticPool === 1 ? '' : 's') + ' for this destination. Consider widening the destination or external sourcing.';
      level = 'amber';
    }

    // suggestion if same-destination pool is thin but similar candidates exist elsewhere
    let suggestion = '';
    if (destWanted && sameDest < need && total > sameDest) {
      const otherDests = [...new Set(matches.filter(r => (r.destination_country || '').toLowerCase() !== destWanted).map(r => r.destination_country))].slice(0, 3);
      if (otherDests.length) suggestion = 'Similar candidates are heading to ' + otherDests.join(', ') + '. Consider these destinations or a remote role to widen the pool.';
    }

    return res.json({
      total_matches: realisticPool,
      visa_ready: visaReadyPool,
      similar_elsewhere: total - sameDest,
      openings: need,
      verdict,
      level,
      suggestion,
      timing: { seconds: (ms / 1000).toFixed(1) },
    });
  } catch (err) {
    return res.status(500).json({ error: 'Could not run feasibility check', detail: err.message });
  }
});

// ---------- CLIENT MANAGEMENT ----------
// List clients
app.get('/clients', async (req, res) => {
  try {
    const q = (req.query.q || '').trim();
    let sql = 'SELECT client_id, client_name, address, email, phone, created_at FROM clients';
    const params = [];
    if (q) { sql += ' WHERE LOWER(client_name) LIKE LOWER($1) OR LOWER(email) LIKE LOWER($1)'; params.push('%' + q + '%'); }
    sql += ' ORDER BY created_at DESC LIMIT 100';
    const { rows } = await pool.query(sql, params);
    return res.json({ clients: rows });
  } catch (err) {
    return res.status(500).json({ error: 'Could not fetch clients', detail: err.message });
  }
});

// Create client
app.post('/clients', async (req, res) => {
  try {
    const { client_name, address, email, phone } = req.body;
    if (!client_name || !client_name.trim()) return res.status(400).json({ error: 'Client name is required' });
    if (!email || !email.trim()) return res.status(400).json({ error: 'Email is required' });
    // duplicate check by email
    const dup = await pool.query('SELECT client_id FROM clients WHERE LOWER(email) = LOWER($1)', [email]);
    if (dup.rows.length) return res.status(409).json({ error: 'duplicate', message: 'A client with this email already exists.' });
    const client_id = 'CL' + Date.now().toString().slice(-7);
    await pool.query(
      'INSERT INTO clients (client_id, client_name, address, email, phone, created_at) VALUES ($1,$2,$3,$4,$5,NOW())',
      [client_id, client_name, address || '', email, phone || '']
    );
    return res.json({ client: { client_id, client_name, address, email, phone } });
  } catch (err) {
    return res.status(500).json({ error: 'Could not create client', detail: err.message });
  }
});

// ---------- CANDIDATE DOCUMENTS (M3 verification) ----------
// Get documents for a candidate
app.get('/candidates/:id/documents', async (req, res) => {
  try {
    const { rows } = await pool.query(
      'SELECT doc_id, candidate_id, doc_type, file_name, status, uploaded_at, verified_at FROM candidate_documents WHERE candidate_id = $1 ORDER BY uploaded_at',
      [req.params.id]
    );
    return res.json({ documents: rows });
  } catch (err) {
    return res.status(500).json({ error: 'Could not fetch documents', detail: err.message });
  }
});

// Upload (register) a document for a candidate
app.post('/candidates/:id/documents', async (req, res) => {
  try {
    const { doc_type, file_name } = req.body;
    if (!doc_type) return res.status(400).json({ error: 'Document type is required' });
    const doc_id = 'DOC' + Date.now().toString().slice(-10);
    await pool.query(
      'INSERT INTO candidate_documents (doc_id, candidate_id, doc_type, file_name, status, uploaded_at) VALUES ($1,$2,$3,$4,$5,NOW())',
      [doc_id, req.params.id, doc_type, file_name || '', 'UPLOADED']
    );
    return res.json({ document: { doc_id, candidate_id: req.params.id, doc_type, file_name, status: 'UPLOADED' } });
  } catch (err) {
    return res.status(500).json({ error: 'Could not upload document', detail: err.message });
  }
});

// Verify a document
app.patch('/documents/:docId/verify', async (req, res) => {
  try {
    await pool.query('UPDATE candidate_documents SET status = $1, verified_at = NOW() WHERE doc_id = $2', ['VERIFIED', req.params.docId]);
    return res.json({ ok: true, doc_id: req.params.docId, status: 'VERIFIED' });
  } catch (err) {
    return res.status(500).json({ error: 'Could not verify document', detail: err.message });
  }
});

// Validate a document with Document AI — check it matches the expected type
app.post('/documents/validate', async (req, res) => {
  try {
    const { fileBase64, mimeType, expectedType, candidateName } = req.body;
    if (!fileBase64) return res.status(400).json({ error: 'File is required' });
    if (!expectedType) return res.status(400).json({ error: 'Expected document type is required' });

    const t0 = Date.now();
    const [result] = await docaiClient.processDocument({ name: PROCESSOR, rawDocument: { content: fileBase64, mimeType: mimeType || 'application/pdf' } });
    const ocrMs = Date.now() - t0;
    const docText = (result.document && result.document.text) ? result.document.text : '';

    // A photograph is an image with no text — that's expected, so accept it.
    const isPhoto = /photo|image|picture/i.test(expectedType);
    const isImageFile = (mimeType || '').startsWith('image/');
    if (!docText.trim()) {
      if (isPhoto || isImageFile) {
        return res.json({ valid: true, reason: 'Image received (a photograph has no text to read).', found_type: 'image', expected_type: expectedType, timing: { seconds: (ocrMs/1000).toFixed(1) } });
      }
      return res.json({ valid: false, reason: 'No readable text found in the document. It may be blank or a low-quality scan.', found_type: 'unknown', expected_type: expectedType, timing: { seconds: (ocrMs/1000).toFixed(1) } });
    }

    const prompt = 'You are a document checker. A CSR uploaded a document that should be a "' + expectedType + '"'
      + (candidateName ? ' for candidate ' + candidateName : '') + '.\n'
      + 'Read the document text and decide if it genuinely appears to be a ' + expectedType + '.\n'
      + 'Return ONLY valid JSON, no markdown, with keys: valid (true/false), reason (one short sentence), found_type (what the document actually appears to be).\n\n'
      + 'DOCUMENT TEXT:\n' + docText.slice(0, 3000);

    const t1 = Date.now();
    const gen = await genModel.generateContent(prompt);
    const genMs = Date.now() - t1;
    let out = gen.response.candidates[0].content.parts[0].text.trim().replace(/```json/g, '').replace(/```/g, '').trim();
    let parsed;
    try { parsed = JSON.parse(out); } catch { parsed = { valid: false, reason: 'Could not analyse the document clearly.', found_type: 'unknown' }; }

    return res.json({
      valid: !!parsed.valid,
      reason: parsed.reason || '',
      found_type: parsed.found_type || '',
      expected_type: expectedType,
      timing: { seconds: ((ocrMs + genMs)/1000).toFixed(1) },
    });
  } catch (err) {
    return res.status(500).json({ error: 'Could not validate document', detail: err.message });
  }
});

// ---------- SAVED CHATS (persist assistant conversations) ----------
app.get('/chats', async (req, res) => {
  try {
    const ue = (req.query.user_email || '').trim();
    let rows;
    if (ue) {
      // Per-user chats, plus any legacy chats saved before per-user scoping (user_email IS NULL).
      ({ rows } = await pool.query('SELECT chat_id, chat_name, messages, created_at FROM saved_chats WHERE user_email = $1 OR user_email IS NULL ORDER BY created_at DESC LIMIT 100', [ue]));
    } else {
      ({ rows } = await pool.query('SELECT chat_id, chat_name, messages, created_at FROM saved_chats ORDER BY created_at DESC LIMIT 100'));
    }
    return res.json({ chats: rows });
  } catch (err) {
    return res.status(500).json({ error: 'Could not fetch chats', detail: err.message });
  }
});

app.post('/chats', async (req, res) => {
  try {
    const { chat_name, messages, user_email } = req.body;
    if (!chat_name || !messages) return res.status(400).json({ error: 'chat_name and messages are required' });
    const chat_id = 'CHAT' + Date.now().toString().slice(-10);
    await pool.query('INSERT INTO saved_chats (chat_id, chat_name, messages, user_email, created_at) VALUES ($1,$2,$3,$4,NOW())',
      [chat_id, chat_name, typeof messages === 'string' ? messages : JSON.stringify(messages), (user_email || '').trim() || null]);
    return res.json({ chat: { chat_id, chat_name } });
  } catch (err) {
    return res.status(500).json({ error: 'Could not save chat', detail: err.message });
  }
});

app.patch('/chats/:id', async (req, res) => {
  try {
    const { chat_name } = req.body;
    if (!chat_name || !chat_name.trim()) return res.status(400).json({ error: 'chat_name is required' });
    await pool.query('UPDATE saved_chats SET chat_name = $1 WHERE chat_id = $2', [chat_name.trim(), req.params.id]);
    return res.json({ ok: true, chat_id: req.params.id, chat_name: chat_name.trim() });
  } catch (err) {
    return res.status(500).json({ error: 'Could not rename chat', detail: err.message });
  }
});

app.delete('/chats/:id', async (req, res) => {
  try {
    await pool.query('DELETE FROM saved_chats WHERE chat_id = $1', [req.params.id]);
    return res.json({ ok: true });
  } catch (err) {
    return res.status(500).json({ error: 'Could not delete chat', detail: err.message });
  }
});

// ---------- PIPELINE (scales to thousands: counts + filters + paging) ----------
// Map visa_status to pipeline stages
app.get('/pipeline/summary', async (req, res) => {
  try {
    const { recruiter, role, destination } = req.query;
    const where = [];
    const params = [];
    let i = 1;
    if (recruiter) { where.push('recruiter = $' + i++); params.push(recruiter); }
    if (role) { where.push('LOWER(role) = LOWER($' + i++ + ')'); params.push(role); }
    if (destination) { where.push('LOWER(destination_country) = LOWER($' + i++ + ')'); params.push(destination); }
    const whereSql = where.length ? ' WHERE ' + where.join(' AND ') : '';
    const sql = 'SELECT visa_status, COUNT(*) AS count FROM candidates' + whereSql + ' GROUP BY visa_status';
    const { rows } = await pool.query(sql, params);

    // Group visa statuses into pipeline stages
    const stageMap = {
      'Intake': ['INTAKE_PENDING', 'STAKEHOLDERS_ASSIGNED', 'TERMS_LOCKED'],
      'Screening': ['DOCUMENTS_VERIFIED', 'SCREENING', 'TRAINING_IN_PROGRESS', 'TRAINING_COMPLETE', 'REMEDIATION_IN_PROGRESS'],
      'Placement': ['PLACEMENT_ACTIVE'],
      'Visa': ['VISA_SUBMITTED', 'VISA_APPROVED', 'VISA_REJECTED'],
      'Travel': ['TRAVEL_CONFIRMED', 'REPORTED', 'NOT_REPORTED', 'ARRIVED'],
    };
    const stageCounts = { Intake: 0, Screening: 0, Placement: 0, Visa: 0, Travel: 0, Other: 0 };
    rows.forEach(r => {
      let placed = false;
      for (const [stage, statuses] of Object.entries(stageMap)) {
        if (statuses.includes(r.visa_status)) { stageCounts[stage] += parseInt(r.count, 10); placed = true; break; }
      }
      if (!placed) stageCounts.Other += parseInt(r.count, 10);
    });
    const total = Object.values(stageCounts).reduce((a, b) => a + b, 0);
    return res.json({ stages: stageCounts, total });
  } catch (err) {
    return res.status(500).json({ error: 'Could not fetch pipeline summary', detail: err.message });
  }
});

// Drill into one stage - paginated list
app.get('/pipeline/stage', async (req, res) => {
  try {
    const { stage, recruiter, role, destination, page } = req.query;
    const stageMap = {
      'Intake': ['INTAKE_PENDING', 'STAKEHOLDERS_ASSIGNED', 'TERMS_LOCKED'],
      'Screening': ['DOCUMENTS_VERIFIED', 'SCREENING', 'TRAINING_IN_PROGRESS', 'TRAINING_COMPLETE', 'REMEDIATION_IN_PROGRESS'],
      'Placement': ['PLACEMENT_ACTIVE'],
      'Visa': ['VISA_SUBMITTED', 'VISA_APPROVED', 'VISA_REJECTED'],
      'Travel': ['TRAVEL_CONFIRMED', 'REPORTED', 'NOT_REPORTED', 'ARRIVED'],
    };
    const statuses = stageMap[stage] || [];
    if (!statuses.length) return res.json({ candidates: [], page: 1, hasMore: false });

    const where = ['visa_status = ANY($1)'];
    const params = [statuses];
    let i = 2;
    if (recruiter) { where.push('recruiter = $' + i++); params.push(recruiter); }
    if (role) { where.push('LOWER(role) = LOWER($' + i++ + ')'); params.push(role); }
    if (destination) { where.push('LOWER(destination_country) = LOWER($' + i++ + ')'); params.push(destination); }

    const pageNum = Math.max(1, parseInt(page, 10) || 1);
    const limit = 15;
    const offset = (pageNum - 1) * limit;
    params.push(limit + 1); const limitIdx = i++;
    params.push(offset); const offsetIdx = i++;

    const sql = 'SELECT candidate_id, full_name, role, destination_country, visa_status, recruiter, csr_owner, urgency_score FROM candidates WHERE ' +
      where.join(' AND ') + ' ORDER BY urgency_score DESC NULLS LAST LIMIT $' + limitIdx + ' OFFSET $' + offsetIdx;
    const { rows } = await pool.query(sql, params);
    const hasMore = rows.length > limit;
    return res.json({ candidates: rows.slice(0, limit), page: pageNum, hasMore });
  } catch (err) {
    return res.status(500).json({ error: 'Could not fetch stage', detail: err.message });
  }
});

// ---------- ADVANCE CANDIDATE STAGE (move through the journey) ----------
// The ordered journey of statuses
// CSR 'getting ready' journey — stops at TRAINING_COMPLETE (candidate is READY).
// Placement, visa and travel are driven by the job application (Job Board) + later steps, not the CSR advance.
const JOURNEY = [
  'INTAKE_PENDING', 'STAKEHOLDERS_ASSIGNED', 'TERMS_LOCKED',
  'DOCUMENTS_VERIFIED', 'SCREENING', 'TRAINING_IN_PROGRESS', 'TRAINING_COMPLETE'
];

// Get a candidate's current status + what the next step would be
app.get('/candidates/:id/journey', async (req, res) => {
  try {
    const { rows } = await pool.query('SELECT candidate_id, full_name, role, destination_country, visa_status, recruiter, csr_owner FROM candidates WHERE candidate_id = $1', [req.params.id]);
    if (!rows.length) return res.status(404).json({ error: 'Candidate not found' });
    const c = rows[0];
    const idx = JOURNEY.indexOf(c.visa_status);
    const nextStatus = (idx >= 0 && idx < JOURNEY.length - 1) ? JOURNEY[idx + 1] : null;
    return res.json({ candidate: c, currentStatus: c.visa_status, nextStatus, atEnd: nextStatus === null, journey: JOURNEY });
  } catch (err) {
    return res.status(500).json({ error: 'Could not fetch journey', detail: err.message });
  }
});

// Advance a candidate to the next status
app.patch('/candidates/:id/advance', async (req, res) => {
  try {
    const { rows } = await pool.query('SELECT visa_status FROM candidates WHERE candidate_id = $1', [req.params.id]);
    if (!rows.length) return res.status(404).json({ error: 'Candidate not found' });
    const current = rows[0].visa_status;
    const idx = JOURNEY.indexOf(current);
    if (idx < 0) return res.status(400).json({ error: 'Current status is not part of the standard journey', current });
    if (idx >= JOURNEY.length - 1) return res.status(400).json({ error: 'Candidate is already at the final stage', current });
    const nextStatus = JOURNEY[idx + 1];
    await pool.query('UPDATE candidates SET visa_status = $1, last_updated = NOW() WHERE candidate_id = $2', [nextStatus, req.params.id]);
    return res.json({ ok: true, candidate_id: req.params.id, previousStatus: current, newStatus: nextStatus });
  } catch (err) {
    return res.status(500).json({ error: 'Could not advance candidate', detail: err.message });
  }
});

// ---------- SUBMISSIONS (M5: screening & submission) ----------
// Screen a candidate against a job - returns a checklist of fit signals
app.post('/submissions/screen', async (req, res) => {
  try {
    const { candidate_id, job_id } = req.body;
    if (!candidate_id) return res.status(400).json({ error: 'candidate_id is required' });

    const cRes = await pool.query('SELECT candidate_id, full_name, role, destination_country, visa_status, experience_years FROM candidates WHERE candidate_id = $1', [candidate_id]);
    let c = cRes.rows[0];
    // Suggested candidates come from the BigQuery pool — fall back there if not in Cloud SQL.
    if (!c) {
      try {
        const rows = await bqQuery(`SELECT candidate_id, full_name, role, destination_country, visa_status, experience_years FROM \`${BQ_DS}.candidates\` WHERE candidate_id = '${(candidate_id || '').replace(/'/g, '')}' LIMIT 1`);
        if (rows && rows.length) c = rows[0];
      } catch (e) { /* fall through */ }
    }
    if (!c) return res.json({ candidate: { candidate_id }, job: null, checklist: [], readiness: 0, verifiedDocs: 0, note: 'Candidate details not found; you can still submit.' });

    let job = null;
    if (job_id) {
      const jRes = await pool.query('SELECT job_id, title, job_location AS location, client, number_of_positions AS openings FROM jobs WHERE job_id = $1', [job_id]);
      if (jRes.rows.length) job = jRes.rows[0];
    }

    // Documents check
    const dRes = await pool.query("SELECT COUNT(*) AS verified FROM candidate_documents WHERE candidate_id = $1 AND status = 'VERIFIED'", [candidate_id]);
    const verifiedDocs = parseInt(dRes.rows[0].verified, 10) || 0;

    // Build screening checklist
    const readyStatuses = ['TRAINING_COMPLETE', 'PLACEMENT_ACTIVE', 'VISA_APPROVED', 'VISA_SUBMITTED', 'TRAVEL_CONFIRMED'];
    const norm = (s) => (s || '').toLowerCase().trim().replace(/s$/, '');  // lowercase, trim, drop trailing plural s
    const contains = (a, b) => { a = norm(a); b = norm(b); return a && b && (a === b || a.includes(b) || b.includes(a)); };
    const roleMatch = job ? contains(c.role, job.title) : null;
    const destMatch = job ? contains(c.destination_country, job.location) : null;

    const checklist = [
      { label: 'Role match', pass: job ? roleMatch : null, detail: job ? `${c.role || '—'} vs ${job.title || '—'}` : `Candidate role: ${c.role || '—'}` },
      { label: 'Destination match', pass: job ? destMatch : null, detail: job ? `${c.destination_country || '—'} vs ${job.location || '—'}` : `Destination: ${c.destination_country || '—'}` },
      { label: 'Documents verified', pass: verifiedDocs >= 4, detail: `${verifiedDocs} of 4 verified` },
      { label: 'Visa progress', pass: readyStatuses.includes(c.visa_status), detail: (c.visa_status || '').replace(/_/g, ' ') },
      { label: 'Experience', pass: (c.experience_years || 0) >= 2, detail: `${c.experience_years || 0} years` },
    ];
    const passCount = checklist.filter(x => x.pass === true).length;
    const applicable = checklist.filter(x => x.pass !== null).length;
    const readiness = applicable ? Math.round((passCount / applicable) * 100) : 0;

    return res.json({ candidate: c, job, checklist, readiness, verifiedDocs });
  } catch (err) {
    return res.status(500).json({ error: 'Could not screen candidate', detail: err.message });
  }
});

// Submit a candidate to a client
app.post('/submissions', async (req, res) => {
  try {
    const { candidate_id, candidate_name, job_id, job_title, client_name, screening_notes, submitted_by } = req.body;
    if (!candidate_id) return res.status(400).json({ error: 'candidate_id is required' });
    // prevent duplicate submission of same candidate to same job
    if (job_id) {
      const dup = await pool.query('SELECT submission_id FROM submissions WHERE candidate_id = $1 AND job_id = $2', [candidate_id, job_id]);
      if (dup.rows.length) return res.status(409).json({ error: 'duplicate', message: 'This candidate has already been submitted to this job.' });
    }
    const submission_id = 'SUB' + Date.now().toString().slice(-10);
    await pool.query(
      'INSERT INTO submissions (submission_id, candidate_id, candidate_name, job_id, job_title, client_name, status, screening_notes, submitted_by, created_at, last_updated) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,NOW(),NOW())',
      [submission_id, candidate_id, candidate_name || '', job_id || null, job_title || '', client_name || '', 'SUBMITTED', screening_notes || '', submitted_by || '']
    );
    return res.json({ submission: { submission_id, candidate_id, candidate_name, job_title, client_name, status: 'SUBMITTED' } });
  } catch (err) {
    return res.status(500).json({ error: 'Could not submit candidate', detail: err.message });
  }
});

// ---------- OFFLINE ADD: create a brand-new candidate + submit to a job (empty-pool flow) ----------
// Recruiter found a real candidate offline, has their resume, and adds them to a job.
// Enters the SAME pipeline: PENDING_HM_APPROVAL -> approve -> (resume already in) -> interviews.
app.post('/submissions/offline', async (req, res) => {
  try {
    const b = req.body || {};
    if (!b.candidate_name || !b.job_id) return res.status(400).json({ error: 'candidate_name and job_id are required' });
    // 1) Create the candidate (or reuse a supplied candidate_id)
    let candidate_id = b.candidate_id;
    if (!candidate_id) {
      candidate_id = 'C' + Date.now().toString().slice(-6);
      const email = b.email || candidateEmail(candidate_id);
      try {
        await pool.query(
          `INSERT INTO candidates (candidate_id, full_name, email, phone, role, origin_city, destination_country, visa_status, consent_given, recruiter, experience_years, created_at, last_updated)
           VALUES ($1,$2,$3,$4,$5,$6,$7,'INTAKE_PENDING',TRUE,$8,$9,NOW(),NOW())`,
          [candidate_id, b.candidate_name, email, b.phone || '', b.role || '', b.origin_city || '', b.destination_country || '', b.submitted_by || '', b.experience_years ? Number(b.experience_years) : null]
        );
      } catch (e) { /* candidate table optional cols — best effort */ }
    }
    // 2) Prevent duplicate on this job
    const dup = await pool.query('SELECT submission_id FROM submissions WHERE candidate_id = $1 AND job_id = $2', [candidate_id, b.job_id]);
    if (dup.rows.length) return res.status(409).json({ error: 'duplicate', message: 'This candidate is already on this job.' });
    // 3) Create the submission in the RECRUITER's hands (new flow: resume + summary come BEFORE
    //    hiring-manager approval). Recruiter with a resume in hand -> ready to send the summary to
    //    the HM. Pool pick with no resume -> recruiter must request the resume first.
    const hasResume = b.has_resume !== false;
    const submission_id = 'SUB' + Date.now().toString().slice(-10);
    await pool.query(
      `INSERT INTO submissions (submission_id, candidate_id, candidate_name, job_id, job_title, client_name, status, screening_notes, submitted_by, resume_received, resume_requested, created_at, last_updated)
       VALUES ($1,$2,$3,$4,$5,$6,'SUBMITTED',$7,$8,$9,$9,NOW(),NOW())`,
      [submission_id, candidate_id, b.candidate_name, b.job_id, b.job_title || '', b.client_name || '', b.screening_notes || '', b.submitted_by || 'recruiter', hasResume]
    );
    // 4) Log the offline document/resume (only when the recruiter actually added one)
    if (hasResume) {
      await logComm({ submission_id, candidate_id, job_id: b.job_id, channel: 'DOCUMENT', to_role: 'recruiter', to_name: b.submitted_by || 'recruiter',
        subject: (b.doc_type || 'Resume') + ' added (offline)',
        body: `${b.doc_type || 'Resume'} for ${b.candidate_name}${b.source ? ' (received via ' + b.source + ')' : ''} added while creating the submission.` + (b.resume_text ? '\n\n--- Pasted content ---\n' + String(b.resume_text).slice(0, 4000) : ''),
        sent_by: b.submitted_by || 'recruiter' });
      // stash the resume text so the AI summary step can use it
      if (b.resume_text) { try { await pool.query('UPDATE submissions SET resume_text = $1 WHERE submission_id = $2', [String(b.resume_text).slice(0, 8000), submission_id]); } catch (e) {} }
    }
    // 5) Guide the recruiter on the next step (HM is looped in later, after the summary).
    const recMsg = hasResume
      ? `${b.candidate_name} added for ${b.job_title || 'a role'} — review the resume and send the summary to the hiring manager.`
      : `${b.candidate_name} added for ${b.job_title || 'a role'} — request the resume from the candidate.`;
    await notify({ candidate_id, recipient: b.submitted_by || 'recruiter', type: 'CANDIDATE_ADDED', message: recMsg });
    return res.json({ ok: true, submission: { submission_id, candidate_id, candidate_name: b.candidate_name, status: 'SUBMITTED' } });
  } catch (err) {
    return res.status(500).json({ error: 'Could not add candidate', detail: err.message });
  }
});

// List submissions (optionally by job or candidate)
app.get('/submissions', async (req, res) => {
  try {
    const { job_id, candidate_id } = req.query;
    let sql = 'SELECT submission_id, candidate_id, candidate_name, job_id, job_title, client_name, status, screening_notes, submitted_by, interview_date, interview_notes, offer_status, offer_notes, created_at FROM submissions';
    const params = [];
    const where = [];
    let i = 1;
    if (job_id) { where.push('job_id = $' + i++); params.push(job_id); }
    if (candidate_id) { where.push('candidate_id = $' + i++); params.push(candidate_id); }
    if (where.length) sql += ' WHERE ' + where.join(' AND ');
    sql += ' ORDER BY created_at DESC LIMIT 100';
    const { rows } = await pool.query(sql, params);
    return res.json({ submissions: rows });
  } catch (err) {
    return res.status(500).json({ error: 'Could not fetch submissions', detail: err.message });
  }
});

// Update a submission's status
app.patch('/submissions/:id/status', async (req, res) => {
  try {
    const { status, scenario } = req.body;
    const allowed = ['SUBMITTED', 'PENDING_HM_APPROVAL', 'RECRUITER_CALL', 'CLIENT_INTERVIEW', 'PENDING_PLACEMENT_APPROVAL', 'OFFER', 'PLACED', 'REJECTED',
                     'CLIENT_CONFIRM', 'TRAINING', 'CLIENT_DECISION', 'CLIENT_REVIEW', 'SHORTLISTED', 'INTERVIEW_SCHEDULED', 'INTERVIEWED', 'OFFERED', 'OFFER_ACCEPTED', 'OFFER_DECLINED'];
    if (!allowed.includes(status)) return res.status(400).json({ error: 'Invalid status', allowed });

    // ENFORCE ORDER (new flow: resume + summary -> HM approval -> recruiter call -> client interview; HR round removed).
    const ORDER = ['SUBMITTED', 'PENDING_HM_APPROVAL', 'RECRUITER_CALL', 'CLIENT_INTERVIEW', 'PENDING_PLACEMENT_APPROVAL', 'OFFER', 'PLACED'];
    const curRes = await pool.query('SELECT status FROM submissions WHERE submission_id = $1', [req.params.id]);
    if (!curRes.rows.length) return res.status(404).json({ error: 'Submission not found' });
    const current = curRes.rows[0].status;
    const terminal = ['REJECTED', 'OFFER_DECLINED', 'PLACED'];
    if (terminal.includes(current)) {
      return res.status(400).json({ error: 'This candidate is already at a final stage (' + current.replace(/_/g,' ') + ') and cannot be changed.' });
    }
    // Rejection/decline allowed from any non-terminal stage
    if (status === 'REJECTED' || status === 'OFFER_DECLINED') {
      // allowed
    } else {
      const ci = ORDER.indexOf(current);
      const ni = ORDER.indexOf(status);
      if (ni === -1) return res.status(400).json({ error: 'That status is not part of the forward flow.' });
      // Forward-only: allow moving forward one OR more steps (e.g. Scenario 3 skips training).
      // Block only backward moves.
      if (ni < ci) {
        return res.status(400).json({ error: 'Cannot move a candidate backward in the flow.', current });
      }
    }

    await pool.query('UPDATE submissions SET status = $1, last_updated = NOW() WHERE submission_id = $2', [status, req.params.id]);
    if (scenario) { try { await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS scenario INTEGER'); await pool.query('UPDATE submissions SET scenario = $1 WHERE submission_id = $2', [Number(scenario), req.params.id]); } catch(e){} }

    // Look up the submission for connection + notifications
    const sRes = await pool.query('SELECT candidate_id, candidate_name, job_title, client_name, submitted_by FROM submissions WHERE submission_id = $1', [req.params.id]);
    const sub = sRes.rows[0];

    if (sub) {
      // CONNECTION #1: when PLACED, advance the candidate's overall journey (Track A)
      if (status === 'PLACED') {
        await pool.query("UPDATE candidates SET visa_status = 'PLACEMENT_ACTIVE', last_updated = NOW() WHERE candidate_id = $1", [sub.candidate_id]);
        await notify({ candidate_id: sub.candidate_id, recipient: sub.submitted_by, type: 'PLACED', message: sub.candidate_name + ' has been placed for ' + (sub.job_title || 'a role') + (sub.client_name ? ' at ' + sub.client_name : '') + '. Their journey moved to Placement.' });
        await notify({ candidate_id: sub.candidate_id, recipient: 'candidate', type: 'PLACED', message: 'Congratulations! You have been placed for ' + (sub.job_title || 'a role') + (sub.client_name ? ' at ' + sub.client_name : '') + '.' });
      } else {
        // Notify the recruiter of meaningful status changes
        const notifyStatuses = ['SHORTLISTED', 'REJECTED', 'OFFERED', 'OFFER_ACCEPTED', 'OFFER_DECLINED'];
        if (notifyStatuses.includes(status)) {
          await notify({ candidate_id: sub.candidate_id, recipient: sub.submitted_by, type: 'STATUS_CHANGE', message: sub.candidate_name + ' is now ' + status.replace(/_/g, ' ').toLowerCase() + ' for ' + (sub.job_title || 'a role') + '.' });
        }
      }
    }
    return res.json({ ok: true, submission_id: req.params.id, status });
  } catch (err) {
    return res.status(500).json({ error: 'Could not update submission', detail: err.message });
  }
});

// Generate a professional client submission note (AI-assisted)
app.post('/submissions/generate-note', async (req, res) => {
  try {
    const { candidate_id, job_title, client_name } = req.body;
    if (!candidate_id) return res.status(400).json({ error: 'candidate_id is required' });

    const cRes = await pool.query('SELECT full_name, role, destination_country, visa_status, experience_years FROM candidates WHERE candidate_id = $1', [candidate_id]);
    if (!cRes.rows.length) return res.status(404).json({ error: 'Candidate not found' });
    const c = cRes.rows[0];

    const dRes = await pool.query("SELECT COUNT(*) AS verified FROM candidate_documents WHERE candidate_id = $1 AND status = 'VERIFIED'", [candidate_id]);
    const verifiedDocs = parseInt(dRes.rows[0].verified, 10) || 0;

    const facts = [
      'Name: ' + c.full_name,
      'Role: ' + (c.role || 'not specified'),
      'Destination: ' + (c.destination_country || 'not specified'),
      'Experience: ' + (c.experience_years != null ? c.experience_years + ' years' : 'not specified'),
      'Visa status: ' + (c.visa_status || 'not specified').replace(/_/g, ' '),
      'Documents verified: ' + verifiedDocs + ' of 4',
      job_title ? 'Applying for: ' + job_title : '',
      client_name ? 'Client: ' + client_name : '',
    ].filter(Boolean).join('\n');

    const prompt = 'You are a recruitment coordinator writing a short, professional submission note to a client about a candidate. '
      + 'Use ONLY the facts provided below - do not invent skills, qualifications, or details that are not listed. '
      + 'Write 2-3 sentences, professional and positive but factual, explaining why this candidate is a suitable fit. '
      + 'Do not use placeholders. Do not exaggerate. Return ONLY the note text, no preamble.\n\n'
      + 'FACTS:\n' + facts;

    const gen = await genModel.generateContent(prompt);
    let note = gen.response.candidates[0].content.parts[0].text.trim();
    return res.json({ note });
  } catch (err) {
    return res.status(500).json({ error: 'Could not generate note', detail: err.message });
  }
});

// Schedule or record an interview for a submission
app.patch('/submissions/:id/interview', async (req, res) => {
  try {
    const { interview_date, interview_notes, outcome } = req.body;
    // Only allow scheduling once the candidate is SHORTLISTED (enforce order)
    const cur = await pool.query('SELECT status, candidate_id, candidate_name, job_id, job_title, client_name, submitted_by FROM submissions WHERE submission_id = $1', [req.params.id]);
    if (!cur.rows.length) return res.status(404).json({ error: 'Submission not found' });
    const sub = cur.rows[0];
    const schedulable = ['RECRUITER_CALL', 'HR_INTERVIEW', 'CLIENT_INTERVIEW', 'SHORTLISTED', 'INTERVIEW_SCHEDULED', 'SUBMITTED', 'CLIENT_CONFIRM', 'TRAINING'];
    if (!outcome && !schedulable.includes(sub.status)) {
      return res.status(400).json({ error: 'The candidate is not at a stage where an interview/call can be scheduled.' });
    }
    // Which round is this? (label for the emails)
    const roundLabel = sub.status === 'CLIENT_INTERVIEW' ? 'client interview' : sub.status === 'HR_INTERVIEW' ? 'HR interview' : sub.status === 'RECRUITER_CALL' ? 'recruiter call' : 'interview';

    // Generate a meeting link (Google Meet style) — do NOT change the stage; stages advance via the step buttons.
    const meetCode = Math.random().toString(36).slice(2, 5) + '-' + Math.random().toString(36).slice(2, 6) + '-' + Math.random().toString(36).slice(2, 5);
    const interview_link = 'https://meet.google.com/' + meetCode;

    // Hiring manager for this job (to keep everyone in the loop)
    let hm = '';
    try { hm = (await pool.query('SELECT hiring_manager FROM jobs WHERE job_id = $1', [sub.job_id])).rows[0]?.hiring_manager || ''; } catch (e) {}

    // Build the CANDIDATE email (link + notes) — returned so the UI can open mailto AND logged.
    const candEmail = candidateEmail(sub.candidate_id);
    const emailSubject = 'Your ' + roundLabel + ' is scheduled — ' + (sub.job_title || 'a role');
    const emailBody = 'Dear ' + (sub.candidate_name || 'Candidate') + ',\n\n'
      + 'Your ' + roundLabel + (sub.client_name ? ' with ' + sub.client_name : '') + ' for ' + (sub.job_title || 'a role') + ' has been scheduled' + (interview_date ? ' for ' + interview_date : '') + '.\n\n'
      + 'Join using this link:\n' + interview_link + '\n\n'
      + (interview_notes ? 'Details / notes:\n' + interview_notes + '\n\n' : '')
      + 'Please confirm your availability by replying to this email.\n\nBest regards,\nRecruit 360 Team';

    await pool.query(
      'UPDATE submissions SET interview_date = $1, interview_notes = $2, interview_link = $3, last_updated = NOW() WHERE submission_id = $4',
      [interview_date || null, interview_notes || '', interview_link, req.params.id]
    );

    // Record & notify: candidate (email), recruiter + hiring manager (update)
    const commArgs = { submission_id: req.params.id, candidate_id: sub.candidate_id, job_id: sub.job_id, sent_by: sub.submitted_by || 'recruiter' };
    await logComm({ ...commArgs, channel: 'EMAIL', to_role: 'candidate', to_name: sub.candidate_name, to_email: candEmail, subject: emailSubject, body: emailBody });
    const updateMsg = sub.candidate_name + '’s ' + roundLabel + ' for ' + (sub.job_title || 'a role') + (interview_date ? ' is set for ' + interview_date : ' has been scheduled') + '. Link: ' + interview_link;
    if (sub.submitted_by) { await notify({ candidate_id: sub.candidate_id, recipient: sub.submitted_by, type: 'INTERVIEW', message: updateMsg }); await logComm({ ...commArgs, channel: 'UPDATE', to_role: 'recruiter', to_name: sub.submitted_by, subject: 'Interview scheduled — ' + sub.candidate_name, body: updateMsg }); }
    if (hm) { await notify({ candidate_id: sub.candidate_id, recipient: hm, type: 'INTERVIEW', message: updateMsg }); await logComm({ ...commArgs, channel: 'UPDATE', to_role: 'hiring_manager', to_name: hm, subject: 'Interview scheduled — ' + sub.candidate_name, body: updateMsg }); }
    await notify({ candidate_id: sub.candidate_id, recipient: 'candidate', type: 'INTERVIEW', message: 'Your ' + roundLabel + ' is scheduled' + (interview_date ? ' on ' + interview_date : '') + '. Link: ' + interview_link });

    return res.json({ ok: true, submission_id: req.params.id, interview_link, email_sent: true, emailSubject, emailBody, candidateEmail: candEmail, notified: { recruiter: sub.submitted_by || null, hiring_manager: hm || null } });
  } catch (err) {
    return res.status(500).json({ error: 'Could not update interview', detail: err.message });
  }
});

// Record an offer decision for a submission
app.patch('/submissions/:id/offer', async (req, res) => {
  try {
    const { offer_status, offer_notes } = req.body;
    // offer_status: 'OFFERED' | 'OFFER_ACCEPTED' | 'OFFER_DECLINED'
    const allowed = ['OFFERED', 'OFFER_ACCEPTED', 'OFFER_DECLINED'];
    if (!allowed.includes(offer_status)) return res.status(400).json({ error: 'Invalid offer status', allowed });
    // Enforce: can only make an offer after the interview is done
    const ocur = await pool.query('SELECT status FROM submissions WHERE submission_id = $1', [req.params.id]);
    if (!ocur.rows.length) return res.status(404).json({ error: 'Submission not found' });
    if (offer_status === 'OFFERED' && ocur.rows[0].status !== 'INTERVIEWED') {
      return res.status(400).json({ error: 'An offer can only be made after the interview is completed.' });
    }
    if ((offer_status === 'OFFER_ACCEPTED' || offer_status === 'OFFER_DECLINED') && ocur.rows[0].status !== 'OFFERED') {
      return res.status(400).json({ error: 'The offer must be made before it can be accepted or declined.' });
    }
    await pool.query(
      'UPDATE submissions SET offer_status = $1, offer_notes = $2, status = $3, last_updated = NOW() WHERE submission_id = $4',
      [offer_status, offer_notes || '', offer_status, req.params.id]
    );
    const oRes = await pool.query('SELECT candidate_id, candidate_name, job_title, client_name, submitted_by FROM submissions WHERE submission_id = $1', [req.params.id]);
    const osub = oRes.rows[0];
    if (osub) {
      const label = offer_status === 'OFFER_ACCEPTED' ? 'accepted the offer' : offer_status === 'OFFER_DECLINED' ? 'declined the offer' : 'received an offer';
      await notify({ candidate_id: osub.candidate_id, recipient: osub.submitted_by, type: 'OFFER', message: osub.candidate_name + ' ' + label + ' for ' + (osub.job_title || 'a role') + '.' });
    }
    return res.json({ ok: true, submission_id: req.params.id, status: offer_status });
  } catch (err) {
    return res.status(500).json({ error: 'Could not update offer', detail: err.message });
  }
});

// Generate interview or offer notes (AI-assisted, grounded in candidate data)
app.post('/submissions/:id/generate-note', async (req, res) => {
  try {
    const { note_type } = req.body; // 'interview' or 'offer'
    const sRes = await pool.query('SELECT candidate_id, candidate_name, job_title, client_name, status FROM submissions WHERE submission_id = $1', [req.params.id]);
    if (!sRes.rows.length) return res.status(404).json({ error: 'Submission not found' });
    const sub = sRes.rows[0];
    const cRes = await pool.query('SELECT role, destination_country, visa_status, experience_years FROM candidates WHERE candidate_id = $1', [sub.candidate_id]);
    const c = cRes.rows[0] || {};

    const facts = [
      'Candidate: ' + sub.candidate_name,
      'Role: ' + (c.role || sub.job_title || 'not specified'),
      'Job: ' + (sub.job_title || 'not specified'),
      'Client: ' + (sub.client_name || 'not specified'),
      'Destination: ' + (c.destination_country || 'not specified'),
      'Experience: ' + (c.experience_years != null ? c.experience_years + ' years' : 'not specified'),
      'Visa status: ' + (c.visa_status || 'not specified').replace(/_/g, ' '),
    ].join('\n');

    let prompt;
    if (note_type === 'client') {
      prompt = 'You are a recruitment coordinator writing a short, professional submission note to a client about a candidate. '
        + 'Use ONLY the facts below - do not invent skills, qualifications, or details. Write 2-3 sentences explaining why this candidate is a suitable fit for the role. '
        + 'Return ONLY the note text, no preamble.\n\nFACTS:\n' + facts;
    } else if (note_type === 'offer') {
      prompt = 'You are a recruitment coordinator writing a short, professional internal note about an offer being made to a candidate. '
        + 'Use ONLY the facts below. Write 2-3 sentences covering the role, client, and that an offer is being extended, with a professional positive tone. '
        + 'Do not invent salary, dates, or terms. Return ONLY the note text.\n\nFACTS:\n' + facts;
    } else {
      prompt = 'You are a recruitment coordinator writing short, professional interview notes / talking points for an upcoming interview between a candidate and a client. '
        + 'Use ONLY the facts below. Write 2-3 sentences summarising the candidate\'s fit for the role and what to highlight in the interview. '
        + 'Do not invent skills or details not provided. Return ONLY the note text.\n\nFACTS:\n' + facts;
    }

    const gen = await genModel.generateContent(prompt);
    const note = gen.response.candidates[0].content.parts[0].text.trim();
    return res.json({ note });
  } catch (err) {
    return res.status(500).json({ error: 'Could not generate note', detail: err.message });
  }
});

// ---------- JOB-CENTRIC SUBMISSIONS (sir's request) ----------
// Given a job, return its submissions + auto-suggested candidates for that role/destination
app.get('/jobs/:jobId/board', async (req, res) => {
  try {
    // 1. The job
    const jRes = await pool.query('SELECT job_id, title, client, location, openings FROM jobs WHERE job_id = $1', [req.params.jobId]);
    if (!jRes.rows.length) return res.status(404).json({ error: 'Job not found' });
    const job = jRes.rows[0];

    // 2. Its submissions
    const subRes = await pool.query(
      'SELECT submission_id, candidate_id, candidate_name, status, screening_notes, submitted_by, resume_received, resume_requested, interview_link, created_at FROM submissions WHERE job_id = $1 ORDER BY created_at DESC',
      [req.params.jobId]
    );
    const submissions = subRes.rows;
    const alreadySubmitted = new Set(submissions.map(s => s.candidate_id));

    // 3. Auto-suggest candidates for this job, excluding already-submitted.
    // Match primarily on ROLE; rank same-destination-country candidates first for a truer score.
    let suggestions = [];
    try {
      // Map common cities to their country so 'Stockholm' -> 'Sweden' etc.
      const cityToCountry = {
        stockholm: 'Sweden', gothenburg: 'Sweden', malmo: 'Sweden',
        berlin: 'Germany', munich: 'Germany', frankfurt: 'Germany', hamburg: 'Germany',
        amsterdam: 'Netherlands', rotterdam: 'Netherlands', eindhoven: 'Netherlands',
        dublin: 'Ireland', cork: 'Ireland',
        paris: 'France', lyon: 'France',
      };
      const loc = (job.location || '').trim();
      const country = cityToCountry[loc.toLowerCase()] || loc; // if already a country, keep it
      // Query text focuses on the ROLE (what actually drives the match)
      const q = job.title;
      const sql = 'SELECT h.base.candidate_id AS candidate_id, c.full_name, c.role, c.destination_country, c.visa_status, c.experience_years, h.distance '
        + 'FROM VECTOR_SEARCH(TABLE `direct-tribute-502305-q5.recruit360.candidate_embeddings`, \'embedding\', '
        + '(SELECT ml_generate_embedding_result AS embedding FROM ML.GENERATE_EMBEDDING(MODEL `direct-tribute-502305-q5.recruit360.text_embedder`, (SELECT @q AS content))), top_k => 25) AS h '
        + 'JOIN `direct-tribute-502305-q5.recruit360.candidates` c ON c.candidate_id = h.base.candidate_id ORDER BY h.distance';
      const [bqRows] = await bq.query({ query: sql, location: 'asia-south1', params: { q } });
      const MATCH_THRESHOLD = 0.92;
      const norm = (s) => (s || '').toLowerCase().trim().replace(/s$/, '');
      const roleMatches = (cand) => { const a = norm(cand), b = norm(job.title); return a && b && (a === b || a.includes(b) || b.includes(a)); };

      suggestions = bqRows
        .filter(r => r.distance <= MATCH_THRESHOLD)
        .filter(r => !alreadySubmitted.has(r.candidate_id))
        .map(r => {
          const sameCountry = (r.destination_country || '').toLowerCase() === country.toLowerCase();
          const sameRole = roleMatches(r.role);
          // Compute a fit score: base from role match, bonus for same country, plus a little from distance
          let fit = 0;
          if (sameRole) fit += 70;
          if (sameCountry) fit += 25;
          fit += Math.round((1 - r.distance) * 5); // small nudge from semantic closeness
          if (fit > 99) fit = 99;
          return {
            candidate_id: r.candidate_id, full_name: r.full_name, role: r.role,
            destination_country: r.destination_country, visa_status: r.visa_status,
            experience_years: r.experience_years,
            match: fit, sameCountry, sameRole,
          };
        })
        // Prefer same role, then same country, then higher fit
        .sort((a, b) => (b.sameRole - a.sameRole) || (b.sameCountry - a.sameCountry) || (b.match - a.match))
        .slice(0, 8);
    } catch (e) { suggestions = []; }

    // Attach a quick history summary to each suggestion (how many times submitted/rejected before)
    try {
      const ids = suggestions.map(s => s.candidate_id);
      if (ids.length) {
        const hRes = await pool.query(
          "SELECT candidate_id, COUNT(*) AS total, COUNT(*) FILTER (WHERE status IN ('REJECTED','OFFER_DECLINED')) AS rejected FROM submissions WHERE candidate_id = ANY($1) GROUP BY candidate_id",
          [ids]
        );
        const hMap = {};
        hRes.rows.forEach(r => { hMap[r.candidate_id] = { total: parseInt(r.total,10), rejected: parseInt(r.rejected,10) }; });
        suggestions = suggestions.map(s => ({
          ...s,
          prior_submissions: (hMap[s.candidate_id] && hMap[s.candidate_id].total) || 0,
          prior_rejections: (hMap[s.candidate_id] && hMap[s.candidate_id].rejected) || 0,
          repeat_reject: !!(hMap[s.candidate_id] && hMap[s.candidate_id].rejected >= 2),
        }));
      }
    } catch (e) { /* history is best-effort */ }

    return res.json({ job, submissions, suggestions });
  } catch (err) {
    return res.status(500).json({ error: 'Could not load job board', detail: err.message });
  }
});

// Manually add a candidate (by id) as a submission to a job — bypasses AI sourcing
app.post('/jobs/:jobId/submit-candidate', async (req, res) => {
  try {
    const { candidate_id, submitted_by, screening_notes, current_ctc, expected_ctc } = req.body;
    if (!candidate_id) return res.status(400).json({ error: 'candidate_id is required' });
    const jRes = await pool.query('SELECT job_id, title, client, hiring_manager FROM jobs WHERE job_id = $1', [req.params.jobId]);
    if (!jRes.rows.length) return res.status(404).json({ error: 'Job not found' });
    const job = jRes.rows[0];
    const cRes = await pool.query('SELECT full_name FROM candidates WHERE candidate_id = $1', [candidate_id]);
    if (!cRes.rows.length) return res.status(404).json({ error: 'Candidate not found' });
    const dup = await pool.query('SELECT submission_id FROM submissions WHERE candidate_id = $1 AND job_id = $2', [candidate_id, req.params.jobId]);
    if (dup.rows.length) return res.status(409).json({ error: 'duplicate', message: 'This candidate is already submitted to this job.' });
    // Ensure CTC columns exist
    try { await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS current_ctc TEXT'); await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS expected_ctc TEXT'); } catch(e){}
    const submission_id = 'SUB' + Date.now().toString().slice(-10);
    // Sir's flow: start at PENDING_HM_APPROVAL — the hiring manager must approve before the candidate is emailed.
    await pool.query(
      'INSERT INTO submissions (submission_id, candidate_id, candidate_name, job_id, job_title, client_name, status, screening_notes, submitted_by, current_ctc, expected_ctc, created_at, last_updated) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,NOW(),NOW())',
      [submission_id, candidate_id, cRes.rows[0].full_name, req.params.jobId, job.title, job.client || '', 'PENDING_HM_APPROVAL', screening_notes || '', submitted_by || '', current_ctc || '', expected_ctc || '']
    );
    // Notify the hiring manager
    try { await pool.query('INSERT INTO notifications (message, type, read, created_at) VALUES ($1,$2,FALSE,NOW())', ['Approval needed: ' + cRes.rows[0].full_name + ' submitted for ' + job.title, 'APPROVAL']); } catch(e){}
    return res.json({ ok: true, submission_id, candidate_name: cRes.rows[0].full_name, hiring_manager: job.hiring_manager, status: 'PENDING_HM_APPROVAL' });
  } catch (err) {
    return res.status(500).json({ error: 'Could not submit candidate', detail: err.message });
  }
});

// ---------- HIRING MANAGER: approve or reject a submission ----------
app.patch('/submissions/:id/approval', async (req, res) => {
  try {
    const { decision, approver } = req.body; // APPROVED | REJECTED
    const sub = (await pool.query('SELECT candidate_id, candidate_name, job_id, job_title, client_name, submitted_by FROM submissions WHERE submission_id = $1', [req.params.id])).rows[0];
    if (!sub) return res.status(404).json({ error: 'not found' });
    const candEmail = candidateEmail(sub.candidate_id);
    const cArgs = { submission_id: req.params.id, candidate_id: sub.candidate_id, job_id: sub.job_id, sent_by: approver || 'hiring_manager' };

    if (decision !== 'APPROVED') {
      await pool.query("UPDATE submissions SET status = 'REJECTED', last_updated = NOW() WHERE submission_id = $1", [req.params.id]);
      await notify({ candidate_id: sub.candidate_id, recipient: sub.submitted_by || 'recruiter', type: 'CANDIDATE_REJECTED', message: `${sub.candidate_name} was not approved for ${sub.job_title || 'a role'}.` });
      return res.json({ ok: true, status: 'REJECTED' });
    }

    // Approved -> candidate is shortlisted. Move to the recruiter-call stage and send the
    // shortlisting email to the candidate (naming the recruiter who will guide them).
    await pool.query("UPDATE submissions SET status = 'RECRUITER_CALL', last_updated = NOW() WHERE submission_id = $1", [req.params.id]);
    const recruiterName = sub.submitted_by || 'your recruiter';
    const emailSubject = 'You have been shortlisted — ' + (sub.job_title || 'a role');
    const emailBody = 'Dear ' + sub.candidate_name + ',\n\nCongratulations! You have been shortlisted for the next rounds for ' + (sub.job_title || 'a role') + (sub.client_name ? ' at ' + sub.client_name : '') + '. ' + recruiterName + ' will guide you through the upcoming interviews and next steps, and will be in touch shortly to arrange your recruiter call.\n\nBest regards,\nRecruit 360 Team';
    await logComm({ ...cArgs, channel: 'EMAIL', to_role: 'candidate', to_name: sub.candidate_name, to_email: candEmail, subject: emailSubject, body: emailBody });
    await notify({ candidate_id: sub.candidate_id, recipient: 'candidate', type: 'SHORTLISTED', message: 'You have been shortlisted for ' + (sub.job_title || 'a role') + '.' });
    await notify({ candidate_id: sub.candidate_id, recipient: sub.submitted_by || 'recruiter', type: 'APPROVED', message: `${sub.candidate_name} approved for ${sub.job_title || 'a role'} — schedule the recruiter call.` });
    return res.json({ ok: true, status: 'RECRUITER_CALL', candidate: sub, emailSubject, emailBody, candidateEmail: candEmail });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- Submissions PENDING approval (for the hiring manager) ----------
app.get('/submissions/pending-approval', async (req, res) => {
  try {
    await ensureSlots();
    // A hiring manager sees only approvals for their own jobs; admin sees all.
    const hm = String(req.query.hm || '').trim();
    const scoped = String(req.query.role || '') !== 'admin' && hm;
    const { rows } = await pool.query(
      `SELECT s.submission_id, s.candidate_id, s.candidate_name, s.job_id, s.job_title, s.client_name, s.current_ctc, s.expected_ctc, s.resume_summary, s.submitted_by, s.created_at
         FROM submissions s LEFT JOIN jobs j ON j.job_id = s.job_id
        WHERE s.status = 'PENDING_HM_APPROVAL' ${scoped ? 'AND j.hiring_manager = $1' : ''}
        ORDER BY s.created_at DESC`, scoped ? [hm] : []);
    return res.json({ pending: rows });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- CANDIDATE HISTORY across jobs (sir's request: avoid repeat-rejects) ----------
app.get('/candidates/:id/history', async (req, res) => {
  try {
    const { rows } = await pool.query(
      'SELECT submission_id, job_id, job_title, client_name, status, created_at FROM submissions WHERE candidate_id = $1 ORDER BY created_at DESC',
      [req.params.id]
    );
    // Summarise
    const total = rows.length;
    const rejected = rows.filter(r => r.status === 'REJECTED' || r.status === 'OFFER_DECLINED').length;
    const active = rows.filter(r => !['REJECTED', 'OFFER_DECLINED', 'PLACED'].includes(r.status)).length;
    const placed = rows.filter(r => r.status === 'PLACED').length;
    // A simple flag: repeatedly rejected
    const repeatReject = rejected >= 2;
    return res.json({ candidate_id: req.params.id, total, rejected, active, placed, repeatReject, history: rows });
  } catch (err) {
    return res.status(500).json({ error: 'Could not fetch candidate history', detail: err.message });
  }
});

// ---------- CANDIDATE EMAIL ----------
// Demo mode: if DEMO_EMAIL is set, every candidate email routes to that one real
// inbox so you can show real delivery live. Otherwise it derives the placeholder.
function candidateEmail(candidate_id) {
  if (process.env.DEMO_EMAIL) return process.env.DEMO_EMAIL;
  return (candidate_id || '').toLowerCase() + '@example.com';
}

// ---------- NOTIFICATIONS ----------
async function notify({ candidate_id, recipient, type, message }) {
  try {
    const notif_id = 'NT' + Date.now().toString().slice(-10) + Math.floor(Math.random()*100);
    await pool.query(
      'INSERT INTO notifications (notif_id, candidate_id, recipient, type, message, created_at) VALUES ($1,$2,$3,$4,$5,NOW())',
      [notif_id, candidate_id || null, recipient || '', type || 'INFO', message]
    );
  } catch (e) { /* notifications are best-effort, never block the main action */ }
}

// ---------- COMMUNICATIONS LOG ----------
// Every email/update in the recruitment flow is recorded here so the full
// communication trail (to candidate, recruiter, hiring manager, client) is
// visible and demonstrable in the app.
// ---------- REAL EMAIL (SendGrid) ----------
// Gated on SENDGRID_API_KEY. No key => demo mode (nothing sends; UI/logs unchanged).
// MAIL_FROM = your verified SendGrid sender. MAIL_REDIRECT (optional) = send every email to this
// one inbox during testing (so sir receives them all) while the app still shows the real recipient.
async function sendEmail({ to, cc, subject, body }) {
  try {
    const key = process.env.SENDGRID_API_KEY;
    const from = process.env.MAIL_FROM;
    if (!key || !from || !to) return { sent: false, reason: 'not-configured' };
    if (typeof fetch !== 'function') return { sent: false, reason: 'no-fetch' };
    const redirect = process.env.MAIL_REDIRECT; // optional single test inbox
    const realTo = redirect || to;
    const subj = redirect ? `[to: ${to}] ${subject}` : subject;
    const personalization = { to: [{ email: realTo }] };
    if (!redirect && cc) {
      const ccList = String(cc).split(',').map(s => s.trim()).filter(e => /.+@.+\..+/.test(e)).map(email => ({ email }));
      if (ccList.length) personalization.cc = ccList;
    }
    const payload = {
      personalizations: [personalization],
      from: { email: from, name: process.env.MAIL_FROM_NAME || 'Recruit 360' },
      subject: subj || '(no subject)',
      content: [{ type: 'text/plain', value: body || '' }],
    };
    const r = await fetch('https://api.sendgrid.com/v3/mail/send', {
      method: 'POST',
      headers: { Authorization: 'Bearer ' + key, 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (r.status >= 200 && r.status < 300) return { sent: true };
    const errTxt = await r.text().catch(() => '');
    console.log('SendGrid error', r.status, errTxt.slice(0, 300));
    return { sent: false, reason: 'sendgrid-' + r.status };
  } catch (e) { console.log('sendEmail warn:', e.message); return { sent: false, reason: e.message }; }
}

async function logComm({ submission_id, candidate_id, job_id, channel, to_role, to_name, to_email, subject, body, sent_by, cc }) {
  try {
    // Resolve the REAL candidate email so emails reach the actual person (what the recruiter entered),
    // not a demo inbox. Falls back to whatever was passed if no real address is on file.
    let dest = to_email || '';
    if ((channel || 'EMAIL') === 'EMAIL' && to_role === 'candidate' && candidate_id) {
      try {
        const r = (await pool.query('SELECT email FROM candidates WHERE candidate_id = $1', [candidate_id])).rows[0];
        if (r && r.email && /.+@.+\..+/.test(r.email) && !/@example\.com$/i.test(r.email)) dest = r.email;
      } catch (e) {}
    }
    const comm_id = 'CM' + Date.now().toString().slice(-10) + Math.floor(Math.random()*1000);
    await pool.query(
      `INSERT INTO communications (comm_id, submission_id, candidate_id, job_id, channel, to_role, to_name, to_email, subject, body, sent_by, created_at)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,NOW())`,
      [comm_id, submission_id || null, candidate_id || null, job_id || null, channel || 'EMAIL', to_role || '', to_name || '', dest || '', subject || '', body || '', sent_by || 'system']
    );
    // Real send for actual emails (best-effort; never blocks the workflow).
    if ((channel || 'EMAIL') === 'EMAIL' && dest) { sendEmail({ to: dest, cc, subject, body }).catch(() => {}); }
    return comm_id;
  } catch (e) { /* best-effort, never block the main action */ return null; }
}

// Test the email configuration: GET /mail/test?to=you@example.com
app.get('/mail/test', async (req, res) => {
  try {
    const to = req.query.to || process.env.MAIL_FROM;
    const configured = !!(process.env.SENDGRID_API_KEY && process.env.MAIL_FROM);
    if (!configured) return res.json({ configured: false, message: 'Set SENDGRID_API_KEY and MAIL_FROM env vars to enable real email.' });
    const r = await sendEmail({ to, subject: 'Recruit 360 — email test', body: 'This is a test email from Recruit 360. If you received this, SendGrid is wired correctly.' });
    return res.json({ configured: true, to, redirectedTo: process.env.MAIL_REDIRECT || null, result: r });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// List the communication trail for a submission (or a whole job)
app.get('/communications', async (req, res) => {
  try {
    const { submission_id, job_id } = req.query;
    let sql = 'SELECT comm_id, submission_id, candidate_id, job_id, channel, to_role, to_name, to_email, subject, body, sent_by, created_at FROM communications';
    const params = []; const where = [];
    if (submission_id) { params.push(submission_id); where.push('submission_id = $' + params.length); }
    if (job_id) { params.push(job_id); where.push('job_id = $' + params.length); }
    if (where.length) sql += ' WHERE ' + where.join(' AND ');
    sql += ' ORDER BY created_at ASC LIMIT 200';
    const { rows } = await pool.query(sql, params);
    return res.json({ communications: rows });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// List notifications (optionally for a recipient)
app.get('/notifications', async (req, res) => {
  try {
    const { recipient } = req.query;
    let sql = 'SELECT notif_id, candidate_id, recipient, type, message, read_flag, created_at FROM notifications';
    const params = [];
    if (recipient) { sql += ' WHERE recipient = $1'; params.push(recipient); }
    sql += ' ORDER BY created_at DESC LIMIT 50';
    const { rows } = await pool.query(sql, params);
    const unread = rows.filter(r => !r.read_flag).length;
    return res.json({ notifications: rows, unread });
  } catch (err) {
    return res.status(500).json({ error: 'Could not fetch notifications', detail: err.message });
  }
});

// Mark all as read
app.patch('/notifications/read', async (req, res) => {
  try {
    const { recipient } = req.body;
    if (recipient) await pool.query('UPDATE notifications SET read_flag = TRUE WHERE recipient = $1', [recipient]);
    else await pool.query('UPDATE notifications SET read_flag = TRUE');
    return res.json({ ok: true });
  } catch (err) {
    return res.status(500).json({ error: 'Could not update notifications', detail: err.message });
  }
});

// ---------- USER SETTINGS (preferences incl. email notifications toggle) ----------
app.get('/settings', async (req, res) => {
  try {
    const { user } = req.query;
    if (!user) return res.json({ settings: {} });
    const { rows } = await pool.query('SELECT prefs FROM user_settings WHERE user_key = $1', [user]);
    return res.json({ settings: rows.length ? rows[0].prefs : {} });
  } catch (e) { return res.json({ settings: {} }); }
});
app.patch('/settings', async (req, res) => {
  try {
    const { user, prefs } = req.body;
    if (!user) return res.status(400).json({ error: 'user required' });
    await pool.query(
      `INSERT INTO user_settings (user_key, prefs, updated_at) VALUES ($1,$2,NOW())
       ON CONFLICT (user_key) DO UPDATE SET prefs = EXCLUDED.prefs, updated_at = NOW()`,
      [user, JSON.stringify(prefs || {})]
    );
    return res.json({ ok: true });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- CREATE a full job requisition (all mandatory fields) ----------
app.post('/jobs/requisition', async (req, res) => {
  try {
    const b = req.body;
    if (!b.title || !b.client || !b.hiring_manager) {
      return res.status(400).json({ error: 'Title, Client and Hiring Manager are required.' });
    }
    const job_id = b.job_id || ('JOB' + Date.now().toString().slice(-7));
    const job_code = b.job_code || ('JC-' + Date.now().toString().slice(-6));
    const q = `INSERT INTO jobs (
        job_id, title, job_code, client, hiring_manager, recruiter, recruitment_manager,
        description, primary_skills, job_location, location, country, zip_code,
        job_start_date, job_end_date, number_of_positions, openings,
        bill_rate, bill_rate_type, tax_terms, priority, status, owner, created_by, created_date
      ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24, NOW())
      ON CONFLICT (job_id) DO UPDATE SET
        title=EXCLUDED.title, job_code=EXCLUDED.job_code, client=EXCLUDED.client,
        hiring_manager=EXCLUDED.hiring_manager, recruiter=EXCLUDED.recruiter,
        recruitment_manager=EXCLUDED.recruitment_manager, description=EXCLUDED.description,
        primary_skills=EXCLUDED.primary_skills, job_location=EXCLUDED.job_location,
        location=EXCLUDED.location, country=EXCLUDED.country, zip_code=EXCLUDED.zip_code,
        job_start_date=EXCLUDED.job_start_date, job_end_date=EXCLUDED.job_end_date,
        number_of_positions=EXCLUDED.number_of_positions, openings=EXCLUDED.openings,
        bill_rate=EXCLUDED.bill_rate, bill_rate_type=EXCLUDED.bill_rate_type,
        tax_terms=EXCLUDED.tax_terms, priority=EXCLUDED.priority, status=EXCLUDED.status,
        owner=EXCLUDED.owner
      RETURNING *`;
    const vals = [
      job_id, b.title, job_code, b.client, b.hiring_manager, b.recruiter || '', b.recruitment_manager || '',
      b.description || '', b.primary_skills || '', b.job_location || '', b.job_location || b.location || '',
      b.country || '', b.zip_code || '',
      b.job_start_date || null, b.job_end_date || null,
      parseInt(b.number_of_positions || 1, 10), parseInt(b.number_of_positions || 1, 10),
      b.bill_rate ? Number(b.bill_rate) : null, b.bill_rate_type || '', b.tax_terms || '',
      b.priority || 'Medium', b.status || 'Open', b.owner || b.recruiter || '', b.created_by || b.hiring_manager || '',
    ];
    const { rows } = await pool.query(q, vals);
    return res.json({ ok: true, job: rows[0] });
  } catch (e) {
    return res.status(500).json({ error: 'Could not create requisition', detail: e.message });
  }
});

// ---------- SINGLE-SCREEN JOB VIEW: everything for one job on one screen ----------
app.get('/jobs/:jobId/full', async (req, res) => {
  try {
    // 1) The job with all fields + ageing
    const jRes = await pool.query(
      `SELECT *, (CURRENT_DATE - created_date::date) AS ageing_days FROM jobs WHERE job_id = $1`,
      [req.params.jobId]
    );
    if (!jRes.rows.length) return res.status(404).json({ error: 'Job not found' });
    const job = jRes.rows[0];

    // 2) Submissions for this job (resume_received included so the row shows it)
    const sRes = await pool.query(
      `SELECT submission_id, candidate_id, candidate_name, status, screening_notes, submitted_by, resume_received, resume_requested, created_at
         FROM submissions WHERE job_id = $1 ORDER BY created_at DESC`,
      [req.params.jobId]
    );

    // 3) Interviews — any submission that has entered the interview stages
    const iRes = await pool.query(
      `SELECT submission_id, candidate_name, interview_date, interview_notes, status
         FROM submissions WHERE job_id = $1
           AND status IN ('RECRUITER_CALL','HR_INTERVIEW','CLIENT_INTERVIEW','INTERVIEW_SCHEDULED','INTERVIEWED')
         ORDER BY interview_date DESC NULLS LAST, created_at DESC`,
      [req.params.jobId]
    );

    // 4) Placements — placed candidates + those awaiting the HM's placement approval
    const pRes = await pool.query(
      `SELECT submission_id, candidate_name, status, created_at
         FROM submissions WHERE job_id = $1
           AND status IN ('PENDING_PLACEMENT_APPROVAL','OFFER','OFFER_ACCEPTED','PLACED')
         ORDER BY created_at DESC`,
      [req.params.jobId]
    );

    return res.json({
      job,
      submissions: sRes.rows,
      interviews: iRes.rows,
      placements: pRes.rows,
      counts: {
        submissions: sRes.rows.length,
        interviews: iRes.rows.length,
        placements: pRes.rows.length,
      },
    });
  } catch (e) {
    return res.status(500).json({ error: 'Could not load job', detail: e.message });
  }
});

// ---------- CLIENTS (admin manages; dropdown for HM & recruiter) ----------
app.get('/admin/clients', async (req, res) => {
  try { const { rows } = await pool.query('SELECT * FROM clients WHERE active = TRUE ORDER BY client_name'); return res.json({ clients: rows }); }
  catch (e) { return res.status(500).json({ error: e.message }); }
});
app.post('/admin/clients', async (req, res) => {
  try {
    const b = req.body;
    if (!b.client_name || !b.client_name.trim()) return res.status(400).json({ error: 'client_name required' });
    const client_id = b.client_id || ('CL-' + Date.now().toString().slice(-6));
    await pool.query(
      `INSERT INTO clients (client_id, client_name, industry, contact_person, contact_email, contact_phone, country, created_by)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT (client_id) DO NOTHING`,
      [client_id, b.client_name, b.industry || '', b.contact_person || '', b.contact_email || '', b.contact_phone || '', b.country || '', b.created_by || 'Admin']
    );
    return res.json({ ok: true, client_id });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- USERS (admin sees all HMs & recruiters) ----------
app.get('/admin/users', async (req, res) => {
  try {
    const { role } = req.query;
    let sql = 'SELECT user_id, full_name, email, role, user_group, phone, active FROM app_users WHERE active = TRUE';
    const params = [];
    if (role) { sql += ' AND role = $1'; params.push(role); }
    sql += ' ORDER BY role, full_name';
    const { rows } = await pool.query(sql, params);
    return res.json({ users: rows });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});
app.post('/admin/users', async (req, res) => {
  try {
    await ensureAuth();
    const b = req.body;
    if (!b.full_name || !b.full_name.trim() || !b.role) return res.status(400).json({ error: 'full_name and role required' });
    const user_id = b.user_id || ('USR-' + Date.now().toString().slice(-6));
    // Give every new user a login password so they can sign in and (for recruiters) appear in the assign dropdown.
    const pw = b.password || process.env.SEED_PASSWORD || 'Avanciers@360';
    const { salt, hash } = hashPassword(pw);
    await pool.query(
      `INSERT INTO app_users (user_id, full_name, email, role, user_group, phone, active, password_hash, password_salt)
       VALUES ($1,$2,$3,$4,$5,$6,TRUE,$7,$8)
       ON CONFLICT (user_id) DO UPDATE SET full_name = EXCLUDED.full_name, email = EXCLUDED.email, role = EXCLUDED.role, active = TRUE`,
      [user_id, b.full_name, b.email || '', b.role, b.user_group || '', b.phone || '', hash, salt]
    );
    return res.json({ ok: true, user_id, email: b.email || '', password: pw });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- AUTH: single login, role resolved from the email ----------
app.post('/auth/login', async (req, res) => {
  try {
    await ensureAuth();
    const email = String(req.body?.email || '').trim().toLowerCase();
    const password = String(req.body?.password || '');
    if (!email || !password) return res.status(400).json({ error: 'Email and password are required.' });
    const { rows } = await pool.query(
      `SELECT user_id, full_name, email, role, password_hash, password_salt FROM app_users
        WHERE LOWER(email) = $1 AND active = TRUE
        ORDER BY (password_hash IS NOT NULL) DESC, created_at DESC LIMIT 1`, [email]);
    const u = rows[0];
    if (!u || !u.password_hash || !verifyPassword(password, u.password_salt, u.password_hash)) {
      return res.status(401).json({ error: 'Invalid email or password.' });
    }
    const initials = String(u.full_name || u.email).split(/\s+/).map(s => s[0]).join('').slice(0, 2).toUpperCase();
    return res.json({ user: { name: u.full_name, email: u.email, role: u.role, userId: u.user_id, initials } });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- AUTH: list users (admin) ----------
app.get('/auth/users', async (req, res) => {
  try {
    await ensureAuth();
    const { rows } = await pool.query("SELECT user_id, full_name, email, role, active FROM app_users WHERE active = TRUE ORDER BY role, full_name");
    return res.json({ users: rows });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// Recruiters list (for the assign dropdown)
app.get('/recruiters', async (req, res) => {
  try { await ensureAuth(); const { rows } = await pool.query("SELECT user_id, full_name, email FROM app_users WHERE role='recruiter' AND active=TRUE ORDER BY full_name"); return res.json({ recruiters: rows }); }
  catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- VISA OFFICES ----------
app.get('/admin/visa-offices', async (req, res) => {
  try { const { rows } = await pool.query('SELECT * FROM visa_offices ORDER BY office_name'); return res.json({ offices: rows }); }
  catch (e) { return res.status(500).json({ error: e.message }); }
});
app.post('/admin/visa-offices', async (req, res) => {
  try { const b = req.body; if (!b.office_name || !b.office_name.trim()) return res.status(400).json({ error: 'office_name required' }); const id = b.office_id || ('VO-' + Date.now().toString().slice(-6));
    await pool.query('INSERT INTO visa_offices (office_id, office_name, country, address, contact) VALUES ($1,$2,$3,$4,$5) ON CONFLICT DO NOTHING', [id, b.office_name, b.country||'', b.address||'', b.contact||'']);
    return res.json({ ok: true, office_id: id }); } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- TRAINING CENTRES ----------
app.get('/admin/training-centres', async (req, res) => {
  try { const { rows } = await pool.query('SELECT * FROM training_centres ORDER BY centre_name'); return res.json({ centres: rows }); }
  catch (e) { return res.status(500).json({ error: e.message }); }
});
app.post('/admin/training-centres', async (req, res) => {
  try { const b = req.body; if (!b.centre_name || !b.centre_name.trim()) return res.status(400).json({ error: 'centre_name required' }); const id = b.centre_id || ('TC-' + Date.now().toString().slice(-6));
    await pool.query('INSERT INTO training_centres (centre_id, centre_name, location, focus_area, capacity) VALUES ($1,$2,$3,$4,$5) ON CONFLICT DO NOTHING', [id, b.centre_name, b.location||'', b.focus_area||'', b.capacity||null]);
    return res.json({ ok: true, centre_id: id }); } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- JOB ROLES + SKILLS ----------
app.get('/admin/job-roles', async (req, res) => {
  try { const { rows } = await pool.query('SELECT * FROM job_roles ORDER BY role_name'); return res.json({ roles: rows }); }
  catch (e) { return res.status(500).json({ error: e.message }); }
});
app.post('/admin/job-roles', async (req, res) => {
  try { const b = req.body; if (!b.role_name || !b.role_name.trim()) return res.status(400).json({ error: 'role_name required' }); const id = b.role_id || ('JR-' + Date.now().toString().slice(-6));
    await pool.query('INSERT INTO job_roles (role_id, role_name, skills) VALUES ($1,$2,$3) ON CONFLICT DO NOTHING', [id, b.role_name, b.skills||'']);
    return res.json({ ok: true, role_id: id }); } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- ASSIGN a job to a recruiter ----------
app.patch('/jobs/:jobId/assign', async (req, res) => {
  try {
    const { recruiter_id, recruiter_name, recruiter_ids, recruiter_names } = req.body;
    try { await pool.query('ALTER TABLE jobs ADD COLUMN IF NOT EXISTS assigned_recruiter_ids TEXT'); await pool.query('ALTER TABLE jobs ADD COLUMN IF NOT EXISTS assigned_recruiter_names TEXT'); } catch(e){}
    if (recruiter_ids) {
      // Multiple recruiters (comma-separated)
      await pool.query('UPDATE jobs SET assigned_recruiter_ids = $1, assigned_recruiter_names = $2, assigned_recruiter_id = $3, recruiter = $4 WHERE job_id = $5',
        [recruiter_ids, recruiter_names || '', (recruiter_ids.split(',')[0] || ''), (recruiter_names || '').split(',')[0] || '', req.params.jobId]);
    } else {
      await pool.query('UPDATE jobs SET assigned_recruiter_id = $1, recruiter = $2, assigned_recruiter_ids = $1, assigned_recruiter_names = $2 WHERE job_id = $3', [recruiter_id, recruiter_name || '', req.params.jobId]);
    }
    return res.json({ ok: true });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- EDIT a job ----------
app.patch('/jobs/:jobId/edit', async (req, res) => {
  try {
    const b = req.body;
    const fields = ['title','job_code','client','hiring_manager','description','primary_skills','job_location','country','zip_code','number_of_positions','bill_rate','bill_rate_type','tax_terms','priority','status'];
    const sets = []; const vals = []; let i = 1;
    fields.forEach(f => { if (b[f] !== undefined) { sets.push(f + ' = $' + i); vals.push(b[f]); i++; } });
    if (!sets.length) return res.json({ ok: true });
    vals.push(req.params.jobId);
    await pool.query('UPDATE jobs SET ' + sets.join(', ') + ' WHERE job_id = $' + i, vals);
    return res.json({ ok: true });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- RECRUITER'S OWN JOBS (only jobs assigned to them) ----------
app.get('/recruiters/:id/jobs', async (req, res) => {
  try {
    // Match by assigned_recruiter_id OR by recruiter name (for jobs assigned before IDs existed)
    const nameRes = await pool.query('SELECT full_name FROM app_users WHERE user_id = $1', [req.params.id]);
    const name = nameRes.rows.length ? nameRes.rows[0].full_name : '';
    const { rows } = await pool.query(
      `SELECT job_id, title, job_code, client, country, job_location, number_of_positions, priority, status, created_date, recruiter,
              (CURRENT_DATE - created_date::date) AS ageing_days
         FROM jobs WHERE assigned_recruiter_id = $1 OR recruiter = $2 ORDER BY created_date DESC`,
      [req.params.id, name]
    );
    return res.json({ jobs: rows, recruiter_name: name });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- ROLE-BASED REPORTS (3 different) ----------
app.get('/reports/:role', async (req, res) => {
  try {
    const role = req.params.role;
    const jobsTotal = await pool.query('SELECT COUNT(*) c FROM jobs');
    const jobsOpen = await pool.query("SELECT COUNT(*) c FROM jobs WHERE status IN ('Open','Active','In Process')");
    const subs = await pool.query('SELECT COUNT(*) c FROM submissions');
    const placed = await pool.query("SELECT COUNT(*) c FROM submissions WHERE status='PLACED'");

    if (role === 'admin') {
      const byRecruiter = await pool.query(`SELECT recruiter, COUNT(*) jobs FROM jobs WHERE recruiter IS NOT NULL GROUP BY recruiter ORDER BY jobs DESC LIMIT 10`);
      const byClient = await pool.query(`SELECT client, COUNT(*) jobs FROM jobs GROUP BY client ORDER BY jobs DESC LIMIT 10`);
      return res.json({ role, scope: 'All hiring managers, recruiters & clients',
        totals: { jobs: +jobsTotal.rows[0].c, open: +jobsOpen.rows[0].c, submissions: +subs.rows[0].c, placements: +placed.rows[0].c },
        byRecruiter: byRecruiter.rows, byClient: byClient.rows });
    }
    if (role === 'hiring_manager') {
      const byStatus = await pool.query(`SELECT status, COUNT(*) jobs FROM jobs GROUP BY status`);
      return res.json({ role, scope: 'Your job requisitions & their progress',
        totals: { jobs: +jobsTotal.rows[0].c, open: +jobsOpen.rows[0].c, submissions: +subs.rows[0].c, placements: +placed.rows[0].c },
        byStatus: byStatus.rows });
    }
    // recruiter
    return res.json({ role, scope: 'Your assigned jobs & submissions',
      totals: { assigned_jobs: +jobsOpen.rows[0].c, submissions: +subs.rows[0].c, placements: +placed.rows[0].c } });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});


// ---------- ONBOARDING: recruiter invites a candidate (creates onboarding + returns email content) ----------
app.post('/candidates/invite', async (req, res) => {
  try {
    const { candidate_id, candidate_name, email, phone, job_id, job_title, client } = req.body;
    if (!candidate_id || !candidate_name) return res.status(400).json({ error: 'candidate_id and candidate_name required' });
    const token = 'INV-' + Math.random().toString(36).slice(2, 10).toUpperCase();
    await pool.query(
      `INSERT INTO candidate_onboarding (candidate_id, job_id, candidate_name, email, phone, invite_token, onboarding_status)
       VALUES ($1,$2,$3,$4,$5,$6,'INVITED')`,
      [candidate_id, job_id || '', candidate_name, email || '', phone || '', token]
    );
    // The portal link the candidate uses to upload their resume
    const portalLink = `${process.env.PORTAL_URL || 'https://direct-tribute-502305-q5.web.app'}/#/candidate-upload?token=${token}`;
    const emailSubject = `You've been shortlisted for ${job_title || 'a role'}${client ? ' at ' + client : ''}`;
    const emailBody = `Dear ${candidate_name},\n\nGood news! You have been shortlisted for the position of ${job_title || 'the role'}${client ? ' with our client ' + client : ''}.\n\nTo proceed, please upload your latest resume and confirm your contact details using the secure link below:\n\n${portalLink}\n\nThis helps us move your application forward. We look forward to working with you.\n\nBest regards,\nRecruit 360 Team`;
    return res.json({ ok: true, token, portalLink, emailSubject, emailBody, email });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- ONBOARDING: candidate fetches their invite by token ----------
app.get('/onboarding/:token', async (req, res) => {
  try {
    const { rows } = await pool.query('SELECT candidate_id, candidate_name, email, phone, job_id, resume_uploaded, onboarding_status FROM candidate_onboarding WHERE invite_token = $1', [req.params.token]);
    if (!rows.length) return res.status(404).json({ error: 'Invite not found' });
    return res.json({ onboarding: rows[0] });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- ONBOARDING: candidate uploads resume + confirms contact ----------
app.post('/onboarding/:token/submit', async (req, res) => {
  try {
    const { email, phone, resume_filename } = req.body;
    const upd = await pool.query(
      `UPDATE candidate_onboarding SET email = COALESCE($1,email), phone = COALESCE($2,phone),
       resume_uploaded = TRUE, resume_filename = $3, onboarding_status = 'RESUME_SUBMITTED'
       WHERE invite_token = $4 RETURNING candidate_id`,
      [email || null, phone || null, resume_filename || 'resume.pdf', req.params.token]
    );
    // Mark the submission's resume as received (status indicator)
    try { if (upd.rows.length) { await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS resume_received BOOLEAN DEFAULT FALSE'); await pool.query('UPDATE submissions SET resume_received = TRUE WHERE candidate_id = $1', [upd.rows[0].candidate_id]); } } catch(e){}
    return res.json({ ok: true, message: 'Resume received. Your application is moving forward.' });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- VALIDATION: recruiter starts validation for a candidate ----------
app.post('/validation/start', async (req, res) => {
  try {
    const { candidate_id, job_id, candidate_name } = req.body;
    const validation_id = 'VAL-' + Date.now().toString().slice(-9);
    await pool.query(
      `INSERT INTO candidate_validation (validation_id, candidate_id, job_id, candidate_name, stage)
       VALUES ($1,$2,$3,$4,'RECRUITER_CALL')`,
      [validation_id, candidate_id, job_id || '', candidate_name || '']
    );
    return res.json({ ok: true, validation_id });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- VALIDATION: set scenario (1/2/3) after the recruiter call ----------
app.patch('/validation/:id/scenario', async (req, res) => {
  try {
    const { scenario, recruiter_notes } = req.body;
    // Scenario 3 skips training -> next stage is CLIENT_INTERVIEW; 1&2 -> TRAINING
    const nextStage = Number(scenario) === 3 ? 'CLIENT_INTERVIEW' : 'CLIENT_CONFIRM';
    await pool.query(
      `UPDATE candidate_validation SET scenario = $1, recruiter_notes = $2, stage = $3, updated_at = NOW() WHERE validation_id = $4`,
      [Number(scenario), recruiter_notes || '', nextStage, req.params.id]
    );
    return res.json({ ok: true, next_stage: nextStage });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- VALIDATION: advance to the next stage ----------
app.patch('/validation/:id/advance', async (req, res) => {
  try {
    const order = ['RECRUITER_CALL', 'CLIENT_CONFIRM', 'TRAINING', 'CLIENT_INTERVIEW', 'DONE'];
    const cur = await pool.query('SELECT stage, scenario FROM candidate_validation WHERE validation_id = $1', [req.params.id]);
    if (!cur.rows.length) return res.status(404).json({ error: 'Not found' });
    let idx = order.indexOf(cur.rows[0].stage);
    let next = order[Math.min(order.length - 1, idx + 1)];
    // Scenario 3 skips TRAINING
    if (next === 'TRAINING' && Number(cur.rows[0].scenario) === 3) next = 'CLIENT_INTERVIEW';
    await pool.query('UPDATE candidate_validation SET stage = $1, updated_at = NOW() WHERE validation_id = $2', [next, req.params.id]);
    return res.json({ ok: true, stage: next });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- VALIDATION: generate an interview link (real Meet-style link) + email content ----------
app.post('/validation/:id/interview-link', async (req, res) => {
  try {
    const code = Math.random().toString(36).slice(2, 6) + '-' + Math.random().toString(36).slice(2, 6);
    const link = `https://meet.google.com/${code}`;
    await pool.query('UPDATE candidate_validation SET interview_link = $1, updated_at = NOW() WHERE validation_id = $2', [link, req.params.id]);
    const v = await pool.query('SELECT candidate_name FROM candidate_validation WHERE validation_id = $1', [req.params.id]);
    const name = v.rows.length ? v.rows[0].candidate_name : 'Candidate';
    const emailSubject = 'Your interview is scheduled';
    const emailBody = `Dear ${name},\n\nYour interview has been scheduled. Please join using the link below at the agreed time:\n\n${link}\n\nBest regards,\nRecruit 360 Team`;
    return res.json({ ok: true, link, emailSubject, emailBody });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- VALIDATION: client decision ----------
app.patch('/validation/:id/decision', async (req, res) => {
  try {
    const { decision } = req.body; // ACCEPTED / REJECTED
    await pool.query('UPDATE candidate_validation SET client_decision = $1, stage = $2, updated_at = NOW() WHERE validation_id = $3',
      [decision, decision === 'ACCEPTED' ? 'DONE' : 'DONE', req.params.id]);
    return res.json({ ok: true });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- VALIDATION: get validations for a job (or all) ----------
app.get('/validations', async (req, res) => {
  try {
    const { job_id } = req.query;
    let sql = 'SELECT * FROM candidate_validation';
    const params = [];
    if (job_id) { sql += ' WHERE job_id = $1'; params.push(job_id); }
    sql += ' ORDER BY updated_at DESC LIMIT 100';
    const { rows } = await pool.query(sql, params);
    return res.json({ validations: rows });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});


// ============================================================
// RECRUIT 360 AI ASSISTANT — ported from the Streamlit agent (BigQuery + Gemini)
// Role-aware, grounded, with the guardrail rules + chat memory.
// ============================================================
const BQ_DS = 'direct-tribute-502305-q5.recruit360';

const REJECTION_FIXES = {
  'R-01':'Incomplete application form — complete all mandatory fields and resubmit.',
  'R-02':'Invalid/expired passport — renew passport (min 6 months validity) and attach a clear copy.',
  'R-03':'Insufficient financial proof — provide bank statements or a sponsorship letter meeting the threshold.',
  'R-04':'Missing employer sponsorship — obtain a signed sponsorship letter and employment contract.',
  'R-05':'Photograph does not meet spec — submit a biometric photo per the embassy specification.',
  'R-06':'Missing/invalid qualifications — attach attested degree/certificates with certified translation.',
  'R-07':'Medical certificate missing — complete the panel medical and attach the report.',
  'R-08':'Travel/health insurance missing — purchase a compliant policy and attach it.',
  'R-09':'Inconsistent personal details — correct mismatched details across all documents.',
  'R-10':'Interview/extra docs required — schedule the embassy interview or provide the requested documents.'
};

const BQ_SCHEMA = `Tables in ${BQ_DS} (join on ids):
- "visa rejected" means candidates.visa_status = 'VISA_REJECTED'.
- The 8 valid roles: Software Engineer, Registered Nurse, Data Engineer, Cloud Architect, DevOps Engineer, Data Analyst, QA Engineer, Mechanical Engineer.
- visa_status STAGES: Intake=INTAKE_PENDING,STAKEHOLDERS_ASSIGNED,TERMS_LOCKED; Screening=DOCUMENTS_VERIFIED,SCREENING,TRAINING_IN_PROGRESS,TRAINING_COMPLETE,REMEDIATION_IN_PROGRESS; Placement=PLACEMENT_ACTIVE; Visa=VISA_SUBMITTED,VISA_APPROVED,VISA_REJECTED; Travel=TRAVEL_CONFIRMED,REPORTED,NOT_REPORTED,ARRIVED.
- Use LOWER(col) LIKE LOWER('%x%') for text filters. experience_years INTEGER (fresher<=1, senior>=5). No skills column.
- origin_city = current Indian city (where they live now). destination_country = foreign country they go to. NEVER confuse them.
candidates(candidate_id, full_name, email, origin_city, destination_country, destination_employer, role, recruiter, csr_owner, training_centre, visa_agency, visa_status, deposit_amount, currency, urgency_score, experience_years, created_at)
visa_workflows(visa_id, candidate_id, visa_agency, jurisdiction, submitted_date, decision, rejection_codes, retry_count, decision_date)
billing_schedules(billing_id, candidate_id, billing_domain, amount, currency, status, due_date)
jobs(job_id, title, client, department, location, status, openings, recruiter, created_date)
placements(placement_id, candidate_id, job_id, placement_date, fee_eur, status)`;

function _readonlySQL(sql) {
  let l = sql.toLowerCase().trim().replace(/;+$/, '').trim();
  if (l.includes(';')) return false;
  if (!l.startsWith('select')) return false;
  if (/\b(insert|update|delete|drop|create|alter|merge|truncate|grant|revoke|call|execute)\b/.test(l)) return false;
  return true;
}

async function bqQuery(sql) {
  const [rows] = await bq.query({ query: sql, useLegacySql: false, maximumBytesBilled: '200000000' });
  return rows;
}

// The main data agent — LLM writes SQL, we run it on BigQuery.
async function agentQueryData(question, role, history) {
  const wantAll = /(show all|list all|all of them|everyone|every candidate|full list|show more|all candidates|complete list|entire list|see all|view all)/i.test(question);
  const listLimit = wantAll ? 500 : 50;
  const roleHint = role === 'recruiter' ? 'The user is a recruiter asking about candidates/jobs.' : role === 'hiring_manager' ? 'The user is a hiring manager.' : role === 'admin' ? 'The user is an admin with full access.' : '';
  const ctx = (history || []).slice(-4).map(h => `${h.role}: ${h.text}`).join(' | ');
  const rules = `Rules: select only needed columns (never SELECT *). ALWAYS alias aggregates with a readable name (e.g. COUNT(*) AS count, never a bare COUNT(*)). NEVER combine DISTINCT with ORDER BY inside an aggregate such as STRING_AGG/ARRAY_AGG — if you need distinct ordered values, use a subquery (SELECT DISTINCT … then aggregate). For a list, add LIMIT ${listLimit} at the end. For a count/total use COUNT(*) AS count with no LIMIT. Use COUNT/SUM/AVG for totals, GROUP BY for 'per/by/each/breakdown'. Use ORDER BY DESC + LIMIT for 'top/most/highest'. Text filters use LOWER(col) LIKE LOWER('%v%'). Map pipeline stages to visa_status IN(...). Use candidates.experience_years for experience. Prefix tables with \`${BQ_DS}.\`.`;
  const prompt = `Write ONE efficient BigQuery SELECT (only SQL, no fences). ${roleHint}${ctx ? ' Recent context: ' + ctx : ''}
${rules}
${BQ_SCHEMA}
Question: ${question}
SQL:`;
  const gen = await genModel.generateContent(prompt);
  let sql = gen.response.candidates[0].content.parts[0].text.replace(/^```(?:sql)?|```$/gim, '').trim();
  if (!_readonlySQL(sql)) return { text: 'I could not turn that into a data lookup. Try asking it as a question about candidates, jobs, visas or placements — or say "show it as a table" to reshape the last answer.', rows: [] };

  const runAndFormat = (rows) => {
    if (!rows.length) return { text: 'No records match this request in the database. The correct answer is that none were found — I will not invent any.', rows: [] };
    const shown = rows.slice(0, wantAll ? 25 : 8);
    const note = rows.length > shown.length ? `\n\n(Showing ${shown.length} of ${rows.length}.)` : '';
    return { text: `Result (${rows.length} found):\n${JSON.stringify(shown, null, 1)}${note}`, rows };
  };

  try {
    return runAndFormat(await bqQuery(sql));
  } catch (e1) {
    // One auto-repair pass: give the model the error and ask for corrected SQL.
    try {
      const fixPrompt = `The following BigQuery SQL failed with this error:\nERROR: ${e1.message}\n\nSQL:\n${sql}\n\n${rules}\nReturn ONLY the corrected SQL (no fences, no explanation).`;
      const fixGen = await genModel.generateContent(fixPrompt);
      let sql2 = fixGen.response.candidates[0].content.parts[0].text.replace(/^```(?:sql)?|```$/gim, '').trim();
      if (!_readonlySQL(sql2)) return { text: 'I could not build a safe query for that. Could you rephrase it — for example, name the role, city or status you want?', rows: [] };
      return runAndFormat(await bqQuery(sql2));
    } catch (e2) {
      return { text: 'Sorry, I could not run that query. Please try rephrasing your question (for example, name the role or the exact field you want).', rows: [] };
    }
  }
}


// Cloud SQL data agent — for jobs, submissions, approvals, placements (the live website data)
async function agentSqlData(question, role) {
  const schema = `Cloud SQL (PostgreSQL) tables:
jobs(job_id, title, client, hiring_manager, recruiter, assigned_recruiter_id, country, job_location, number_of_positions, priority, status, created_date)
  -- open jobs = status IN ('Open','Active','POSTED','In Process'); closed = status IN ('Closed','CLOSED','Filled')
submissions(submission_id, candidate_id, candidate_name, job_id, job_title, client_name, status, submitted_by, current_ctc, expected_ctc, resume_received, created_at)
  -- statuses: SUBMITTED, PENDING_HM_APPROVAL, RECRUITER_CALL, CLIENT_INTERVIEW, PENDING_PLACEMENT_APPROVAL, OFFER, PLACED, REJECTED
  -- "who needs approval" / "pending approval" / "awaiting approval" / "to approve" = status = 'PENDING_HM_APPROVAL' (select candidate_name, job_title)
  -- "submitted candidates" = all submissions. "placed" = status='PLACED'. "in interview" = status IN ('RECRUITER_CALL','CLIENT_INTERVIEW')
  -- "rejected" = status='REJECTED". Count questions use COUNT(*).
app_users(user_id, full_name, email, role, user_group)  -- roles: admin, hiring_manager, recruiter
clients(client_id, client_name, country, industry)`;
  const prompt = `Write ONE plain PostgreSQL SELECT statement only — no explanation, no code fences, no trailing semicolon, no ':' named parameters, no '::' type casts. Use COUNT(*) for counts and ALWAYS alias aggregates with a readable name (e.g. COUNT(*) AS count, never a bare COUNT(*)). Use ILIKE for text matching, and LIMIT 50 for lists (never SELECT *). Write real literal values directly in the WHERE clause.
${schema}
Question: ${question}
SQL:`;
  const gen = await genModel.generateContent(prompt);
  let sql = gen.response.candidates[0].content.parts[0].text.replace(/```sql/gi,'').replace(/```/g,'').trim().replace(/;+\s*$/,'').trim();
  // take only the first statement / SELECT onward
  const selIdx = sql.toLowerCase().indexOf('select');
  if (selIdx > 0) sql = sql.slice(selIdx);
  if (!/^select/i.test(sql) || /\b(insert|update|delete|drop|alter|create)\b/i.test(sql)) return { text: 'Could not build a safe query for that.', rows: [] };
  try {
    const { rows } = await pool.query(sql);
    if (!rows.length) return { text: 'No records found for this in the live database.', rows: [] };
    return { text: `Result (${rows.length} found):\n${JSON.stringify(rows.slice(0,15), null, 1)}`, rows };
  } catch (e) { return { text: 'Query failed: ' + e.message, rows: [] }; }
}

async function agentVisaFix(candidateId) {
  const cid = (candidateId || '').trim().toUpperCase();
  try {
    const rows = await bqQuery(`SELECT candidate_id, decision, rejection_codes, retry_count FROM \`${BQ_DS}.visa_workflows\` WHERE candidate_id='${cid}' AND decision='REJECTED'`);
    if (!rows.length) return `No rejected visa found for ${cid} (nothing to remediate).`;
    const codes = [];
    rows.forEach(r => String(r.rejection_codes || '').split(';').forEach(c => { if (c) codes.push(c.trim()); }));
    const mapped = codes.map(c => `- ${c}: ${REJECTION_FIXES[c] || 'Refer to embassy guidance.'}`).join('\n');
    const gen = await genModel.generateContent(`A candidate's visa was rejected with these issues:\n${mapped}\n\nWrite a clear, numbered remediation checklist a recruiter can act on to fix and re-apply.`);
    return `Rejection codes for ${cid}: ${codes.join(', ')}\n\n${gen.response.candidates[0].content.parts[0].text.trim()}`;
  } catch (e) { return 'Visa lookup failed: ' + e.message; }
}

async function agentUrgency(topN) {
  try {
    const rows = await bqQuery(`SELECT candidate_id, full_name, role, destination_country, visa_status, urgency_score FROM \`${BQ_DS}.candidates\` WHERE visa_status IN ('VISA_REJECTED','REMEDIATION_IN_PROGRESS','NOT_REPORTED') OR urgency_score >= 75 ORDER BY urgency_score DESC LIMIT ${parseInt(topN||10,10)}`);
    if (!rows.length) return { text: 'No high-urgency candidates right now.', rows: [] };
    const lines = rows.map(r => `- ${r.full_name} (${r.candidate_id}) — ${r.role} → ${r.destination_country||'—'} · visa ${(r.visa_status||'').replace(/_/g,' ')} · urgency ${r.urgency_score||0}`);
    return { text: 'Top urgent / at-risk candidates:\n' + lines.join('\n'), rows };
  } catch (e) { return { text: 'Urgency query failed.', rows: [] }; }
}

// PREDICT-SCORE — BigQuery ML model: who is most likely to be placed
async function agentPredictPlacement(topN) {
  try {
    const n = parseInt(topN || 10, 10);
    const sql = `SELECT candidate_id, ROUND((SELECT prob FROM UNNEST(predicted_is_placed_probs) WHERE label=1),3) AS placement_probability
      FROM ML.PREDICT(MODEL \`${BQ_DS}.placement_predictor\`, (SELECT * FROM \`${BQ_DS}.candidates\`))
      ORDER BY placement_probability DESC LIMIT ${n}`;
    const [rows] = await bq.query({ query: sql, location: 'asia-south1' });
    if (!rows || !rows.length) return { text: 'No placement predictions are available right now.', rows: [] };
    const ids = rows.map(r => `'${String(r.candidate_id).replace(/'/g,'')}'`).join(',');
    let info = [];
    try { info = await bqQuery(`SELECT candidate_id, full_name, role, destination_country FROM \`${BQ_DS}.candidates\` WHERE candidate_id IN (${ids})`); } catch (e) {}
    const map = {}; info.forEach(x => { map[x.candidate_id] = x; });
    const merged = rows.map(r => {
      const c = map[r.candidate_id] || {};
      return { candidate_id: r.candidate_id, full_name: c.full_name || r.candidate_id, role: c.role || '', destination_country: c.destination_country || '', placement_probability: r.placement_probability };
    });
    const lines = merged.map((r, i) => `${i + 1}. ${r.full_name} (${r.candidate_id}) — ${Math.round((r.placement_probability || 0) * 100)}% likely · ${r.role || '—'} → ${r.destination_country || '—'}`);
    return { text: 'Candidates most likely to be placed, ranked by a model-estimated probability (0-100%, an estimate — not a guarantee, and not tied to a calendar month):\n' + lines.join('\n'), rows: merged };
  } catch (e) { return { text: 'The placement-prediction model is not available right now.', rows: [] }; }
}

// SEMANTIC SEARCH — find candidates by MEANING (Vertex embeddings + BigQuery vector search).
// Fit score mirrors the Job Board: strong role/destination signals dominate, with a small
// semantic nudge — so an obviously right candidate reads as a high match, not a raw distance.
async function agentSemanticMatch(question) {
  let desc = (question || '').replace(/^\s*(find|show|get|search|list)\s+(me\s+)?(candidates?|people|profiles?|someone)\s*/i, '')
    .replace(/^(that\s+are\s+|who\s+are\s+|who\s+|that\s+|like\s+|similar\s+to\s+|matching\s+|with\s+)/i, '').trim();
  if (!desc) desc = question;
  const qL = desc.toLowerCase();
  const COUNTRIES = ['Germany','Sweden','Ireland','Netherlands','France','Norway','Denmark','Finland','Canada','Australia','United Kingdom','United States'];
  const wantCountry = COUNTRIES.find(c => qL.includes(c.toLowerCase())) || '';
  const roleWords = (qL.match(/\b(software|backend|frontend|full.?stack|developer|engineer|nurse|architect|analyst|devops|data|mechanical|cloud|qa|quality)\b/g) || []);
  const norm = (s) => (s || '').toLowerCase();
  const roleHit = (candRole) => {
    const cr = norm(candRole);
    return roleWords.some(w => {
      if (w === 'developer' || w === 'backend' || w === 'frontend' || w === 'full-stack' || w === 'fullstack' || w === 'software') return cr.includes('software') || cr.includes('data engineer');
      if (w === 'qa' || w === 'quality') return cr.includes('qa');
      if (w === 'devops') return cr.includes('devops');
      if (w === 'data') return cr.includes('data');
      if (w === 'cloud') return cr.includes('cloud');
      if (w === 'mechanical') return cr.includes('mechanical');
      if (w === 'nurse') return cr.includes('nurse');
      if (w === 'architect') return cr.includes('architect');
      if (w === 'analyst') return cr.includes('analyst');
      if (w === 'engineer') return cr.includes('engineer');
      return false;
    });
  };
  try {
    const sql = `SELECT h.base.candidate_id AS candidate_id, c.full_name, c.role, c.destination_country, c.visa_status, c.experience_years, h.distance
      FROM VECTOR_SEARCH(TABLE \`${BQ_DS}.candidate_embeddings\`, 'embedding',
        (SELECT ml_generate_embedding_result AS embedding FROM ML.GENERATE_EMBEDDING(MODEL \`${BQ_DS}.text_embedder\`, (SELECT @q AS content))),
        top_k => 15) AS h
      JOIN \`${BQ_DS}.candidates\` c ON c.candidate_id = h.base.candidate_id
      ORDER BY h.distance`;
    const [rows] = await bq.query({ query: sql, location: 'asia-south1', params: { q: desc } });
    let good = (rows || []).filter(r => r.distance <= 0.92);
    if (!good.length) return { text: `No candidates in the database closely match "${desc}". I won't guess — there is no strong match for this request.`, rows: [] };
    good = good.map(r => {
      const sameRole = roleHit(r.role);
      const sameCountry = wantCountry ? norm(r.destination_country) === norm(wantCountry) : false;
      let fit;
      if (sameRole || sameCountry) {
        fit = (sameRole ? 66 : 0) + (sameCountry ? 25 : 0) + Math.round((1 - r.distance) * 8);
      } else {
        // no explicit role/country in the query — score on semantic closeness, readably scaled
        fit = 45 + Math.round((1 - r.distance) * 45);
      }
      fit = Math.max(1, Math.min(99, fit));
      return { candidate_id: r.candidate_id, full_name: r.full_name, role: r.role, destination_country: r.destination_country, visa_status: r.visa_status, experience_years: r.experience_years, match: fit };
    }).sort((a, b) => b.match - a.match).slice(0, 10);
    const lines = good.map(r => `- ${r.full_name} (${r.candidate_id}) — ${r.role}, ${r.experience_years || 0} yrs → ${r.destination_country || '—'} · ${r.match}% match`);
    return { text: `Candidates matching "${desc}" by meaning (ranked by fit):\n` + lines.join('\n'), rows: good };
  } catch (e) { return { text: 'Semantic search is not available right now.', rows: [] }; }
}

// LOCATION — candidates whose CURRENT city (origin_city) is in/near a named city
async function agentNearCity(question) {
  const m = (question || '').match(/(?:near|around|close to|within\s+\d+\s*km(?:\s+of)?)\s+([A-Za-z][\w .'-]+)/i);
  let city = m ? m[1].trim() : '';
  city = city.replace(/[.?!,]+$/,'').replace(/\b(right now|now|please|currently|today|area|region)\b/gi,'').trim();
  if (!city) return { text: 'Please name a city, e.g. "candidates near Bengaluru".', rows: [] };
  try {
    const rows = await bqQuery(`SELECT candidate_id, full_name, role, origin_city, destination_country, visa_status FROM \`${BQ_DS}.candidates\` WHERE LOWER(origin_city) LIKE LOWER('%${city.replace(/'/g,'')}%') LIMIT 50`);
    if (!rows.length) return { text: `No candidates currently live in or near ${city}. (origin_city = where a candidate lives now, an Indian city.)`, rows: [] };
    const lines = rows.slice(0, 10).map(r => `- ${r.full_name} (${r.candidate_id}) — ${r.role}, currently in ${r.origin_city} → going to ${r.destination_country || '—'}`);
    const note = rows.length > 10 ? `\n\n(Showing 10 of ${rows.length}. Full list in the table below.)` : '';
    return { text: `Candidates whose current city is in/near ${city}:\n` + lines.join('\n') + note, rows };
  } catch (e) { return { text: 'City lookup failed.', rows: [] }; }
}

// ============================================================
// USE-CASE SKILLS — deterministic answers to sir's exact questions.
// These give accurate, grounded results where plain text-to-SQL is unreliable.
// ============================================================

// RECRUITER Q1 — jobs sitting idle >5 days + WHY (blocker classification)
async function skillIdleJobs(role, name, userId) {
  // Scope: a recruiter sees only THEIR assigned idle jobs; a hiring manager their own jobs; admin sees all.
  let scope = ''; const params = [];
  if (role === 'recruiter') { params.push(name || '', userId || ''); scope = 'AND (j.recruiter = $1 OR j.assigned_recruiter_id = $2)'; }
  else if (role === 'hiring_manager') { params.push(name || ''); scope = 'AND j.hiring_manager = $1'; }
  const jobs = (await pool.query(`
    SELECT j.job_id, j.title, j.client, j.recruiter, j.description, j.primary_skills,
           (CURRENT_DATE - j.created_date::date) AS age_days,
           COUNT(s.submission_id) AS subs,
           COUNT(*) FILTER (WHERE s.status = 'PENDING_HM_APPROVAL') AS pending_hm,
           COUNT(*) FILTER (WHERE s.status = 'SUBMITTED' AND COALESCE(s.resume_received,FALSE)=FALSE) AS awaiting_resume,
           COUNT(*) FILTER (WHERE s.status = 'CLIENT_INTERVIEW') AS at_client,
           COUNT(*) FILTER (WHERE s.status = 'PENDING_PLACEMENT_APPROVAL') AS at_placement,
           COUNT(*) FILTER (WHERE s.status IN ('RECRUITER_CALL')) AS interviewing
      FROM jobs j
      LEFT JOIN submissions s ON s.job_id = j.job_id
     WHERE COALESCE(j.status,'Open') NOT IN ('Closed','CLOSED','Filled','CANCELLED') ${scope}
     GROUP BY j.job_id, j.title, j.client, j.recruiter, j.description, j.primary_skills, j.created_date
     HAVING (CURRENT_DATE - j.created_date::date) >= 5
     ORDER BY age_days DESC LIMIT 25`, params)).rows;
  if (!jobs.length) return role === 'recruiter' ? 'None of your assigned jobs have been idle for more than 5 days.' : 'No open jobs have been idle for more than 5 days.';
  const lines = jobs.map(j => {
    let blocker;
    const hasJD = (j.description && j.description.trim()) || (j.primary_skills && j.primary_skills.trim());
    const hasRecruiter = j.recruiter && j.recruiter.trim();
    // Most specific blocker first, so each job reads distinctly.
    if (!hasJD) blocker = 'Missing JD — add the job description/skills so sourcing can start';
    else if (Number(j.subs) === 0 && !hasRecruiter) blocker = 'No recruiter assigned — assign a recruiter to start sourcing';
    else if (Number(j.subs) === 0) blocker = `No candidates sourced in ${j.age_days} days — source & submit candidates`;
    else if (Number(j.pending_hm) > 0) blocker = `${j.pending_hm} submission(s) waiting on hiring-manager approval`;
    else if (Number(j.awaiting_resume) > 0) blocker = `${j.awaiting_resume} candidate(s) approved but resume not received`;
    else if (Number(j.interviewing) > 0) blocker = `${j.interviewing} candidate(s) mid-interview — move the rounds forward`;
    else if (Number(j.at_client) > 0) blocker = `Awaiting client feedback on ${j.at_client} candidate(s)`;
    else if (Number(j.at_placement) > 0) blocker = `${j.at_placement} candidate(s) waiting on placement approval`;
    else blocker = 'No active movement — review and push the pipeline';
    return `- ${j.title} (${j.client || '—'}, ${j.job_id}) · idle ${j.age_days} days · Blocker: ${blocker}`;
  });
  return `Open jobs idle 5+ days (${jobs.length}), each with what is blocking it:\n` + lines.join('\n');
}

// BEST FIT — rank candidates for a named role (role match + readiness + experience)
async function skillBestFit(question) {
  const roles = ['Software Engineer','Registered Nurse','Data Engineer','Cloud Architect','DevOps Engineer','Data Analyst','QA Engineer','Mechanical Engineer'];
  const found = roles.find(r => question.toLowerCase().includes(r.toLowerCase()))
    || (/\bqa\b/i.test(question) ? 'QA Engineer' : null)
    || (/\bnurse\b/i.test(question) ? 'Registered Nurse' : null)
    || (/\bdevops\b/i.test(question) ? 'DevOps Engineer' : null);
  if (!found) return null; // let the fallback handle it
  try {
    const rows = await bqQuery(`SELECT candidate_id, full_name, role, destination_country, visa_status, experience_years
      FROM \`${BQ_DS}.candidates\` WHERE LOWER(role) = LOWER('${found.replace(/'/g,'')}')
      ORDER BY
        CASE WHEN visa_status IN ('VISA_APPROVED','PLACEMENT_ACTIVE','TRAINING_COMPLETE','VISA_SUBMITTED','TRAVEL_CONFIRMED') THEN 0 ELSE 1 END,
        experience_years DESC
      LIMIT 6`);
    if (!rows.length) return `No candidates found for ${found} in the pool.`;
    const lines = rows.map((r, i) => `${i + 1}. ${r.full_name} (${r.candidate_id}) — ${r.experience_years || 0} yrs · ${r.destination_country || '—'} · visa ${(r.visa_status || '').replace(/_/g, ' ')}`);
    return `Best-fit ${found} candidates (ranked by visa readiness, then experience):\n` + lines.join('\n');
  } catch (e) { return null; }
}

// RECRUITER Q2 — candidates with visa issues / rejections + reason + action
async function skillVisaIssues() {
  try {
    const rows = await bqQuery(`SELECT c.candidate_id, c.full_name, c.role, c.destination_country, w.rejection_codes
      FROM \`${BQ_DS}.candidates\` c JOIN \`${BQ_DS}.visa_workflows\` w ON w.candidate_id = c.candidate_id
      WHERE w.decision = 'REJECTED' LIMIT 25`);
    if (!rows.length) return { text: 'No candidates currently have a rejected visa.', rows: [] };
    const out = [];
    const lines = rows.map(r => {
      const codes = String(r.rejection_codes || '').split(';').map(c => c.trim()).filter(Boolean);
      const actions = codes.map(c => REJECTION_FIXES[c] || 'Refer to embassy guidance.').join(' ');
      out.push({ candidate_id: r.candidate_id, full_name: r.full_name, role: r.role, destination_country: r.destination_country, reason: codes.join(', ') || 'unspecified', action: actions });
      return `- ${r.full_name} (${r.candidate_id}, ${r.role} → ${r.destination_country}) · Reason: ${codes.join(', ') || 'unspecified'} · Action: ${actions}`;
    });
    return { text: `Candidates with visa rejections (${rows.length}) — reason and action required:\n` + lines.join('\n'), rows: out };
  } catch (e) { return { text: 'Visa issues lookup failed.', rows: [] }; }
}

// RECRUITER Q3 — "what should I work on today?" (prioritised work queue)
async function skillRecruiterQueue(name, userId) {
  const who = name || '';
  const uid = userId || '';
  // Scoped to THIS recruiter: only their own candidates (submitted_by) and jobs assigned to them.
  const needResume = (await pool.query(`SELECT candidate_name, job_title FROM submissions WHERE status='SUBMITTED' AND COALESCE(resume_received,FALSE)=FALSE AND submitted_by = $1 ORDER BY created_at DESC LIMIT 10`, [who])).rows;
  const toSchedule = (await pool.query(`SELECT candidate_name, job_title, status FROM submissions WHERE status IN ('RECRUITER_CALL','CLIENT_INTERVIEW') AND submitted_by = $1 ORDER BY last_updated ASC NULLS FIRST LIMIT 10`, [who])).rows;
  const forPlacement = (await pool.query(`SELECT candidate_name, job_title FROM submissions WHERE status='CLIENT_INTERVIEW' AND submitted_by = $1 ORDER BY last_updated ASC NULLS FIRST LIMIT 10`, [who])).rows;
  // All OPEN jobs assigned to this recruiter, with how many candidates each has so far.
  const assigned = (await pool.query(
    `SELECT j.title, j.client, (CURRENT_DATE - j.created_date::date) AS age, COUNT(s.submission_id) AS subs
       FROM jobs j LEFT JOIN submissions s ON s.job_id = j.job_id
      WHERE (j.recruiter = $1 OR j.assigned_recruiter_id = $2)
        AND COALESCE(j.status,'Open') NOT IN ('Closed','CLOSED','Filled','CANCELLED')
      GROUP BY j.job_id, j.title, j.client, j.created_date
      ORDER BY j.created_date DESC LIMIT 20`, [who, uid])).rows;
  const needSourcing = assigned.filter(j => Number(j.subs) === 0);
  const aging = assigned.filter(j => Number(j.age) >= 5);
  const parts = [];
  // Surface assigned jobs first — a freshly assigned job with no candidates still needs action.
  if (assigned.length) {
    parts.push(`You have ${assigned.length} assigned job(s): ` + assigned.map(j => j.title + (j.client ? ' (' + j.client + ')' : '') + ' — ' + Number(j.subs) + ' candidate(s)').join('; '));
    if (needSourcing.length) parts.push(`Start sourcing candidates for (${needSourcing.length}): ` + needSourcing.map(j => j.title + (j.client ? ' (' + j.client + ')' : '')).join('; '));
  }
  if (needResume.length) parts.push(`Chase resumes (${needResume.length}): ` + needResume.map(r => r.candidate_name + ' — ' + r.job_title).join('; '));
  if (toSchedule.length) parts.push(`Move interviews forward (${toSchedule.length}): ` + toSchedule.map(r => r.candidate_name + ' (' + (r.status || '').replace(/_/g,' ') + ')').join('; '));
  if (forPlacement.length) parts.push(`Close out client interviews / send for placement approval (${forPlacement.length}): ` + forPlacement.map(r => r.candidate_name).join('; '));
  if (aging.length) parts.push(`Unblock aging jobs (${aging.length}): ` + aging.map(j => j.title + ' — ' + j.age + 'd').join('; '));
  if (!parts.length) return 'You have no assigned jobs or candidates yet. Once a hiring manager assigns you a job and you start adding candidates, your prioritised work for the day will appear here.';
  return 'Your prioritised work for today:\n' + parts.join('\n');
}

// RECRUITER complex Q1 — candidates close to placement but blocked
async function skillPlacementRisk() {
  const rows = (await pool.query(`SELECT submission_id, candidate_id, candidate_name, job_title, client_name, status, COALESCE(resume_received,FALSE) AS resume_received
     FROM submissions WHERE status IN ('CLIENT_INTERVIEW','PENDING_PLACEMENT_APPROVAL','OFFER') ORDER BY last_updated ASC NULLS FIRST LIMIT 25`)).rows;
  if (!rows.length) return 'No candidates are near placement right now.';
  // enrich with visa status from BigQuery in one shot
  let visaMap = {};
  try {
    const ids = rows.map(r => `'${(r.candidate_id||'').replace(/'/g,'')}'`).filter(x => x !== "''");
    if (ids.length) {
      const v = await bqQuery(`SELECT candidate_id, visa_status FROM \`${BQ_DS}.candidates\` WHERE candidate_id IN (${ids.join(',')})`);
      v.forEach(x => { visaMap[x.candidate_id] = x.visa_status; });
    }
  } catch (e) {}
  const lines = rows.map(r => {
    const visa = visaMap[r.candidate_id];
    let blocker, who;
    if (visa && /REJECT/.test(visa)) { blocker = 'Visa rejected — documents incomplete'; who = 'candidate action'; }
    else if (!r.resume_received && r.status === 'CLIENT_INTERVIEW') { blocker = 'Resume not on file'; who = 'candidate action'; }
    else if (r.status === 'PENDING_PLACEMENT_APPROVAL' || r.status === 'OFFER') { blocker = 'Placement approval pending'; who = 'hiring-manager action'; }
    else { blocker = 'Awaiting client decision'; who = 'client / recruiter action'; }
    return `- ${r.candidate_name} → ${(r.status||'').replace(/_/g,' ')} → ${blocker} → ${who}`;
  });
  return `${rows.length} placement(s) at risk — blocker, then who needs to act:\n` + lines.join('\n');
}

// HIRING MANAGER Q1 — what needs my attention today
async function skillHMToday(name) {
  const who = name || '';
  // When a hiring-manager name is given, scope to their jobs; admin (no name) sees everything.
  const subs = (await pool.query(`SELECT s.candidate_name, s.job_title, s.client_name FROM submissions s LEFT JOIN jobs j ON j.job_id = s.job_id WHERE s.status='PENDING_HM_APPROVAL' AND ($1 = '' OR j.hiring_manager = $1) ORDER BY s.created_at DESC LIMIT 15`, [who])).rows;
  const placements = (await pool.query(`SELECT s.candidate_name, s.job_title, s.client_name FROM submissions s LEFT JOIN jobs j ON j.job_id = s.job_id WHERE s.status='PENDING_PLACEMENT_APPROVAL' AND ($1 = '' OR j.hiring_manager = $1) ORDER BY s.created_at DESC LIMIT 15`, [who])).rows;
  const parts = [];
  parts.push(`Submission approvals waiting (${subs.length})` + (subs.length ? ':\n' + subs.map(s => '  - ' + s.candidate_name + ' for ' + s.job_title + (s.client_name ? ' (' + s.client_name + ')' : '')).join('\n') : ' — none.'));
  parts.push(`Placement approvals waiting (${placements.length})` + (placements.length ? ':\n' + placements.map(s => '  - ' + s.candidate_name + ' for ' + s.job_title + (s.client_name ? ' (' + s.client_name + ')' : '')).join('\n') : ' — none.'));
  return 'What needs your attention today:\n' + parts.join('\n');
}

// ADMIN — offer letters awaiting the authorized signatory (admin's pending work).
async function skillPendingSignatures() {
  try {
    const { rows } = await pool.query(`SELECT candidate_name, job_title, client_name, offer_region, submitted_by FROM submissions WHERE offer_status = 'DRAFT' AND status = 'OFFER' ORDER BY last_updated DESC LIMIT 25`);
    if (!rows.length) return 'Offer letters awaiting your signature (0) — none right now.';
    return `Offer letters awaiting your signature (${rows.length}):\n` + rows.map(r => '  - ' + r.candidate_name + ' for ' + (r.job_title || 'a role') + (r.client_name ? ' (' + r.client_name + ')' : '') + (r.offer_region ? ' · ' + r.offer_region : '') + (r.submitted_by ? ' · prepared by ' + r.submitted_by : '') + ' — review & sign in the Offer Signatures tab.').join('\n');
  } catch (e) { return 'Offer letters awaiting your signature (0) — none right now.'; }
}

// HIRING MANAGER Q2 — why haven't we found candidates for this role?
async function skillWhyNoCandidates(question) {
  const roles = ['Software Engineer','Registered Nurse','Data Engineer','Cloud Architect','DevOps Engineer','Data Analyst','QA Engineer','Mechanical Engineer'];
  const found = roles.find(r => question.toLowerCase().includes(r.toLowerCase()));
  if (!found) return null; // let the fallback handle it
  try {
    const total = (await bqQuery(`SELECT COUNT(*) AS n FROM \`${BQ_DS}.candidates\` WHERE LOWER(role)=LOWER('${found}')`))[0]?.n || 0;
    const rejected = (await bqQuery(`SELECT COUNT(*) AS n FROM \`${BQ_DS}.candidates\` WHERE LOWER(role)=LOWER('${found}') AND visa_status='VISA_REJECTED'`))[0]?.n || 0;
    const ready = (await bqQuery(`SELECT COUNT(*) AS n FROM \`${BQ_DS}.candidates\` WHERE LOWER(role)=LOWER('${found}') AND visa_status IN ('TRAINING_COMPLETE','VISA_APPROVED','PLACEMENT_ACTIVE')`))[0]?.n || 0;
    return `Feasibility for ${found}: ${total} candidates in the pool, ${ready} are placement-ready, ${rejected} have a visa rejection. ` +
      (Number(total) === 0 ? 'The pool is empty for this role — it is a skill-scarcity problem; widen sourcing or relax location/rate.' :
       Number(ready) === 0 ? 'Candidates exist but none are placement-ready yet (training/visa in progress).' :
       'There is a ready pool — the gap is likely matching or client feedback, not supply.');
  } catch (e) { return null; }
}

const ASSISTANT_RULES = `You are the Recruit 360 AI Assistant. Ground every answer in the data provided by the tools; never invent a candidate name or ID. If a tool returns nothing, say so plainly — do not fabricate. Answer the exact question directly and answer every part of a multi-part question. origin_city = current Indian city; destination_country = where they go (never confuse). Placement probability is a model estimate (0-1), not a guarantee. You inform, you do not decide — never tell the user to hire/reject a specific candidate. You only help with Recruit 360 recruitment data; politely decline off-topic questions. Be concise and professional.`;

function roleScope(role) {
  if (role === 'admin') return 'The user is an ADMIN with full visibility across all recruiters, clients, candidates and the pipeline.';
  if (role === 'hiring_manager') return 'The user is a HIRING MANAGER — focus on their requisitions, pending approvals, submissions and candidate progress.';
  if (role === 'recruiter') return 'The user is a RECRUITER — focus on their assigned jobs, their candidates, visa status and next actions.';
  return 'The user is a platform user.';
}

// ---------- THE ASSISTANT ENDPOINT (accurate, role-scoped, grounded) ----------
app.post('/assistant/ask', async (req, res) => {
  try {
    const { question, role, history, userName, userId } = req.body;
    if (!question) return res.status(400).json({ error: 'question required' });
    const q = question.toLowerCase().trim();

    // --- Greetings & vague follow-ups: show a helpful menu instead of guessing ---
    const menu = (role === 'hiring_manager')
      ? 'I can help with:\n• What needs my attention today?\n• Why haven’t we found candidates for a role (e.g. Registered Nurse)?\n• Which candidates have visa issues?\n• How many submissions are pending my approval?'
      : (role === 'admin')
      ? 'I can help with:\n• Which jobs are idle more than 5 days and what is blocking each?\n• What needs attention today?\n• Which candidates have visa issues?\n• Which placements are at risk?'
      : 'I can help with:\n• What should I work on today?\n• Which jobs are idle more than 5 days and what is blocking each?\n• Which candidates have visa issues and what is the action?\n• Which of my placements are at risk?';
    if (/^(hi+|hello|hey|yo|namaste|good (morning|afternoon|evening))\b/i.test(q)) {
      return res.json({ answer: 'Hello! I’m the Recruit 360 assistant. ' + menu, agent: 'Assistant' });
    }
    // Gratitude — always reply politely, never error.
    if (/\b(thanks|thank you|thankyou|thx|thnx|ty)\b/i.test(q) && q.length < 40) {
      return res.json({ answer: 'You’re welcome! Is there anything else I can help you with about your recruitment data?', agent: 'Assistant' });
    }
    // Acknowledgements / sign-offs / tiny messages — reply gently instead of running a query.
    if (/^(bye|goodbye|see you|that'?s all|that is all|nothing|no)\b/i.test(q)) {
      return res.json({ answer: 'Alright — I’m here whenever you need anything about candidates, jobs, approvals, visas or placements.', agent: 'Assistant' });
    }
    if (q.length < 6 || /^(anything else|what else|more|ok(ay)?|k|cool|great|nice|good|awesome|perfect|got it|fine|done|yes|hmm+|and\??|next|continue)\b[\s\S]{0,15}$/i.test(q)) {
      return res.json({ answer: 'Sure — here’s what I can help with. ' + menu, agent: 'Assistant' });
    }

    // --- Formatting / transform follow-ups: reshape the PREVIOUS answer, don't re-query ---
    // e.g. "give in proper grid view", "list this properly", "summarise", "make it shorter".
    const hasDataNoun = /(candidate|job|jobs|visa|placement|submission|interview|approv|recruiter|client|billing|training|nurse|engineer|architect|analyst|role|city|country|experience|fresher|senior|urgent|idle|near|likely)/i.test(q);
    const isReformat = /(grid|table|tabular|column|proper format|properly|reformat|format it|format this|format the|summari[sz]e|give.*summary|shorten|make it short|in detail|expand it|as bullet|as a list|list it|organi[sz]e|nicely|cleaner|clean it|clean this|better format|neat)/i.test(q) && !hasDataNoun;
    if (isReformat && Array.isArray(history) && history.length) {
      const lastBot = [...history].reverse().find(h => h && h.role !== 'user' && h.text);
      if (lastBot) {
        const wantsTable = /(grid|table|tabular|column)/i.test(q);
        const instruction = wantsTable
          ? 'Reformat the previous answer as a clean, aligned text table (a header row, then one record per line with columns separated by " | "). Keep every record. Plain text only.'
          : 'Reformat the previous answer exactly as the user asks (e.g. shorter, a summary, or a tidy list). Keep the facts identical — do not add or invent anything.';
        const rp = `You are reformatting your own previous answer for the user. ${instruction}
Do NOT use markdown asterisks, bold or backticks. Do not add new data. If the previous answer was an error or empty, ask the user to try the question again instead.

USER INSTRUCTION: ${question}

PREVIOUS ANSWER:
${lastBot.text}

REFORMATTED ANSWER:`;
        try {
          const g = await genModel.generateContent(rp);
          return res.json({ answer: g.response.candidates[0].content.parts[0].text.trim(), agent: 'Formatting' });
        } catch (e) { /* fall through to normal routing */ }
      }
    }

    // --- Fast intent detection (reliable keyword routing) ---
    const idMatch = question.match(/\bC\d{4}\b/i);
    // Any question about a SPECIFIC candidate's rejection / reason / fix -> visa remediation tool.
    const isVisaFix = !!idMatch && /(fix|remediat|reject|rejection|reason|what.?s wrong|how to fix|how do i fix|resolve|why)/i.test(q);
    const isUrgency = /(urgent|urgency|at.?risk|awol|not reported|priority candidates|who needs attention)/i.test(q);

    let toolResult = '', agentName = '', rows = [];
    // Route: jobs/submissions/approvals/placements -> Cloud SQL (live website data). Candidates/visa -> BigQuery.
    // Jobs/submissions live in Cloud SQL; candidates/visa live in BigQuery — never send a "visa" question here.
    const isJobsData = /(open job|jobs\b|job posting|requisition|submission|submitted|approv|pending|awaiting|assigned|my job|placement|placed|offer|interview|recruiter call|hr round|shortlist|reject)/i.test(q) && !/\bvisa\b/i.test(q);

    // --- Use-case skills (sir's exact questions) — checked first for reliable answers ---
    const isIdleJobs = /((idle|sitting idle|stuck|blocked|blocker|not moving|no movement|aging|ageing)[^.]*\b(job|jobs|requisition)|which jobs[^.]*(idle|blocked|stuck|blocking))/i.test(q);
    // "how many / count / total" questions go to the data agent for the clean number (e.g. 78),
    // not the visa-issues list skill (which only counts formal rejection workflows).
    const isCountQ = /\b(how many|count|number of|total|how much)\b/i.test(q);
    const isVisaIssues = /(visa (issue|problem|reject)|rejections?|who[^.]*visa|candidates[^.]*visa[^.]*(issue|reject))/i.test(q) && !idMatch && !isCountQ;
    const isWorkToday = /(work on today|what should i (do|work)|my (work|queue|priorities|priority|tasks|pending)|today.?s (work|priorities|tasks)|prioriti[sz]e[^.]*today|need[^.]*my attention|what needs[^.]*attention|attention today|pending work|what.?s pending|whats pending|summary of (my )?(work|pending|tasks)|work summary|summari[sz]e (my )?(work|pending|tasks|day))/i.test(q);
    const isPlacementRisk = /(close to placement|near placement|placement[^.]*(risk|blocker|block)|at risk[^.]*placement|placements? at risk)/i.test(q);
    const isWhyNoCand = /(why[^.]*(no|haven.?t|not)[^.]*candidat|why[^.]*can.?t[^.]*find|feasibility)/i.test(q);
    const isBestFit = /(best fit|best candidate|best.*match|top candidate|good fit|suitable candidate|who.?s? (the )?best|which candidate.*best|candidates? (for|suitable for) (a |the )?[a-z])/i.test(q) && !isVisaIssues && !isWhyNoCand;
    const isPredict = /(likely to be placed|most likely|placement (probability|likelihood|chance|score)|best prospect|who should i prioriti[sz]e|predict placement|placement predict)/i.test(q);
    const isNearCity = /\b(near|around|close to|within\s+\d+\s*km)\b/i.test(q) && /candidat/i.test(q);
    const isSemantic = /(candidates? (like|similar to|matching)|find (me )?(someone|people|candidates?) (like|with|who match|matching)|semantic|profiles? (like|matching)|similar profile)/i.test(q);

    if (isVisaFix) {
      toolResult = await agentVisaFix(idMatch[0]); agentName = 'Visa Fix-It';
    } else if (isBestFit) {
      const r = await skillBestFit(question);
      if (r) { toolResult = r; agentName = 'Best-Fit Match'; }
      else { const rr = await agentQueryData(question, role, history); toolResult = rr.text; rows = rr.rows; agentName = 'Candidate Data'; }
    } else if (isIdleJobs) {
      toolResult = await skillIdleJobs(role, userName, userId); agentName = 'Idle & Blocked Jobs';
    } else if (isPlacementRisk) {
      toolResult = await skillPlacementRisk(); agentName = 'Placement Risk';
    } else if (/(pending|awaiting|need|needs|to)\s*(my\s*)?(signature|sign)|offers?\s*(to|for|awaiting|pending|need)?\s*sign|sign\s*(the\s*)?offer|signature queue|offers? (awaiting|pending|to be) sign/i.test(q)) {
      toolResult = await skillPendingSignatures(); agentName = 'Offer Signatures';
    } else if (isWorkToday) {
      if (role === 'hiring_manager') toolResult = await skillHMToday(userName);
      else if (role === 'admin') { const sig = await skillPendingSignatures(); const a = await skillHMToday(); const b = await skillIdleJobs('admin'); toolResult = sig + '\n\n' + a + '\n\n' + b; }
      else toolResult = await skillRecruiterQueue(userName, userId);
      agentName = 'My Work Today';
    } else if (isVisaIssues) {
      const r = await skillVisaIssues(); toolResult = r.text; rows = r.rows; agentName = 'Visa Issues';
    } else if (isWhyNoCand) {
      const r = await skillWhyNoCandidates(question);
      if (r) { toolResult = r; agentName = 'Role Feasibility'; }
      else { const rr = await agentQueryData(question, role, history); toolResult = rr.text; rows = rr.rows; agentName = 'Candidate Data'; }
    } else if (isPredict) {
      const r = await agentPredictPlacement(10); toolResult = r.text; rows = r.rows; agentName = 'Placement Prediction';
    } else if (isNearCity) {
      const r = await agentNearCity(question); toolResult = r.text; rows = r.rows; agentName = 'Location Match';
    } else if (isSemantic) {
      const r = await agentSemanticMatch(question); toolResult = r.text; rows = r.rows; agentName = 'Semantic Match';
    } else if (isUrgency) {
      const r = await agentUrgency(10); toolResult = r.text; rows = r.rows; agentName = 'Urgency Watch';
    } else if (/(who needs approv|needs approval|pending approv|awaiting approv|to approve|approval queue|whom.*approve)/i.test(q)) {
      // Direct, reliable query for pending approvals
      const pr = await pool.query(`SELECT candidate_name, job_title, client_name FROM submissions WHERE status = 'PENDING_HM_APPROVAL' ORDER BY created_at DESC`);
      toolResult = pr.rows.length ? ('Candidates awaiting approval (' + pr.rows.length + '):\n' + pr.rows.map(x => '- ' + x.candidate_name + ' for ' + x.job_title + (x.client_name ? ' (' + x.client_name + ')' : '')).join('\n')) : 'There are no candidates awaiting approval right now.';
      rows = pr.rows; agentName = 'Approvals';
    } else if (isCountQ && (/\bvisa\b/.test(q) || /candidate|applicant|pool|placed|placement|submission|submitted|shortlist/.test(q))) {
      // Deterministic counts — reliable numbers for the common "how many / total" questions,
      // so they never depend on the AI query-writer. Falls back to the data agent only if these fail.
      try {
        let n, label;
        if (/\bvisa\b/.test(q) && /reject/.test(q)) {
          n = Number((await bqQuery(`SELECT COUNT(*) AS n FROM \`${BQ_DS}.candidates\` WHERE UPPER(visa_status) = 'VISA_REJECTED'`))[0]?.n || 0);
          label = `There are ${n} candidates whose visa has been rejected.`;
        } else if (/\bvisa\b/.test(q) && /(approved|cleared)/.test(q)) {
          n = Number((await bqQuery(`SELECT COUNT(*) AS n FROM \`${BQ_DS}.candidates\` WHERE UPPER(visa_status) = 'VISA_APPROVED'`))[0]?.n || 0);
          label = `There are ${n} candidates with an approved visa.`;
        } else if (/placed|placement/.test(q)) {
          n = Number((await pool.query("SELECT COUNT(*) AS n FROM submissions WHERE status = 'PLACED'")).rows[0]?.n || 0);
          label = `There are ${n} placed candidates.`;
        } else if (/submission|submitted|shortlist/.test(q)) {
          n = Number((await pool.query('SELECT COUNT(*) AS n FROM submissions')).rows[0]?.n || 0);
          label = `There are ${n} submissions in total.`;
        } else {
          n = Number((await bqQuery(`SELECT COUNT(*) AS n FROM \`${BQ_DS}.candidates\``))[0]?.n || 0);
          label = `There are ${n} candidates in total in the pool.`;
        }
        toolResult = label; agentName = 'Candidate Data';
      } catch (e) {
        const rr = await agentQueryData(question, role, history); toolResult = rr.text; rows = rr.rows; agentName = 'Candidate Data';
      }
    } else if (isJobsData) {
      const r = await agentSqlData(question, role); toolResult = r.text; rows = r.rows; agentName = 'Jobs & Submissions';
    } else {
      const r = await agentQueryData(question, role, history);
      toolResult = r.text; rows = r.rows; agentName = 'Candidate Data';
    }

    // Compose a clean answer that faithfully reflects the tool result (no invention).
    const scope = role === 'admin' ? 'admin (full visibility)' : role === 'hiring_manager' ? 'hiring manager' : role === 'recruiter' ? 'recruiter' : 'user';
    const finalPrompt = `You are the Recruit 360 AI assistant answering a ${scope}. Below is the exact result from the database for the user's question. Answer using ONLY this result — never add or invent names, numbers or candidates that are not in it. If the result says none were found, say clearly that there are none. Never output raw column names or aliases like "f0_" or JSON keys — describe the numbers in a plain sentence (e.g. "There are 78 …"). If the result is an internal error or guard message, do NOT repeat it — instead briefly ask the user to rephrase.
STRUCTURE: open with ONE short summary sentence that answers the question at a glance (e.g. "You have 12 candidates with visa issues — most need document fixes."), then a blank line, then the details. For a list, put each item on its own line starting with "• ". If a value is a probability/score, briefly note it is a model estimate.
FORMAT as plain text for a chat bubble: do NOT use markdown, asterisks (*), bold (**), backticks or headings. Be concise and professional.

DATABASE RESULT:
${toolResult}

USER QUESTION: ${question}

ANSWER:`;
    const finalGen = await genModel.generateContent(finalPrompt);
    let answer = finalGen.response.candidates[0].content.parts[0].text.trim();
    // Final safety net: never leak internal guard/error phrasing to the user.
    if (/read-only guard|Query failed|ensureSchema|bqQuery|SELECT \*|\bf0_\b/i.test(answer)) {
      answer = 'I hit a snag turning that into an answer. Could you rephrase it as a question about candidates, jobs, visas or placements?';
    }
    // Return rows + agent so the UI can show the "Result data" table and the agent trace.
    return res.json({ answer, agent: agentName, rows: Array.isArray(rows) ? rows.slice(0, 500) : [] });
  } catch (e) {
    // Never surface a raw "Assistant error" — always answer gracefully.
    return res.json({ answer: 'Sorry, I had trouble with that one. Please try rephrasing it — for example, ask about candidates, jobs, approvals, visas or placements.', agent: 'Assistant', rows: [] });
  }
});

// ---------- CONTEXTUAL RECRUITER AGENT: suggest & execute the next action per candidate ----------
// Given a submission's current status, it knows exactly what the recruiter should do next.
app.get('/submissions/:id/next-action', async (req, res) => {
  try {
    await ensureSlots();
    const { rows } = await pool.query('SELECT submission_id, candidate_id, candidate_name, job_title, client_name, status, interview_link, interview_date, resume_received, resume_requested, resume_summary, client_panel, current_ctc, expected_ctc, submitted_by FROM submissions WHERE submission_id = $1', [req.params.id]);
    if (!rows.length) return res.status(404).json({ error: 'not found' });
    const s = rows[0];
    // New flow: resume + AI summary come BEFORE hiring-manager approval.
    // SUBMITTED: received -> send summary to HM; requested -> waiting; else -> request resume.
    const submittedStep = s.resume_received
      ? { action: 'send_to_hm', label: 'Send summary to hiring manager', can: true, hint: 'Resume received — review it and send an AI summary to the hiring manager for approval.' }
      : s.resume_requested
        ? { action: 'await_resume', label: 'Awaiting resume', can: false, hint: 'The resume-upload email was sent — waiting for the candidate to upload.' }
        : { action: 'request_resume', label: 'Request resume', can: true, hint: 'Send the candidate the resume-upload email.' };
    const MAP = {
      SUBMITTED:                  submittedStep,
      PENDING_HM_APPROVAL:        { action: 'awaiting_approval', label: 'Awaiting approval', can: false, hint: 'The hiring manager is reviewing the resume summary.' },
      RECRUITER_CALL:             { action: 'schedule_call', label: 'Schedule recruiter call', can: true, hint: 'Shortlisted — propose slots for the recruiter call and capture CTC there.' },
      CLIENT_INTERVIEW:           { action: 'request_placement', label: 'Send for placement approval', can: true, hint: 'Request the client panel, schedule the client interview, then send to the hiring manager for placement approval.' },
      PENDING_PLACEMENT_APPROVAL: { action: 'await_placement', label: 'Awaiting placement approval', can: false, hint: 'Placement approval is pending with the hiring manager.' },
      OFFER:                      { action: 'prepare_offer', label: 'Prepare & send the offer letter', can: false, hint: 'Approved to hire — prepare the offer letter below, get it signed by an admin, then send it to the candidate.' },
      PLACED:                     { action: 'done', label: 'Placed', can: false, hint: 'This candidate is placed.' },
      REJECTED:                   { action: 'done', label: 'Rejected', can: false, hint: 'This candidate was rejected.' },
      OFFER_DECLINED:             { action: 'done', label: 'Offer declined', can: false, hint: 'The candidate declined the offer.' },
    };
    const next = MAP[s.status] || { action: 'review', label: 'Review', can: true, hint: 'Review this candidate.' };
    return res.json({ submission: s, next });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// Contextual agent EXECUTE — performs the suggested action (and returns any email content)
app.post('/submissions/:id/context-action', async (req, res) => {
  try {
    const { action } = req.body;
    await ensureSlots();
    const { rows } = await pool.query('SELECT candidate_id, candidate_name, job_id, job_title, client_name, status FROM submissions WHERE submission_id = $1', [req.params.id]);
    if (!rows.length) return res.status(404).json({ error: 'not found' });
    const s = rows[0];
    const candEmail = candidateEmail(s.candidate_id);

    if (action === 'request_resume') {
      // create onboarding + invite email
      const token = 'INV-' + Math.random().toString(36).slice(2, 10).toUpperCase();
      try { await pool.query(`INSERT INTO candidate_onboarding (candidate_id, job_id, candidate_name, email, invite_token, onboarding_status) VALUES ($1,$2,$3,$4,$5,'INVITED')`, [s.candidate_id, s.job_id || '', s.candidate_name, candEmail, token]); } catch(e){}
      const link = (process.env.PORTAL_URL || 'https://direct-tribute-502305-q5.web.app') + '/#/candidate-upload?token=' + token;
      const emailSubject = 'Please upload your resume — ' + (s.job_title || 'a role');
      const emailBody = 'Dear ' + s.candidate_name + ',\n\nYou have been shortlisted for ' + (s.job_title || 'a role') + (s.client_name ? ' at ' + s.client_name : '') + '. Please upload your latest resume and confirm your contact details here:\n\n' + link + '\n\nBest regards,\nRecruit 360 Team';
      try { await pool.query('UPDATE submissions SET resume_requested = TRUE, last_updated = NOW() WHERE submission_id = $1', [req.params.id]); } catch (e) {}
      await logComm({ submission_id: req.params.id, candidate_id: s.candidate_id, job_id: s.job_id, channel: 'EMAIL', to_role: 'candidate', to_name: s.candidate_name, to_email: candEmail, subject: emailSubject, body: emailBody, sent_by: 'recruiter' });
      return res.json({ ok: true, done: 'request_resume', emailSubject, emailBody, candidateEmail: candEmail, link });
    }

    // NEW FLOW: recruiter reviews the resume and sends an AI summary to the hiring manager for approval.
    if (action === 'send_to_hm') {
      // pull resume text (if stored) + candidate profile to ground the summary
      let resumeText = '';
      try { resumeText = (await pool.query('SELECT resume_text FROM submissions WHERE submission_id = $1', [req.params.id])).rows[0]?.resume_text || ''; } catch (e) {}
      // Summary is grounded in whatever we already have: the resume text if on file,
      // otherwise the candidate's SAVED profile (pool candidates are already in our database).
      let basis = '', basisNote = '';
      if (resumeText && resumeText.trim().length >= 20) {
        basis = 'Resume:\n' + resumeText.slice(0, 6000); basisNote = 'from the resume on file';
      } else {
        let prof = {};
        try { prof = (await pool.query('SELECT role, origin_city, destination_country, experience_years FROM candidates WHERE candidate_id = $1', [s.candidate_id])).rows[0] || {}; } catch (e) {}
        if (!prof || !prof.role) { try { const b = await bqQuery(`SELECT role, origin_city, destination_country, experience_years FROM \`${BQ_DS}.candidates\` WHERE candidate_id = '${(s.candidate_id || '').replace(/'/g, '')}'`); if (b && b[0]) prof = b[0]; } catch (e) {} }
        basis = 'Candidate profile on file — role: ' + (prof.role || s.job_title || 'n/a') + ', experience: ' + (prof.experience_years != null ? prof.experience_years + ' years' : 'n/a') + ', current city: ' + (prof.origin_city || 'n/a') + ', destination: ' + (prof.destination_country || 'n/a') + '.';
        basisNote = 'from the candidate’s saved profile';
      }
      let summary = '';
      try {
        const gen = await genModel.generateContent('You are a recruiter preparing a concise candidate summary for a hiring manager to approve. Candidate: ' + s.candidate_name + ' for the role ' + (s.job_title || '') + (s.client_name ? ' at ' + s.client_name : '') + '. Base it ONLY on the data below (' + basisNote + ').\n' + basis + '\n\nWrite 4-5 short lines: fit for the role, experience, key points, and any strength or gap. Plain text, no markdown.');
        summary = gen.response.candidates[0].content.parts[0].text.trim();
      } catch (e) { summary = `${s.candidate_name} — candidate for ${s.job_title || 'the role'}, summarized from the profile on file.`; }
      try { await pool.query('UPDATE submissions SET status = $1, resume_summary = $2, last_updated = NOW() WHERE submission_id = $3', ['PENDING_HM_APPROVAL', summary, req.params.id]); }
      catch (e) { await pool.query("UPDATE submissions SET status = 'PENDING_HM_APPROVAL', last_updated = NOW() WHERE submission_id = $1", [req.params.id]); }
      let hm = '';
      try { hm = (await pool.query('SELECT hiring_manager FROM jobs WHERE job_id = $1', [s.job_id])).rows[0]?.hiring_manager || ''; } catch (e) {}
      const hmMsg = `Candidate for approval: ${s.candidate_name} for ${s.job_title || 'a role'}${s.client_name ? ' (' + s.client_name + ')' : ''}. Resume reviewed — summary attached.`;
      await notify({ candidate_id: s.candidate_id, recipient: hm || 'hiring_manager', type: 'CANDIDATE_REVIEW', message: hmMsg });
      await logComm({ submission_id: req.params.id, candidate_id: s.candidate_id, job_id: s.job_id, channel: 'UPDATE', to_role: 'hiring_manager', to_name: hm || 'hiring_manager', subject: 'Candidate summary for approval — ' + s.candidate_name, body: summary, sent_by: 'recruiter' });
      return res.json({ ok: true, done: 'send_to_hm', status: 'PENDING_HM_APPROVAL', summary, message: 'Summary sent to the hiring manager for approval.' });
    }
    if (action === 'send_summary') {
      const gen = await genModel.generateContent('Write a short professional candidate summary email to a client for ' + s.candidate_name + ', role ' + (s.job_title || '') + '. 4-5 lines, highlight fit. Return only the email body.');
      const emailBody = gen.response.candidates[0].content.parts[0].text.trim();
      return res.json({ ok: true, done: 'send_summary', emailSubject: 'Candidate summary — ' + s.candidate_name, emailBody, candidateEmail: (s.client_name||'client').toLowerCase().replace(/\s+/g,'') + '@client.com' });
    }

    // Recruiter sends the candidate to the hiring manager for PLACEMENT approval.
    // Recruiters cannot place directly — this only queues it for the HM.
    if (action === 'request_placement') {
      if (s.status !== 'CLIENT_INTERVIEW') {
        return res.status(400).json({ error: 'Placement can only be requested after the client interview.' });
      }
      await pool.query("UPDATE submissions SET status = 'PENDING_PLACEMENT_APPROVAL', last_updated = NOW() WHERE submission_id = $1", [req.params.id]);
      // Notify the hiring manager for this job
      let hm = '';
      try { hm = (await pool.query('SELECT hiring_manager FROM jobs WHERE job_id = $1', [s.job_id])).rows[0]?.hiring_manager || ''; } catch (e) {}
      const hmMsg = `Placement approval needed: ${s.candidate_name} for ${s.job_title || 'a role'}${s.client_name ? ' (' + s.client_name + ')' : ''}.`;
      await notify({ candidate_id: s.candidate_id, recipient: hm || 'hiring_manager', type: 'PLACEMENT_APPROVAL', message: hmMsg });
      await logComm({ submission_id: req.params.id, candidate_id: s.candidate_id, job_id: s.job_id, channel: 'UPDATE', to_role: 'hiring_manager', to_name: hm || 'hiring_manager', subject: 'Placement approval needed — ' + s.candidate_name, body: hmMsg, sent_by: s.submitted_by || 'recruiter' });
      return res.json({ ok: true, done: 'request_placement', status: 'PENDING_PLACEMENT_APPROVAL', message: 'Sent to the hiring manager for placement approval.' });
    }

    // Recruiter call finished -> move to the client-interview stage (no extra email/link; the call was already scheduled via slots).
    if (action === 'complete_recruiter_call') {
      if (s.status !== 'RECRUITER_CALL') return res.status(400).json({ error: 'The candidate is not at the recruiter-call stage.' });
      await pool.query("UPDATE submissions SET status = 'CLIENT_INTERVIEW', last_updated = NOW() WHERE submission_id = $1", [req.params.id]);
      let hm = '';
      try { hm = (await pool.query('SELECT hiring_manager FROM jobs WHERE job_id = $1', [s.job_id])).rows[0]?.hiring_manager || ''; } catch (e) {}
      await notify({ candidate_id: s.candidate_id, recipient: s.submitted_by || 'recruiter', type: 'STAGE', message: `${s.candidate_name} moved to the client interview — request the client panel.` });
      await logComm({ submission_id: req.params.id, candidate_id: s.candidate_id, job_id: s.job_id, channel: 'UPDATE', to_role: 'recruiter', to_name: s.submitted_by || 'recruiter', subject: 'Recruiter call complete — ' + s.candidate_name, body: 'Recruiter call completed. Proceeding to the client interview (request the panel from the client).', sent_by: 'recruiter' });
      return res.json({ ok: true, done: 'complete_recruiter_call', status: 'CLIENT_INTERVIEW', message: 'Recruiter call complete — moved to the client interview.' });
    }

    // Schedule actions advance the pipeline forward (HR round removed — recruiter call then client interview).
    const ADVANCE = { schedule_call: 'CLIENT_INTERVIEW', schedule_client: 'CLIENT_INTERVIEW' };
    if (ADVANCE[action]) {
      await pool.query('UPDATE submissions SET status = $1, last_updated = NOW() WHERE submission_id = $2', [ADVANCE[action], req.params.id]);
      const labels = { schedule_call: 'recruiter call', schedule_client: 'client interview' };
      const meetCode = Math.random().toString(36).slice(2, 5) + '-' + Math.random().toString(36).slice(2, 6) + '-' + Math.random().toString(36).slice(2, 5);
      const interview_link = 'https://meet.google.com/' + meetCode;
      const emailSubject = 'Your ' + labels[action] + ' is scheduled — ' + (s.job_title || 'a role');
      const emailBody = 'Dear ' + s.candidate_name + ',\n\nYour ' + labels[action] + ' for ' + (s.job_title || 'a role') + (s.client_name ? ' at ' + s.client_name : '') + ' has been scheduled.\n\nJoin using this link:\n' + interview_link + '\n\nPlease confirm your availability by replying to this email.\n\nBest regards,\nRecruit 360 Team';
      await pool.query('UPDATE submissions SET interview_link = $1 WHERE submission_id = $2', [interview_link, req.params.id]).catch(() => {});
      // Record & keep recruiter + hiring manager in the loop
      let hm2 = '';
      try { hm2 = (await pool.query('SELECT hiring_manager FROM jobs WHERE job_id = $1', [s.job_id])).rows[0]?.hiring_manager || ''; } catch (e) {}
      const cArgs = { submission_id: req.params.id, candidate_id: s.candidate_id, job_id: s.job_id, sent_by: 'recruiter' };
      await logComm({ ...cArgs, channel: 'EMAIL', to_role: 'candidate', to_name: s.candidate_name, to_email: candEmail, subject: emailSubject, body: emailBody });
      const upd = s.candidate_name + '’s ' + labels[action] + ' for ' + (s.job_title || 'a role') + ' has been scheduled. Link: ' + interview_link;
      await notify({ candidate_id: s.candidate_id, recipient: 'candidate', type: 'INTERVIEW', message: 'Your ' + labels[action] + ' is scheduled. Link: ' + interview_link });
      if (hm2) { await notify({ candidate_id: s.candidate_id, recipient: hm2, type: 'INTERVIEW', message: upd }); await logComm({ ...cArgs, channel: 'UPDATE', to_role: 'hiring_manager', to_name: hm2, subject: 'Interview scheduled — ' + s.candidate_name, body: upd }); }
      return res.json({ ok: true, done: action, status: ADVANCE[action], interview_link, emailSubject, emailBody, candidateEmail: candEmail });
    }

    return res.json({ ok: true, done: action });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- PLACEMENT APPROVAL (hiring manager only) ----------
// Recruiters queue placements; the hiring manager approves here.
app.get('/placements/pending-approval', async (req, res) => {
  try {
    const hm = String(req.query.hm || '').trim();
    const scoped = String(req.query.role || '') !== 'admin' && hm;
    const { rows } = await pool.query(
      `SELECT s.submission_id, s.candidate_id, s.candidate_name, s.job_id, s.job_title, s.client_name, s.current_ctc, s.expected_ctc, s.submitted_by, s.created_at
         FROM submissions s LEFT JOIN jobs j ON j.job_id = s.job_id
        WHERE s.status = 'PENDING_PLACEMENT_APPROVAL' ${scoped ? 'AND j.hiring_manager = $1' : ''}
        ORDER BY s.created_at DESC`, scoped ? [hm] : []);
    return res.json({ pending: rows });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- MANUAL / OFFLINE RESUME (recruiter received it via email/chat/call) ----------
// sir's flow: recruiter reaches out -> candidate sends resume -> recruiter adds it here.
app.patch('/submissions/:id/resume', async (req, res) => {
  try {
    const { filename, doc_type, pasted_text, source, added_by } = req.body || {};
    await ensureSlots();
    const s = (await pool.query('SELECT candidate_id, candidate_name, job_id, job_title FROM submissions WHERE submission_id = $1', [req.params.id])).rows[0];
    if (!s) return res.status(404).json({ error: 'not found' });
    // Store the actual resume text so the AI summary to the hiring manager is grounded in the real resume.
    await pool.query('UPDATE submissions SET resume_received = TRUE, resume_requested = TRUE, resume_text = COALESCE($2, resume_text), last_updated = NOW() WHERE submission_id = $1',
      [req.params.id, pasted_text ? String(pasted_text).slice(0, 8000) : null]);
    const tag = doc_type || 'Resume';
    const via = source ? ' (received via ' + source + ')' : '';
    const body = `${tag} received for ${s.candidate_name}${filename ? ' — ' + filename : ''}${via} and added manually by the recruiter.`
      + (pasted_text ? '\n\n--- Pasted content ---\n' + String(pasted_text).slice(0, 4000) : '')
      + '\n\nReady for the recruiter call.';
    await logComm({ submission_id: req.params.id, candidate_id: s.candidate_id, job_id: s.job_id, channel: 'DOCUMENT', to_role: 'recruiter', to_name: added_by || 'recruiter', subject: tag + ' added (offline)', body, sent_by: added_by || 'recruiter' });
    return res.json({ ok: true, resume_received: true });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

app.patch('/submissions/:id/placement-approval', async (req, res) => {
  try {
    const { decision, approver, reason } = req.body; // APPROVED | REJECTED
    const s = (await pool.query('SELECT candidate_id, candidate_name, job_id, job_title, client_name, submitted_by FROM submissions WHERE submission_id = $1', [req.params.id])).rows[0];
    if (!s) return res.status(404).json({ error: 'not found' });

    const cArgs = { submission_id: req.params.id, candidate_id: s.candidate_id, job_id: s.job_id, sent_by: approver || 'hiring_manager' };
    const candEmail = candidateEmail(s.candidate_id);
    if (decision === 'APPROVED') {
      // Stage 6: HM gives internal approval to hire -> move to the OFFER stage (prepare offer letter).
      // The candidate is NOT placed yet; placement (onboarding + visa email + job-fill) happens once the
      // admin-signed offer is accepted (see /offer/accept below).
      await ensureSlots();
      await pool.query("UPDATE submissions SET status = 'OFFER', offer_status = 'PENDING', last_updated = NOW() WHERE submission_id = $1", [req.params.id]);
      const recMsg = `Approved to hire: ${s.candidate_name} for ${s.job_title || 'a role'}. Next step: prepare the offer letter for admin signature.`;
      await notify({ candidate_id: s.candidate_id, recipient: s.submitted_by || 'recruiter', type: 'PLACEMENT_APPROVED', message: recMsg });
      let hm = '';
      try { hm = (await pool.query('SELECT hiring_manager FROM jobs WHERE job_id = $1', [s.job_id])).rows[0]?.hiring_manager || ''; } catch (e) {}
      await notify({ candidate_id: s.candidate_id, recipient: hm || 'hiring_manager', type: 'OFFER_PREP', message: `Prepare the offer letter for ${s.candidate_name} (${s.job_title || 'a role'}).` });
      await logComm({ ...cArgs, channel: 'UPDATE', to_role: 'hiring_manager', to_name: hm || 'hiring_manager', subject: 'Approved to hire — prepare offer: ' + s.candidate_name, body: recMsg });
      return res.json({ ok: true, status: 'OFFER', next: 'prepare_offer', message: 'Approved to hire. Prepare the offer letter next, then send it to admin for signature.' });
    }
    // Rejected -> send it back to the recruiter at the client-interview stage with a reason
    await pool.query("UPDATE submissions SET status = 'CLIENT_INTERVIEW', last_updated = NOW() WHERE submission_id = $1", [req.params.id]);
    const rejMsg = `Placement not approved for ${s.candidate_name}${reason ? ': ' + reason : ''}.`;
    await notify({ candidate_id: s.candidate_id, recipient: s.submitted_by || 'recruiter', type: 'PLACEMENT_REJECTED', message: rejMsg });
    await logComm({ ...cArgs, channel: 'UPDATE', to_role: 'recruiter', to_name: s.submitted_by || 'recruiter', subject: 'Placement not approved — ' + s.candidate_name, body: rejMsg });
    return res.json({ ok: true, status: 'CLIENT_INTERVIEW' });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ============================================================
// OFFER LETTER (Stage 6 & 7) — HM prepares final terms, AI drafts the letter,
// ADMIN (authorized signatory) signs/seals, then it is sent to the candidate.
// Region-aware (India / Germany / Poland). No infra deps: a branded, print-ready
// HTML letter the UI renders and downloads as a PDF.
// ============================================================
const COMPANY_NAME = 'Avanciers';
// Default authorized signatory (admin). Can be overridden per-sign from the UI.
const DEFAULT_SIGNATORY = { name: process.env.SIGNATORY_NAME || 'Authorized Signatory', title: process.env.SIGNATORY_TITLE || 'Director, Avanciers' };

// Region presets: currency, statutory leave, service-commitment norms and payroll cadence hints.
function offerRegionDefaults(region) {
  const r = String(region || 'India').trim().toLowerCase();
  if (r === 'germany' || r === 'de') {
    return { region: 'Germany', country: 'Germany', currency: '€', currency_code: 'EUR', locale: 'de-DE',
      vacation_days: 28, min_stay_months: 12, repayment_cap: '€8,000', pay_type: 'biweekly',
      note: 'European salary standards apply; figures are gross before tax, social security and statutory contributions.' };
  }
  if (r === 'poland' || r === 'pl') {
    return { region: 'Poland', country: 'Poland', currency: 'zł', currency_code: 'PLN', locale: 'pl-PL',
      vacation_days: 26, min_stay_months: 12, repayment_cap: 'zł30,000', pay_type: 'monthly',
      note: 'Figures are gross and aligned to the applicable Polish national minimum-wage baseline and statutory contributions.' };
  }
  return { region: 'India', country: 'India', currency: '₹', currency_code: 'INR', locale: 'en-IN',
    vacation_days: 24, min_stay_months: 6, repayment_cap: '₹400,000', pay_type: 'annual',
    note: 'Figures are gross before tax, statutory deductions and employer contributions as applicable in India.' };
}

function esc(s) { return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }

function payLine(d) {
  const a = esc(d.pay_amount || '');
  const c = d.currency || '';
  if (!a) return 'As discussed and confirmed on the recruiter call.';
  switch (d.pay_type) {
    case 'hourly':   return `${c}${a} per hour` + (d.annualized ? `, annualizing to approximately ${c}${esc(d.annualized)} gross per year` : '');
    case 'monthly':  return `${c}${a} gross per month` + (d.annualized ? `, i.e. approximately ${c}${esc(d.annualized)} gross per year` : '');
    case 'biweekly': return `${c}${a} gross every two weeks (bi-weekly)` + (d.annualized ? `, i.e. approximately ${c}${esc(d.annualized)} gross per year` : '');
    default:         return `${c}${a} gross per year` + (d.monthly ? ` (approx. ${c}${esc(d.monthly)} gross per month, indicative, before tax and deductions)` : '');
  }
}

// Build the full, print-ready offer letter. `signed` adds the authorized-signatory block + seal.
function buildOfferHtml(d) {
  const today = new Date().toLocaleDateString(d.locale || 'en-IN', { day: 'numeric', month: 'long', year: 'numeric' });
  const startDate = d.start_date ? new Date(d.start_date).toLocaleDateString(d.locale || 'en-IN', { day: 'numeric', month: 'long', year: 'numeric' }) : 'to be confirmed';
  const resp = (d.responsibilities || '').trim();
  const respHtml = resp
    ? resp.split(/\n+/).filter(Boolean).map(x => `<li>${esc(x.replace(/^[-•]\s*/, ''))}</li>`).join('')
    : '<li>Deliver the responsibilities of the role as directed by your reporting manager.</li>';
  const signedBlock = d.signed ? `
      <div class="sign">
        <div class="seal">AUTHORIZED<br/>&bull; ${esc(COMPANY_NAME).toUpperCase()} &bull;<br/>SIGNATORY</div>
        <div class="sigline">
          <div class="signame">${esc(d.signatory_name || DEFAULT_SIGNATORY.name)}</div>
          <div class="sigtitle">${esc(d.signatory_title || DEFAULT_SIGNATORY.title)}</div>
          <div class="sigdate">Signed on ${esc(d.signed_at ? new Date(d.signed_at).toLocaleDateString(d.locale || 'en-IN', { day: 'numeric', month: 'long', year: 'numeric' }) : today)}</div>
        </div>
      </div>` : `
      <div class="sign pending"><div class="sigpending">Pending authorized-signatory approval (admin)</div></div>`;
  return `<!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
  @page { size: A4; margin: 18mm; }
  * { box-sizing: border-box; }
  body { font-family: Georgia, 'Times New Roman', serif; color: #1f2430; line-height: 1.5; font-size: 12.5px; margin: 0; background: #fff; }
  .page { max-width: 760px; margin: 0 auto; padding: 28px 30px; }
  .brandbar { display:flex; justify-content:space-between; align-items:flex-end; border-bottom: 3px solid #4f46e5; padding-bottom: 10px; margin-bottom: 4px; }
  .brand { font-family: Arial, Helvetica, sans-serif; font-weight: 800; letter-spacing: 2px; font-size: 20px; color:#312e81; }
  .brand small { display:block; font-size: 9px; letter-spacing: 3px; color:#6b7280; font-weight:600; margin-top:2px; }
  .conf { font-family: Arial, sans-serif; font-size: 9px; color:#9ca3af; text-transform: uppercase; letter-spacing: 1px; text-align:right; }
  h1 { font-family: Arial, Helvetica, sans-serif; font-size: 15px; letter-spacing: 1px; text-align:center; margin: 16px 0 2px; color:#111827; }
  .sub { text-align:center; font-family: Arial, sans-serif; font-size: 10.5px; color:#6b7280; margin-bottom: 14px; }
  table.meta { width:100%; border-collapse: collapse; margin: 10px 0 16px; font-size: 11.5px; }
  table.meta td { border:1px solid #e5e7eb; padding: 6px 9px; vertical-align: top; }
  table.meta td.k { background:#f8fafc; font-family: Arial, sans-serif; font-weight:700; color:#374151; width: 38%; }
  h2 { font-family: Arial, Helvetica, sans-serif; font-size: 12px; color:#312e81; margin: 16px 0 4px; }
  ul { margin: 4px 0 4px 18px; padding:0; }
  li { margin: 2px 0; }
  p { margin: 6px 0; }
  .greeting { margin-top: 10px; }
  .footnote { font-size: 10px; color:#6b7280; font-style: italic; margin-top:4px; }
  .sign { margin-top: 30px; display:flex; align-items:center; gap: 26px; }
  .sign.pending { justify-content:flex-start; }
  .seal { width: 92px; height: 92px; border: 2px dashed #4f46e5; border-radius: 50%; display:flex; align-items:center; justify-content:center; text-align:center; font-family: Arial, sans-serif; font-size: 8px; font-weight:800; color:#4f46e5; letter-spacing:1px; transform: rotate(-9deg); }
  .sigline { }
  .signame { font-family: 'Segoe Script', 'Brush Script MT', cursive; font-size: 24px; color:#1e3a8a; border-bottom:1px solid #9ca3af; padding-bottom:3px; min-width: 240px; }
  .sigtitle { font-family: Arial, sans-serif; font-size: 11px; color:#374151; margin-top:4px; font-weight:700; }
  .sigdate { font-family: Arial, sans-serif; font-size: 10px; color:#6b7280; margin-top:2px; }
  .sigpending { font-family: Arial, sans-serif; font-size: 11px; color:#b45309; background:#fffbeb; border:1px solid #fde68a; border-radius:6px; padding:8px 12px; }
  .foot { margin-top: 26px; border-top:1px solid #e5e7eb; padding-top:8px; font-family: Arial, sans-serif; font-size: 9px; color:#9ca3af; text-align:center; }
</style></head>
<body><div class="page">
  <div class="brandbar">
    <div class="brand">${esc(COMPANY_NAME).toUpperCase()}<small>EMPLOYMENT OFFER</small></div>
    <div class="conf">Confidential<br/>Offer reference draft</div>
  </div>
  <h1>EMPLOYMENT OFFER LETTER</h1>
  <div class="sub">${esc(d.location || d.region || '')}${d.role_title ? ' &nbsp;|&nbsp; ' + esc(d.role_title) : ''}</div>

  <p>Date: ${esc(today)}</p>
  <p>To: <strong>${esc(d.candidate_name || 'Candidate')}</strong></p>
  <p>Proposed start date: ${esc(startDate)}</p>

  <p class="greeting">Dear ${esc((d.candidate_name || 'Candidate').split(' ')[0])},</p>
  <p>${esc(d.opening || `We are pleased to offer you employment with ${COMPANY_NAME} (the “Company”) as ${d.role_title || 'a member of our team'}${d.client_name ? ', on assignment with ' + d.client_name : ''}. This letter summarizes the principal terms of the proposed employment. Your employment will be governed by the definitive local employment agreement, Company policies, and all mandatory laws applicable in ${d.country || d.region || 'your work location'}.`)}</p>

  <table class="meta">
    <tr><td class="k">Position</td><td>${esc(d.role_title || '')}</td></tr>
    <tr><td class="k">Primary work location</td><td>${esc(d.location || d.region || '')}</td></tr>
    <tr><td class="k">Reporting manager</td><td>${esc(d.reporting_manager || 'To be confirmed')}</td></tr>
    <tr><td class="k">Assignment / engagement</td><td>${esc(d.assignment_duration || 'Ongoing, subject to performance and business need')}</td></tr>
    <tr><td class="k">Compensation</td><td>${payLine(d)}</td></tr>
    <tr><td class="k">Paid leave</td><td>${esc(d.vacation_days || offerRegionDefaults(d.region).vacation_days)} paid working days per calendar year, plus statutory holidays</td></tr>
  </table>

  <h2>1. Compensation and payroll</h2>
  <p>Your compensation will be ${payLine(d)}. ${esc(d.note || '')} Actual payroll may vary with approved hours, unpaid absence, overtime treatment, the payroll calendar, taxes, social contributions, pension obligations and other lawful deductions. Overtime or additional-hours payments, where applicable, are handled under local law and Company policy.</p>

  <h2>2. Health insurance and statutory benefits</h2>
  <p>${esc(d.benefits || `Coverage under the Company's group medical policy and statutory benefits, subject to insurer terms and the applicable ${d.country || d.region || 'local'} benefits schedule.`)} Final eligibility, contribution levels, waiting periods, dependent coverage and exclusions will be set out in the local benefits documentation.</p>

  <h2>3. Vacation and leave</h2>
  <p>You will be eligible for <strong>${esc(d.vacation_days || offerRegionDefaults(d.region).vacation_days)} paid working days</strong> per calendar year, plus applicable public holidays and statutory leave. Sick leave, parental leave, family leave and other protected leave will be provided in accordance with applicable law and Company policy. Leave scheduling requires reasonable advance approval except in emergencies.</p>
  ${d.relocation ? `<h2>4. Relocation support</h2><p>${esc(d.relocation)}</p>` : ''}

  <h2>${d.relocation ? '5' : '4'}. Roles and responsibilities</h2>
  <ul>${respHtml}</ul>

  <h2>${d.relocation ? '6' : '5'}. Initial service commitment and cost repayment</h2>
  <p>The Company expects you to remain employed for at least <strong>${esc(d.min_stay_months || offerRegionDefaults(d.region).min_stay_months)} months</strong> after your start date. If you voluntarily resign before completing that period, you may be required to repay actual, documented and legally recoverable Company-paid costs relating to joining travel, temporary accommodation and employer-paid training, up to a maximum of <strong>${esc(d.repayment_cap || offerRegionDefaults(d.region).repayment_cap)}</strong>. This is intended as reimbursement of specified costs, not as a penalty for resigning. Any repayment will be prorated where required by law or Company policy, will not include ordinary wages, and will not be deducted from salary or final pay unless such deduction is lawful and you have provided any authorization required by local law. No repayment will be sought where prohibited by law, and the Company may waive repayment in cases such as redundancy, Company-initiated termination without cause, serious illness, or other circumstances approved by HR.</p>

  <h2>${d.relocation ? '7' : '6'}. Point of contact</h2>
  <p>For employment, travel, relocation or workplace-support matters, contact: ${esc(d.hr_name || 'HR Business Partner')}${d.hr_phone ? ' | ' + esc(d.hr_phone) : ''}${d.hr_email ? ' | ' + esc(d.hr_email) : ''}.</p>

  <h2>${d.relocation ? '8' : '7'}. Conditions of offer</h2>
  <p>This offer is conditional upon satisfactory identity and right-to-work verification, reference/background checks where lawful, any required visa or work authorization, and execution of the Company's confidentiality, intellectual-property and data-protection agreements. This letter is a summary; the definitive local employment agreement will control.</p>

  <p style="margin-top:14px">We look forward to welcoming you to ${esc(COMPANY_NAME)}.</p>
  <p>Yours sincerely,</p>
  ${signedBlock}

  <div class="foot">${esc(COMPANY_NAME)} &bull; Confidential draft employment offer &bull; This document is not valid until signed by an authorized signatory.</div>
</div></body></html>`;
}

// Assemble the offer data object from request body + region presets + submission/candidate context.
async function assembleOfferData(s, body) {
  const defs = offerRegionDefaults(body.region);
  const data = {
    submission_id: s.submission_id, candidate_id: s.candidate_id, candidate_name: s.candidate_name,
    job_title: s.job_title, client_name: s.client_name,
    region: defs.region, country: defs.country, locale: defs.locale, note: defs.note,
    currency: body.currency || defs.currency, currency_code: body.currency_code || defs.currency_code,
    pay_type: body.pay_type || defs.pay_type,
    pay_amount: body.pay_amount || '', annualized: body.annualized || '', monthly: body.monthly || '',
    role_title: body.role_title || s.job_title || '',
    responsibilities: body.responsibilities || '',
    benefits: body.benefits || '',
    vacation_days: body.vacation_days || defs.vacation_days,
    relocation: body.relocation || '',
    assignment_duration: body.assignment_duration || '',
    location: body.location || '',
    start_date: body.start_date || '',
    min_stay_months: body.min_stay_months || defs.min_stay_months,
    repayment_cap: body.repayment_cap || defs.repayment_cap,
    reporting_manager: body.reporting_manager || '',
    hr_name: body.hr_name || '', hr_phone: body.hr_phone || '', hr_email: body.hr_email || '',
    gen_at: new Date().toISOString(),
  };
  // AI assist (opening + responsibilities) — grounded, never invents numbers. Safe fallback on any error.
  try {
    const want = [];
    if (!data.responsibilities) want.push('responsibilities');
    want.push('opening');
    const prompt = `You are drafting parts of a formal employment offer letter for ${COMPANY_NAME}. Region: ${data.region}. Candidate: ${data.candidate_name}. Role: ${data.role_title}${data.client_name ? ' (assignment with ' + data.client_name + ')' : ''}. Location: ${data.location || data.region}.\n`
      + `Return a strict JSON object with keys ${want.map(w => '"' + w + '"').join(' and ')}.\n`
      + `"opening": 2-3 sentence professional opening paragraph welcoming the candidate and framing this as a summary of principal terms; do NOT state any salary number.\n`
      + (want.includes('responsibilities') ? `"responsibilities": 4-6 concise role responsibilities for a ${data.role_title}, as a single string with each item on its own line, no bullets characters.\n` : '')
      + `Plain professional English. Output ONLY the JSON, nothing else.`;
    const gen = await genModel.generateContent(prompt);
    let txt = gen.response.candidates[0].content.parts[0].text.trim().replace(/^```json/i, '').replace(/^```/, '').replace(/```$/, '').trim();
    const j = JSON.parse(txt);
    if (j.opening) data.opening = String(j.opening).trim();
    if (!data.responsibilities && j.responsibilities) data.responsibilities = String(j.responsibilities).trim();
  } catch (e) { /* deterministic template fallback is used */ }
  return data;
}

// GET the current offer for a submission
app.get('/submissions/:id/offer', async (req, res) => {
  try {
    await ensureSlots();
    const r = (await pool.query('SELECT submission_id, candidate_id, candidate_name, job_title, client_name, status, offer_status, offer_json, offer_html, offer_region, offer_signed_by, offer_signed_at, submitted_by FROM submissions WHERE submission_id = $1', [req.params.id])).rows[0];
    if (!r) return res.status(404).json({ error: 'not found' });
    let offer = null; try { offer = r.offer_json ? JSON.parse(r.offer_json) : null; } catch (e) {}
    return res.json({ submission_id: r.submission_id, status: r.status, offer_status: r.offer_status || (r.status === 'OFFER' ? 'PENDING' : null), offer, html: r.offer_html || '', region: r.offer_region || '', signed_by: r.offer_signed_by || '', signed_at: r.offer_signed_at || null, candidate_name: r.candidate_name, job_title: r.job_title, client_name: r.client_name, submitted_by: r.submitted_by });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// GENERATE (hiring manager or admin) — AI drafts, template guarantees the numbers/clauses.
app.post('/submissions/:id/offer/generate', async (req, res) => {
  try {
    await ensureSlots();
    const by = req.body?.by || 'hiring_manager';
    if (!['hiring_manager', 'admin'].includes(by)) return res.status(403).json({ error: 'Only the hiring manager or admin can prepare an offer.' });
    const s = (await pool.query('SELECT submission_id, candidate_id, candidate_name, job_id, job_title, client_name, status, submitted_by FROM submissions WHERE submission_id = $1', [req.params.id])).rows[0];
    if (!s) return res.status(404).json({ error: 'not found' });
    if (!['OFFER', 'PENDING_PLACEMENT_APPROVAL'].includes(s.status)) return res.status(400).json({ error: 'An offer can only be prepared after placement is approved (approved to hire).' });
    const data = await assembleOfferData(s, req.body || {});
    const html = buildOfferHtml({ ...data, signed: false });
    await pool.query("UPDATE submissions SET status = 'OFFER', offer_status = 'DRAFT', offer_json = $2, offer_html = $3, offer_region = $4, offer_signed_by = NULL, offer_signed_at = NULL, last_updated = NOW() WHERE submission_id = $1",
      [req.params.id, JSON.stringify(data), html, data.region]);
    await logComm({ submission_id: req.params.id, candidate_id: s.candidate_id, job_id: s.job_id, channel: 'DOCUMENT', to_role: 'admin', to_name: 'Authorized signatory', subject: 'Offer letter drafted — ' + s.candidate_name, body: 'Offer letter prepared for ' + s.candidate_name + ' (' + data.region + '). Awaiting admin signature.', sent_by: by });
    return res.json({ ok: true, offer_status: 'DRAFT', offer: data, html });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// SIGN (ADMIN ONLY — authorized signatory). Seals the letter.
app.post('/submissions/:id/offer/sign', async (req, res) => {
  try {
    await ensureSlots();
    const by = req.body?.by || '';
    if (by !== 'admin') return res.status(403).json({ error: 'Only an admin (authorized signatory) can sign the offer letter.' });
    const r = (await pool.query('SELECT candidate_id, candidate_name, job_id, offer_json, offer_status FROM submissions WHERE submission_id = $1', [req.params.id])).rows[0];
    if (!r) return res.status(404).json({ error: 'not found' });
    if (!r.offer_json) return res.status(400).json({ error: 'No offer has been prepared yet.' });
    let data = {}; try { data = JSON.parse(r.offer_json); } catch (e) {}
    const signatory_name = req.body?.signatory_name || DEFAULT_SIGNATORY.name;
    const signatory_title = req.body?.signatory_title || DEFAULT_SIGNATORY.title;
    const signed_at = new Date().toISOString();
    data = { ...data, signatory_name, signatory_title, signed_at };
    const html = buildOfferHtml({ ...data, signed: true });
    await pool.query("UPDATE submissions SET offer_status = 'SIGNED', offer_json = $2, offer_html = $3, offer_signed_by = $4, offer_signed_at = $5, last_updated = NOW() WHERE submission_id = $1",
      [req.params.id, JSON.stringify(data), html, signatory_name, signed_at]);
    await logComm({ submission_id: req.params.id, candidate_id: r.candidate_id, job_id: r.job_id, channel: 'DOCUMENT', to_role: 'hiring_manager', to_name: 'Hiring manager', subject: 'Offer letter signed — ' + r.candidate_name, body: 'Offer letter signed by ' + signatory_name + ' (' + signatory_title + '). Ready to send to the candidate.', sent_by: 'admin' });
    return res.json({ ok: true, offer_status: 'SIGNED', signatory_name, signatory_title, signed_at, html });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// SEND the signed offer to the candidate (DEMO-routed email; CC recruiter).
app.post('/submissions/:id/offer/send', async (req, res) => {
  try {
    await ensureSlots();
    const r = (await pool.query('SELECT candidate_id, candidate_name, job_id, job_title, client_name, offer_status, offer_json, submitted_by FROM submissions WHERE submission_id = $1', [req.params.id])).rows[0];
    if (!r) return res.status(404).json({ error: 'not found' });
    if (r.offer_status !== 'SIGNED') return res.status(400).json({ error: 'The offer must be signed by an admin before it can be sent.' });
    let data = {}; try { data = JSON.parse(r.offer_json); } catch (e) {}
    const candEmail = candidateEmail(r.candidate_id);
    const emailSubject = 'Your Employment Offer — ' + (r.job_title || 'a role') + ' at ' + COMPANY_NAME;
    const emailBody = 'Dear ' + (r.candidate_name || 'Candidate') + ',\n\nCongratulations! On behalf of ' + COMPANY_NAME + ', we are delighted to extend a formal offer of employment for the role of ' + (data.role_title || r.job_title || 'the position') + (r.client_name ? ' (assignment with ' + r.client_name + ')' : '') + '.\n\nYour signed offer letter is attached. It sets out your compensation, benefits, leave, the initial service commitment and the conditions of the offer. Please review it carefully.\n\nTo accept, kindly confirm by replying to this email. On acceptance, we will begin onboarding and will need the following documents to proceed:\n1. Identity & address proof (passport / national ID).\n2. Education and experience certificates.\n3. Recent photographs and completed joining forms (we will share these).\n4. Any documents required for visa processing — our visa partner (' + VISA_PARTNER_NAME + ') will guide you.\n\nIf you have any questions, our team is happy to help. We look forward to welcoming you aboard.\n\nBest regards,\n' + COMPANY_NAME + ' Talent Team';
    const cc = [r.submitted_by ? 'recruiter (' + r.submitted_by + ')' : null].filter(Boolean).join(', ');
    await pool.query("UPDATE submissions SET offer_status = 'SENT', last_updated = NOW() WHERE submission_id = $1", [req.params.id]);
    await notify({ candidate_id: r.candidate_id, recipient: 'candidate', type: 'OFFER', message: 'You have received an employment offer for ' + (r.job_title || 'a role') + '. Please review and confirm.' });
    await logComm({ submission_id: req.params.id, candidate_id: r.candidate_id, job_id: r.job_id, channel: 'EMAIL', to_role: 'candidate', to_name: r.candidate_name, to_email: candEmail, subject: emailSubject, body: emailBody, sent_by: req.body?.by || 'hiring_manager' });
    return res.json({ ok: true, offer_status: 'SENT', emailSubject, emailBody, candidateEmail: candEmail, cc });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ACCEPT -> PLACED (candidate accepted). Fires onboarding + visa email, marks placement, fills the job.
app.post('/submissions/:id/offer/accept', async (req, res) => {
  try {
    await ensureSlots();
    const s = (await pool.query('SELECT candidate_id, candidate_name, job_id, job_title, client_name, submitted_by, offer_status FROM submissions WHERE submission_id = $1', [req.params.id])).rows[0];
    if (!s) return res.status(404).json({ error: 'not found' });
    const candEmail = candidateEmail(s.candidate_id);
    const cArgs = { submission_id: req.params.id, candidate_id: s.candidate_id, job_id: s.job_id, sent_by: req.body?.by || 'recruiter' };
    await pool.query("UPDATE submissions SET status = 'PLACED', offer_status = 'ACCEPTED', last_updated = NOW() WHERE submission_id = $1", [req.params.id]);
    try { await pool.query("UPDATE candidates SET visa_status = 'PLACEMENT_ACTIVE', last_updated = NOW() WHERE candidate_id = $1", [s.candidate_id]); } catch (e) {}
    // Close the job only when all openings are filled.
    let jobFilled = false;
    try {
      const placedCount = Number((await pool.query("SELECT COUNT(*) AS n FROM submissions WHERE job_id = $1 AND status = 'PLACED'", [s.job_id])).rows[0]?.n || 0);
      const openings = Number((await pool.query('SELECT number_of_positions FROM jobs WHERE job_id = $1', [s.job_id])).rows[0]?.number_of_positions || 1);
      if (placedCount >= openings) { await pool.query("UPDATE jobs SET status = 'Filled' WHERE job_id = $1", [s.job_id]); jobFilled = true; }
    } catch (e) {}
    const congratsSubject = 'Welcome Aboard — Onboarding Next Steps for ' + (s.job_title || 'a role');
    const congratsBody = 'Dear ' + s.candidate_name + ',\n\nThank you for accepting your offer with ' + COMPANY_NAME + ' for ' + (s.job_title || 'a role') + (s.client_name ? ' at ' + s.client_name : '') + '. We are thrilled to have you join us.\n\nTo begin onboarding, our team will guide you through:\n1. Document Verification — please keep your identity and education documents ready.\n2. Application Forms — we will share the joining application for you to complete.\n3. Visa Documentation — our visa partner (' + VISA_PARTNER_NAME + ') will collect and process the required documents and keep you updated.\n\nWe will contact you shortly with the details. Congratulations and welcome aboard!\n\nBest regards,\n' + COMPANY_NAME + ' Talent Team\n\nCc: Recruiter (' + (s.submitted_by || 'assigned recruiter') + '), Visa Application Team (' + VISA_TEAM_EMAIL + ')';
    await notify({ candidate_id: s.candidate_id, recipient: 'candidate', type: 'PLACED', message: 'Offer accepted — you have been placed for ' + (s.job_title || 'a role') + '. Visa processing will begin.' });
    await notify({ candidate_id: s.candidate_id, recipient: s.submitted_by || 'recruiter', type: 'PLACED', message: s.candidate_name + ' accepted the offer and is now placed for ' + (s.job_title || 'a role') + '.' });
    await logComm({ ...cArgs, channel: 'EMAIL', to_role: 'candidate', to_name: s.candidate_name, to_email: candEmail, subject: congratsSubject, body: congratsBody, cc: VISA_TEAM_EMAIL });
    return res.json({ ok: true, status: 'PLACED', offer_status: 'ACCEPTED', candidateEmail: candEmail, emailSubject: congratsSubject, emailBody: congratsBody, cc: VISA_TEAM_EMAIL, jobFilled });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// DECLINE
app.post('/submissions/:id/offer/decline', async (req, res) => {
  try {
    await ensureSlots();
    const s = (await pool.query('SELECT candidate_id, candidate_name, job_id, job_title, submitted_by FROM submissions WHERE submission_id = $1', [req.params.id])).rows[0];
    if (!s) return res.status(404).json({ error: 'not found' });
    await pool.query("UPDATE submissions SET status = 'OFFER_DECLINED', offer_status = 'DECLINED', last_updated = NOW() WHERE submission_id = $1", [req.params.id]);
    const reason = req.body?.reason || '';
    await notify({ candidate_id: s.candidate_id, recipient: s.submitted_by || 'recruiter', type: 'OFFER_DECLINED', message: s.candidate_name + ' declined the offer for ' + (s.job_title || 'a role') + (reason ? ': ' + reason : '') + '.' });
    await logComm({ submission_id: req.params.id, candidate_id: s.candidate_id, job_id: s.job_id, channel: 'UPDATE', to_role: 'recruiter', to_name: s.submitted_by || 'recruiter', subject: 'Offer declined — ' + s.candidate_name, body: 'Offer declined' + (reason ? ': ' + reason : '') + '.', sent_by: req.body?.by || 'recruiter' });
    return res.json({ ok: true, status: 'OFFER_DECLINED', offer_status: 'DECLINED' });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ADMIN signature queue — offers drafted by the hiring manager, awaiting the authorized signatory.
app.get('/offers/pending-signature', async (req, res) => {
  try {
    await ensureSlots();
    const { rows } = await pool.query(
      `SELECT submission_id, candidate_id, candidate_name, job_id, job_title, client_name, offer_region, offer_status, submitted_by, last_updated
         FROM submissions WHERE offer_status = 'DRAFT' AND status = 'OFFER' ORDER BY last_updated DESC`);
    return res.json({ pending: rows });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// HIRING MANAGER offer queue — every candidate currently at the OFFER stage (prepare / signed / send).
app.get('/offers/active', async (req, res) => {
  try {
    await ensureSlots();
    const hm = String(req.query.hm || '').trim();
    const scoped = String(req.query.role || '') !== 'admin' && hm;
    const { rows } = await pool.query(
      `SELECT s.submission_id, s.candidate_id, s.candidate_name, s.job_id, s.job_title, s.client_name, s.offer_region, s.offer_status, s.submitted_by, s.last_updated
         FROM submissions s LEFT JOIN jobs j ON j.job_id = s.job_id
        WHERE s.status = 'OFFER' ${scoped ? 'AND j.hiring_manager = $1' : ''}
        ORDER BY s.last_updated DESC`, scoped ? [hm] : []);
    return res.json({ offers: rows });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ============================================================
// INTERVIEW SLOT ORCHESTRATOR — propose non-overlapping slots, candidate confirms one.
// (Care 360-style: the agent suggests slots and never double-books a candidate or recruiter.)
// ============================================================
function meetLink() {
  const p = () => Math.random().toString(36).slice(2, 6);
  return 'https://meet.google.com/' + p().slice(0, 3) + '-' + p() + '-' + p().slice(0, 3);
}
function clientEmailFor(name) { return (String(name || 'client').toLowerCase().replace(/[^a-z0-9]+/g, '') || 'client') + '@client.com'; }
const VISA_TEAM_EMAIL = 'visa-team@fragomen.com'; // third-party visa agency (Fragomen) placeholder
const VISA_PARTNER_NAME = 'Fragomen';
// A REAL, clickable Google Calendar event link (prefilled title/time/details; adding it creates a Google Meet).
function calendarLink(startISO, durationMin, title, details) {
  try {
    const start = new Date(startISO);
    const end = new Date(start.getTime() + (durationMin || 30) * 60000);
    const fmt = (d) => d.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
    const p = new URLSearchParams({ action: 'TEMPLATE', text: title || 'Interview', dates: fmt(start) + '/' + fmt(end), details: details || '', add: '' });
    return 'https://calendar.google.com/calendar/render?' + p.toString();
  } catch (e) { return ''; }
}
// Suggest N future business slots (skips weekends), at 10:00 / 12:00 / 15:00 / 17:00 IST-ish.
function suggestSlots(n = 3) {
  const hours = [10, 12, 15, 17];
  const out = [];
  const d = new Date();
  let hi = 0;
  while (out.length < n) {
    d.setHours(hours[hi], 0, 0, 0);
    if (d > new Date() && d.getDay() !== 0 && d.getDay() !== 6) out.push(new Date(d).toISOString());
    hi++;
    if (hi >= hours.length) { hi = 0; d.setDate(d.getDate() + 1); }
  }
  return out;
}
// True if a proposed time clashes with an existing CONFIRMED slot for this candidate or proposer.
async function slotClashes(candidateId, proposedBy, isoTime, durationMin) {
  const t = new Date(isoTime).getTime();
  const winMs = (durationMin || 30) * 60000;
  const { rows } = await pool.query(
    `SELECT slot_time, duration_min FROM interview_slots
      WHERE status = 'CONFIRMED' AND (candidate_id = $1 OR proposed_by = $2)`, [candidateId, proposedBy || '']);
  return rows.some(r => {
    const rt = new Date(r.slot_time).getTime();
    const rWin = (r.duration_min || 30) * 60000;
    return Math.abs(rt - t) < Math.max(winMs, rWin);
  });
}

// Propose slots for a submission (recruiter call or client interview). Auto-suggests if none given.
app.post('/submissions/:id/slots/propose', async (req, res) => {
  try {
    await ensureSlots();
    const { kind = 'recruiter', slots, proposed_by, duration_min } = req.body || {};
    const s = (await pool.query('SELECT candidate_id, candidate_name, job_id, job_title, client_name FROM submissions WHERE submission_id = $1', [req.params.id])).rows[0];
    if (!s) return res.status(404).json({ error: 'not found' });
    const dur = parseInt(duration_min, 10) || (kind === 'client' ? 60 : 45);
    let wanted = Array.isArray(slots) && slots.length ? slots : suggestSlots(3);
    // clear previous still-PROPOSED slots of this kind (re-propose replaces them)
    await pool.query("UPDATE interview_slots SET status = 'CANCELLED' WHERE submission_id = $1 AND kind = $2 AND status = 'PROPOSED'", [req.params.id, kind]);
    const accepted = [], skipped = [];
    for (const iso of wanted) {
      if (await slotClashes(s.candidate_id, proposed_by, iso, dur)) { skipped.push(iso); continue; }
      const slot_id = 'SLOT' + Date.now().toString(36).slice(-6) + Math.random().toString(36).slice(2, 5);
      await pool.query(`INSERT INTO interview_slots (slot_id, submission_id, candidate_id, job_id, kind, slot_time, duration_min, status, proposed_by) VALUES ($1,$2,$3,$4,$5,$6,$7,'PROPOSED',$8)`,
        [slot_id, req.params.id, s.candidate_id, s.job_id, kind, iso, dur, proposed_by || 'recruiter']);
      accepted.push({ slot_id, slot_time: iso });
    }
    // email the candidate the options
    const candEmail = candidateEmail(s.candidate_id);
    const opts = accepted.map((a, i) => `  ${i + 1}. ${new Date(a.slot_time).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })}`).join('\n');
    const label = kind === 'client' ? 'client interview' : 'recruiter call';
    const emailSubject = `Choose a time for your ${label} — ${s.job_title || 'a role'}`;
    const emailBody = 'Dear ' + s.candidate_name + ',\n\nPlease pick one of the available time slots for your ' + label + (s.client_name && kind === 'client' ? ' with ' + s.client_name : '') + ':\n\n' + opts + '\n\nReply with your preferred slot and we will confirm it with the meeting link.\n\nBest regards,\nRecruit 360 Team';
    await logComm({ submission_id: req.params.id, candidate_id: s.candidate_id, job_id: s.job_id, channel: 'EMAIL', to_role: 'candidate', to_name: s.candidate_name, to_email: candEmail, subject: emailSubject, body: emailBody, sent_by: proposed_by || 'recruiter' });
    await notify({ candidate_id: s.candidate_id, recipient: 'candidate', type: 'SLOTS', message: `Please choose a time for your ${label}.` });
    return res.json({ ok: true, kind, proposed: accepted, skipped_overlaps: skipped, emailSubject, emailBody, candidateEmail: candEmail });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// Directly FIX a manual slot (recruiter sets the exact time) — no candidate choosing.
// Books it as CONFIRMED, creates the meeting link, and emails the candidate the fixed time + link.
app.post('/submissions/:id/slots/fix', async (req, res) => {
  try {
    await ensureSlots();
    const { kind = 'recruiter', slot_time, fixed_by, duration_min } = req.body || {};
    if (!slot_time) return res.status(400).json({ error: 'slot_time is required' });
    const s = (await pool.query('SELECT candidate_id, candidate_name, job_id, job_title, client_name, submitted_by FROM submissions WHERE submission_id = $1', [req.params.id])).rows[0];
    if (!s) return res.status(404).json({ error: 'not found' });
    const dur = parseInt(duration_min, 10) || (kind === 'client' ? 60 : 45);
    if (await slotClashes(s.candidate_id, fixed_by, slot_time, dur)) {
      return res.status(409).json({ error: 'That time clashes with an existing confirmed interview. Pick a different time.' });
    }
    // clear any pending proposed options of this kind, then book the fixed one as CONFIRMED
    await pool.query("UPDATE interview_slots SET status = 'CANCELLED' WHERE submission_id = $1 AND kind = $2 AND status = 'PROPOSED'", [req.params.id, kind]);
    const link = meetLink();
    const slot_id = 'SLOT' + Date.now().toString(36).slice(-6) + Math.random().toString(36).slice(2, 5);
    await pool.query(`INSERT INTO interview_slots (slot_id, submission_id, candidate_id, job_id, kind, slot_time, duration_min, status, proposed_by, meet_link) VALUES ($1,$2,$3,$4,$5,$6,$7,'CONFIRMED',$8,$9)`,
      [slot_id, req.params.id, s.candidate_id, s.job_id, kind, slot_time, dur, fixed_by || 'recruiter', link]);
    await pool.query('UPDATE submissions SET interview_link = $1, interview_date = $2, last_updated = NOW() WHERE submission_id = $3', [link, slot_time, req.params.id]).catch(() => {});
    const candEmail = candidateEmail(s.candidate_id);
    const label = kind === 'client' ? 'client interview' : 'recruiter call';
    const when = new Date(slot_time).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' });
    const clientCc = kind === 'client' ? clientEmailFor(s.client_name) : '';
    const calLink = calendarLink(slot_time, dur, label.replace(/^./, c => c.toUpperCase()) + ' — ' + (s.job_title || 'role'), 'Candidate: ' + s.candidate_name + (s.client_name ? ' | Client: ' + s.client_name : '') + ' | Join: ' + link);
    const emailSubject = `Your ${label} is scheduled — ${s.job_title || 'a role'}`;
    const emailBody = 'Dear ' + s.candidate_name + ',\n\nYour ' + label + (s.client_name && kind === 'client' ? ' with ' + s.client_name : '') + ' has been scheduled for ' + when + ' (IST).\n\nJoin using this link:\n' + link + '\n\nAdd it to your calendar (with Google Meet):\n' + calLink + '\n\nNo reply is needed — please join at the scheduled time.\n\nBest regards,\nRecruit 360 Team' + (clientCc ? '\n\n(The client interview panel is copied on this invite.)' : '');
    await logComm({ submission_id: req.params.id, candidate_id: s.candidate_id, job_id: s.job_id, channel: 'EMAIL', to_role: 'candidate', to_name: s.candidate_name, to_email: candEmail, subject: emailSubject, body: emailBody + (clientCc ? '\nCc: ' + clientCc : ''), sent_by: fixed_by || 'recruiter' });
    await notify({ candidate_id: s.candidate_id, recipient: 'candidate', type: 'INTERVIEW', message: `Your ${label} is fixed for ${when}. Link: ${link}` });
    if (s.submitted_by) await notify({ candidate_id: s.candidate_id, recipient: s.submitted_by, type: 'INTERVIEW', message: `${s.candidate_name}'s ${label} fixed for ${when}.` });
    return res.json({ ok: true, slot_id, status: 'CONFIRMED', meet_link: link, slot_time, emailSubject, emailBody, candidateEmail: candEmail, cc: clientCc });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// Taken (already-booked) slot times for a date — so a booked slot is hidden from every candidate.
app.get('/slots/taken', async (req, res) => {
  try {
    await ensureSlots();
    const date = (req.query.date || '').trim();
    if (!date) return res.json({ taken: [] });
    const params = [date];
    let sql = "SELECT slot_time FROM interview_slots WHERE status = 'CONFIRMED' AND slot_time::date = $1";
    if (req.query.kind) { sql += ' AND kind = $2'; params.push(req.query.kind); }
    const { rows } = await pool.query(sql, params);
    return res.json({ taken: rows.map(r => new Date(r.slot_time).toISOString()) });
  } catch (e) { return res.status(500).json({ error: e.message, taken: [] }); }
});

// List slots for a submission
app.get('/submissions/:id/slots', async (req, res) => {
  try {
    await ensureSlots();
    const { rows } = await pool.query('SELECT slot_id, kind, slot_time, duration_min, status, meet_link, proposed_by FROM interview_slots WHERE submission_id = $1 ORDER BY slot_time ASC', [req.params.id]);
    return res.json({ slots: rows });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// Confirm one slot -> books it, cancels the other proposed options, creates the meet link.
app.post('/slots/:slotId/confirm', async (req, res) => {
  try {
    await ensureSlots();
    const slot = (await pool.query('SELECT * FROM interview_slots WHERE slot_id = $1', [req.params.slotId])).rows[0];
    if (!slot) return res.status(404).json({ error: 'slot not found' });
    if (await slotClashes(slot.candidate_id, slot.proposed_by, slot.slot_time, slot.duration_min)) {
      return res.status(409).json({ error: 'That time now clashes with another confirmed interview. Please pick a different slot.' });
    }
    const link = meetLink();
    await pool.query("UPDATE interview_slots SET status = 'CONFIRMED', meet_link = $1 WHERE slot_id = $2", [link, req.params.slotId]);
    await pool.query("UPDATE interview_slots SET status = 'CANCELLED' WHERE submission_id = $1 AND kind = $2 AND status = 'PROPOSED' AND slot_id <> $3", [slot.submission_id, slot.kind, req.params.slotId]);
    const s = (await pool.query('SELECT candidate_id, candidate_name, job_id, job_title, client_name, submitted_by FROM submissions WHERE submission_id = $1', [slot.submission_id])).rows[0] || {};
    await pool.query('UPDATE submissions SET interview_link = $1, interview_date = $2, last_updated = NOW() WHERE submission_id = $3', [link, slot.slot_time, slot.submission_id]).catch(() => {});
    const candEmail = candidateEmail(slot.candidate_id);
    const label = slot.kind === 'client' ? 'client interview' : 'recruiter call';
    const when = new Date(slot.slot_time).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' });
    const clientCc = slot.kind === 'client' ? clientEmailFor(s.client_name) : '';
    const calLink = calendarLink(slot.slot_time, slot.duration_min, label.replace(/^./, c => c.toUpperCase()) + ' — ' + (s.job_title || 'role'), 'Candidate: ' + (s.candidate_name || '') + (s.client_name ? ' | Client: ' + s.client_name : '') + ' | Join: ' + link);
    const emailSubject = `Your ${label} is confirmed — ${s.job_title || 'a role'}`;
    const emailBody = 'Dear ' + (s.candidate_name || 'Candidate') + ',\n\nYour ' + label + (s.client_name && slot.kind === 'client' ? ' with ' + s.client_name : '') + ' is confirmed for ' + when + ' (IST).\n\nJoin using this link:\n' + link + '\n\nAdd it to your calendar (with Google Meet):\n' + calLink + '\n\nBest regards,\nRecruit 360 Team' + (clientCc ? '\n\n(The client interview panel is copied on this invite.)' : '');
    await logComm({ submission_id: slot.submission_id, candidate_id: slot.candidate_id, job_id: slot.job_id, channel: 'EMAIL', to_role: 'candidate', to_name: s.candidate_name, to_email: candEmail, subject: emailSubject, body: emailBody + (clientCc ? '\nCc: ' + clientCc : ''), sent_by: 'recruiter' });
    await notify({ candidate_id: slot.candidate_id, recipient: 'candidate', type: 'INTERVIEW', message: `Your ${label} is confirmed for ${when}. Link: ${link}` });
    if (s.submitted_by) await notify({ candidate_id: slot.candidate_id, recipient: s.submitted_by, type: 'INTERVIEW', message: `${s.candidate_name}'s ${label} confirmed for ${when}.` });
    return res.json({ ok: true, slot_id: req.params.slotId, status: 'CONFIRMED', meet_link: link, slot_time: slot.slot_time, emailSubject, emailBody, candidateEmail: candEmail, cc: clientCc });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// Cancel / postpone a slot (postpone = cancel, then propose again)
app.post('/slots/:slotId/cancel', async (req, res) => {
  try {
    const { reason, by } = req.body || {};
    await ensureSlots();
    const slot = (await pool.query('SELECT * FROM interview_slots WHERE slot_id = $1', [req.params.slotId])).rows[0];
    if (!slot) return res.status(404).json({ error: 'slot not found' });
    await pool.query("UPDATE interview_slots SET status = 'CANCELLED' WHERE slot_id = $1", [req.params.slotId]);
    const s = (await pool.query('SELECT candidate_name, job_title FROM submissions WHERE submission_id = $1', [slot.submission_id])).rows[0] || {};
    const label = slot.kind === 'client' ? 'client interview' : 'recruiter call';
    await notify({ candidate_id: slot.candidate_id, recipient: 'candidate', type: 'INTERVIEW', message: `Your ${label} was cancelled${reason ? ': ' + reason : ''}. We will share new times shortly.` });
    await logComm({ submission_id: slot.submission_id, candidate_id: slot.candidate_id, job_id: slot.job_id, channel: 'UPDATE', to_role: 'candidate', to_name: s.candidate_name, subject: label + ' cancelled', body: `The ${label} slot was cancelled${reason ? ': ' + reason : ''}.`, sent_by: by || 'recruiter' });
    return res.json({ ok: true, status: 'CANCELLED' });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ============================================================
// CLIENT COMMUNICATION — separate from candidate comms (panel request + client scheduling).
// ============================================================
// Auto-draft the client email requesting a panel, sharing the candidate + resume summary.
app.post('/submissions/:id/client/request-panel', async (req, res) => {
  try {
    const { requested_by } = req.body || {};
    await ensureSlots();
    const s = (await pool.query('SELECT candidate_id, candidate_name, job_id, job_title, client_name, resume_summary, current_ctc, expected_ctc FROM submissions WHERE submission_id = $1', [req.params.id])).rows[0];
    if (!s) return res.status(404).json({ error: 'not found' });
    const clientEmail = (String(s.client_name || 'client').toLowerCase().replace(/[^a-z0-9]+/g, '') || 'client') + '@client.com';
    const summary = s.resume_summary || (s.candidate_name + ' — strong candidate for ' + (s.job_title || 'the role') + '.');
    const ctcLine = (s.current_ctc || s.expected_ctc) ? `\nCompensation: current ${s.current_ctc || 'n/a'}, expected ${s.expected_ctc || 'n/a'}.\n` : '\n';
    const emailSubject = `Candidate for ${s.job_title || 'your role'} — please set up the interview panel`;
    const emailBody = 'Dear ' + (s.client_name || 'Client') + ' team,\n\nFor the ' + (s.job_title || 'role') + ', we have identified a suitable candidate, ' + s.candidate_name + ', who has cleared our internal recruiter screening. We would like to request you to set up the interview panel and share your available interview slots.\n\nCandidate summary:\n' + summary + ctcLine + '\nPlease reply with:\n1. The interview panel members.\n2. A few available interview time slots.\n\nWe will then coordinate the interview with the candidate and share the confirmed schedule. After the interview, please let us know your decision (selected / not selected).\n\nBest regards,\nRecruit 360 Team';
    await logComm({ submission_id: req.params.id, candidate_id: s.candidate_id, job_id: s.job_id, channel: 'CLIENT', to_role: 'client', to_name: s.client_name || 'client', to_email: clientEmail, subject: emailSubject, body: emailBody, sent_by: requested_by || 'recruiter' });
    return res.json({ ok: true, emailSubject, emailBody, clientEmail });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// Record the panel details the client sent back.
app.post('/submissions/:id/client/panel', async (req, res) => {
  try {
    const { panel, saved_by } = req.body || {};
    await ensureSlots();
    const s = (await pool.query('SELECT candidate_id, candidate_name, job_id, client_name FROM submissions WHERE submission_id = $1', [req.params.id])).rows[0];
    if (!s) return res.status(404).json({ error: 'not found' });
    await pool.query('UPDATE submissions SET client_panel = $1, last_updated = NOW() WHERE submission_id = $2', [panel || '', req.params.id]).catch(() => {});
    await logComm({ submission_id: req.params.id, candidate_id: s.candidate_id, job_id: s.job_id, channel: 'CLIENT', to_role: 'client', to_name: s.client_name || 'client', subject: 'Interview panel received', body: 'Panel details from ' + (s.client_name || 'client') + ':\n' + (panel || ''), sent_by: saved_by || 'recruiter' });
    return res.json({ ok: true, panel: panel || '' });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- BULK APPROVALS (hiring manager) ----------
// NEW FLOW: resume + summary already done -> APPROVED shortlists the candidate, moves to the
// recruiter-call stage, and emails the shortlisting note. No resume-upload invite here.
app.post('/submissions/bulk-approval', async (req, res) => {
  try {
    const { submission_ids, decision, approver } = req.body; // decision: APPROVED | REJECTED
    if (!Array.isArray(submission_ids) || !submission_ids.length) return res.status(400).json({ error: 'submission_ids required' });
    const emails = [];
    for (const id of submission_ids) {
      const sub = (await pool.query('SELECT candidate_id, candidate_name, job_id, job_title, client_name, submitted_by FROM submissions WHERE submission_id = $1', [id])).rows[0];
      if (!sub) continue;
      const candEmail = candidateEmail(sub.candidate_id);
      if (decision === 'APPROVED') {
        await pool.query("UPDATE submissions SET status = 'RECRUITER_CALL', last_updated = NOW() WHERE submission_id = $1", [id]);
        const recruiterName = sub.submitted_by || 'your recruiter';
        const subject = 'You have been shortlisted — ' + (sub.job_title || 'a role');
        const body = 'Dear ' + sub.candidate_name + ',\n\nCongratulations! You have been shortlisted for the next rounds for ' + (sub.job_title || 'a role') + (sub.client_name ? ' at ' + sub.client_name : '') + '. ' + recruiterName + ' will guide you through the upcoming interviews and next steps, and will be in touch shortly to arrange your recruiter call.\n\nBest regards,\nRecruit 360 Team';
        await logComm({ submission_id: id, candidate_id: sub.candidate_id, job_id: sub.job_id, channel: 'EMAIL', to_role: 'candidate', to_name: sub.candidate_name, to_email: candEmail, subject, body, sent_by: approver || 'hiring_manager' });
        await notify({ candidate_id: sub.candidate_id, recipient: 'candidate', type: 'SHORTLISTED', message: 'You have been shortlisted for ' + (sub.job_title || 'a role') + '.' });
        await notify({ candidate_id: sub.candidate_id, recipient: sub.submitted_by || 'recruiter', type: 'APPROVED', message: sub.candidate_name + ' approved for ' + (sub.job_title || 'a role') + ' — schedule the recruiter call.' });
        emails.push({ name: sub.candidate_name, email: candEmail, subject, body });
      } else {
        await pool.query("UPDATE submissions SET status = 'REJECTED', last_updated = NOW() WHERE submission_id = $1", [id]);
        await notify({ candidate_id: sub.candidate_id, recipient: sub.submitted_by || 'recruiter', type: 'CANDIDATE_REJECTED', message: sub.candidate_name + ' was not approved for ' + (sub.job_title || 'a role') + '.' });
      }
    }
    // `emails` = the shortlisting emails just sent (shown to the HM); `invites` kept for compatibility.
    return res.json({ ok: true, count: submission_ids.length, decision, emails, invites: [] });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- Update CTC AFTER the interview process ----------
app.patch('/submissions/:id/ctc', async (req, res) => {
  try {
    const { current_ctc, expected_ctc } = req.body;
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS current_ctc TEXT');
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS expected_ctc TEXT');
    await pool.query('UPDATE submissions SET current_ctc = COALESCE($1,current_ctc), expected_ctc = COALESCE($2,expected_ctc), last_updated = NOW() WHERE submission_id = $3', [current_ctc || null, expected_ctc || null, req.params.id]);
    return res.json({ ok: true });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- VERIFY a candidate document is the CORRECT type (Document AI + Gemini) ----------
app.post('/documents/verify-type', async (req, res) => {
  try {
    const { fileBase64, mimeType, expectedType } = req.body;
    if (!fileBase64 || !expectedType) return res.status(400).json({ error: 'fileBase64 and expectedType required' });
    // 1) Read the document text with Document AI
    let text = '';
    try {
      const [result] = await docaiClient.processDocument({ name: PROCESSOR, rawDocument: { content: fileBase64, mimeType: mimeType || 'application/pdf' } });
      text = (result.document && result.document.text) || '';
    } catch (e) { return res.status(500).json({ error: 'Could not read the document', detail: e.message }); }
    if (!text.trim()) return res.json({ verified: false, reason: 'The document could not be read. Please upload a clear scan.' });

    // 2) Ask Gemini to classify + check it matches the expected type
    const prompt = `You are a document verification checker. A candidate uploaded a document that should be a "${expectedType}".
Read the extracted text below and decide: does it look like a genuine "${expectedType}"?
Guidance: A Passport has fields like 'Passport No', 'Republic of India', 'Nationality', 'Date of Birth', 'Place of Issue'. An Aadhaar card has a 12-digit Aadhaar number, 'Unique Identification Authority of India', 'Government of India'. Educational certificates mention a degree/university/marks. Experience letters mention employer/designation/duration.
Reply ONLY with JSON: {"matches": true/false, "detected": "what document it actually looks like", "reason": "one short sentence"}.

EXTRACTED TEXT:
${text.slice(0, 4000)}`;
    const gen = await genModel.generateContent(prompt);
    let out = gen.response.candidates[0].content.parts[0].text.replace(/```json|```/g, '').trim();
    let parsed;
    try { parsed = JSON.parse(out); } catch { parsed = { matches: false, detected: 'unknown', reason: 'Could not classify the document.' }; }
    return res.json({ verified: !!parsed.matches, detected: parsed.detected || '', reason: parsed.reason || '', expectedType });
  } catch (e) { return res.status(500).json({ error: 'Verification failed', detail: e.message }); }
});

// ---------- Self-healing schema: make sure supporting tables/columns exist ----------
// Lazy self-heal for the newer flow tables/columns — guarantees they exist on first use,
// even if the startup ensureSchema() was interrupted by a cold-start DB blip.
let _slotsReady = false;
async function ensureSlots() {
  if (_slotsReady) return;
  try {
    await pool.query(`CREATE TABLE IF NOT EXISTS interview_slots (
      slot_id       TEXT PRIMARY KEY,
      submission_id TEXT,
      candidate_id  TEXT,
      job_id        TEXT,
      kind          TEXT,
      slot_time     TIMESTAMPTZ,
      duration_min  INTEGER DEFAULT 30,
      status        TEXT DEFAULT 'PROPOSED',
      proposed_by   TEXT,
      meet_link     TEXT,
      created_at    TIMESTAMPTZ DEFAULT NOW()
    )`);
    await pool.query('CREATE INDEX IF NOT EXISTS idx_slots_submission ON interview_slots (submission_id)');
    await pool.query('CREATE INDEX IF NOT EXISTS idx_slots_time ON interview_slots (slot_time)');
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS resume_text TEXT');
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS resume_summary TEXT');
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS client_panel TEXT');
    // Offer letter (Stage 6-7): AI-generated letter, admin signature, region-aware terms.
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS offer_status TEXT');   // PENDING | DRAFT | SIGNED | SENT | ACCEPTED | DECLINED
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS offer_json TEXT');     // structured terms (JSON string)
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS offer_html TEXT');     // rendered letter (print-ready)
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS offer_region TEXT');
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS offer_signed_by TEXT');
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS offer_signed_at TIMESTAMPTZ');
    _slotsReady = true;
    console.log('Slot schema ensured (lazy).');
  } catch (e) { console.log('ensureSlots warn:', e.message); }
}

async function ensureSchema() {
  try {
    await pool.query(`CREATE TABLE IF NOT EXISTS notifications (
      notif_id     TEXT PRIMARY KEY,
      candidate_id TEXT,
      recipient    TEXT,
      type         TEXT,
      message      TEXT,
      read_flag    BOOLEAN DEFAULT FALSE,
      created_at   TIMESTAMPTZ DEFAULT NOW()
    )`);
    await pool.query('ALTER TABLE notifications ADD COLUMN IF NOT EXISTS read_flag BOOLEAN DEFAULT FALSE');
    await pool.query('CREATE INDEX IF NOT EXISTS idx_notifications_recipient ON notifications (recipient)');
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS resume_received BOOLEAN DEFAULT FALSE');
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS resume_requested BOOLEAN DEFAULT FALSE');
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS current_ctc TEXT');
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS expected_ctc TEXT');
    await pool.query(`CREATE TABLE IF NOT EXISTS user_settings (
      user_key   TEXT PRIMARY KEY,
      prefs      JSONB DEFAULT '{}',
      updated_at TIMESTAMPTZ DEFAULT NOW()
    )`);
    await pool.query(`CREATE TABLE IF NOT EXISTS communications (
      comm_id       TEXT PRIMARY KEY,
      submission_id TEXT,
      candidate_id  TEXT,
      job_id        TEXT,
      channel       TEXT,
      to_role       TEXT,
      to_name       TEXT,
      to_email      TEXT,
      subject       TEXT,
      body          TEXT,
      sent_by       TEXT,
      created_at    TIMESTAMPTZ DEFAULT NOW()
    )`);
    await pool.query('CREATE INDEX IF NOT EXISTS idx_comm_submission ON communications (submission_id)');
    // Saved assistant chats (per-user). Self-healing so the AI Assistant works on a fresh DB.
    await pool.query(`CREATE TABLE IF NOT EXISTS saved_chats (
      chat_id    TEXT PRIMARY KEY,
      chat_name  TEXT,
      messages   TEXT,
      user_email TEXT,
      created_at TIMESTAMPTZ DEFAULT NOW()
    )`);
    await pool.query('ALTER TABLE saved_chats ADD COLUMN IF NOT EXISTS user_email TEXT');
    await pool.query('CREATE INDEX IF NOT EXISTS idx_saved_chats_user ON saved_chats (user_email)');
    // New flow: resume text + AI summary for hiring-manager approval, and panel details for the client interview.
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS resume_text TEXT');
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS resume_summary TEXT');
    await pool.query('ALTER TABLE submissions ADD COLUMN IF NOT EXISTS client_panel TEXT');
    // Interview slot orchestrator — proposed/confirmed slots, used to prevent overlaps.
    await pool.query(`CREATE TABLE IF NOT EXISTS interview_slots (
      slot_id       TEXT PRIMARY KEY,
      submission_id TEXT,
      candidate_id  TEXT,
      job_id        TEXT,
      kind          TEXT,               -- 'recruiter' | 'client'
      slot_time     TIMESTAMPTZ,
      duration_min  INTEGER DEFAULT 30,
      status        TEXT DEFAULT 'PROPOSED',  -- PROPOSED | CONFIRMED | CANCELLED
      proposed_by   TEXT,
      meet_link     TEXT,
      created_at    TIMESTAMPTZ DEFAULT NOW()
    )`);
    await pool.query('CREATE INDEX IF NOT EXISTS idx_slots_submission ON interview_slots (submission_id)');
    await pool.query('CREATE INDEX IF NOT EXISTS idx_slots_time ON interview_slots (slot_time)');
    console.log('Schema ensured.');
  } catch (e) { console.log('ensureSchema warning:', e.message); }
}

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => { console.log('API on ' + PORT); ensureSchema(); });
