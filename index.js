import express from 'express';
import cors from 'cors';
import pg from 'pg';
import { VertexAI } from '@google-cloud/vertexai';
import { DocumentProcessorServiceClient } from '@google-cloud/documentai';
import { BigQuery } from '@google-cloud/bigquery';

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
    const { rows } = await pool.query('SELECT chat_id, chat_name, messages, created_at FROM saved_chats ORDER BY created_at DESC LIMIT 50');
    return res.json({ chats: rows });
  } catch (err) {
    return res.status(500).json({ error: 'Could not fetch chats', detail: err.message });
  }
});

app.post('/chats', async (req, res) => {
  try {
    const { chat_name, messages } = req.body;
    if (!chat_name || !messages) return res.status(400).json({ error: 'chat_name and messages are required' });
    const chat_id = 'CHAT' + Date.now().toString().slice(-10);
    await pool.query('INSERT INTO saved_chats (chat_id, chat_name, messages, created_at) VALUES ($1,$2,$3,NOW())',
      [chat_id, chat_name, typeof messages === 'string' ? messages : JSON.stringify(messages)]);
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

// Rename a saved chat
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
    if (!cRes.rows.length) return res.status(404).json({ error: 'Candidate not found' });
    const c = cRes.rows[0];

    let job = null;
    if (job_id) {
      const jRes = await pool.query('SELECT job_id, title, location, client, openings FROM jobs WHERE job_id = $1', [job_id]);
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
    const { status } = req.body;
    const allowed = ['SUBMITTED', 'CLIENT_REVIEW', 'SHORTLISTED', 'REJECTED', 'INTERVIEW_SCHEDULED', 'INTERVIEWED', 'OFFERED', 'OFFER_ACCEPTED', 'OFFER_DECLINED', 'PLACED'];
    if (!allowed.includes(status)) return res.status(400).json({ error: 'Invalid status', allowed });

    // ENFORCE ORDER: you can only move forward one step, or reject/decline from the current stage.
    const ORDER = ['SUBMITTED', 'CLIENT_REVIEW', 'SHORTLISTED', 'INTERVIEW_SCHEDULED', 'INTERVIEWED', 'OFFERED', 'OFFER_ACCEPTED', 'PLACED'];
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
      if (ni !== ci + 1) {
        const expected = ORDER[ci + 1] ? ORDER[ci + 1].replace(/_/g, ' ') : 'none';
        return res.status(400).json({ error: 'Steps must be followed in order. The next step for this candidate is: ' + expected + '.', current, next: ORDER[ci + 1] || null });
      }
    }

    await pool.query('UPDATE submissions SET status = $1, last_updated = NOW() WHERE submission_id = $2', [status, req.params.id]);

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
    const cur = await pool.query('SELECT status, candidate_id, candidate_name, job_title, client_name, submitted_by FROM submissions WHERE submission_id = $1', [req.params.id]);
    if (!cur.rows.length) return res.status(404).json({ error: 'Submission not found' });
    const sub = cur.rows[0];
    if (!outcome && sub.status !== 'SHORTLISTED' && sub.status !== 'INTERVIEW_SCHEDULED') {
      return res.status(400).json({ error: 'The candidate must be shortlisted before scheduling an interview.' });
    }
    let status = 'INTERVIEW_SCHEDULED';
    if (outcome === 'done') status = 'INTERVIEWED';

    // Generate a meeting link (demo — a Google Meet style link)
    const meetCode = Math.random().toString(36).slice(2, 5) + '-' + Math.random().toString(36).slice(2, 6) + '-' + Math.random().toString(36).slice(2, 5);
    const interview_link = outcome === 'done' ? undefined : 'https://meet.google.com/' + meetCode;

    if (interview_link) {
      await pool.query(
        'UPDATE submissions SET interview_date = $1, interview_notes = $2, interview_link = $3, status = $4, last_updated = NOW() WHERE submission_id = $5',
        [interview_date || null, interview_notes || '', interview_link, status, req.params.id]
      );
      // Simulate sending the interview invite email (recorded as a notification)
      await notify({ candidate_id: sub.candidate_id, recipient: 'candidate', type: 'INTERVIEW', message: 'Interview scheduled for ' + (sub.job_title || 'a role') + (sub.client_name ? ' at ' + sub.client_name : '') + (interview_date ? ' on ' + interview_date : '') + '. Meeting link: ' + interview_link });
      await notify({ candidate_id: sub.candidate_id, recipient: sub.submitted_by, type: 'INTERVIEW', message: 'Interview invite sent to ' + sub.candidate_name + ' for ' + (sub.job_title || 'a role') + '.' });
    } else {
      await pool.query(
        'UPDATE submissions SET interview_notes = $1, status = $2, last_updated = NOW() WHERE submission_id = $3',
        [interview_notes || '', status, req.params.id]
      );
    }
    return res.json({ ok: true, submission_id: req.params.id, status, interview_link: interview_link || null, email_sent: !!interview_link });
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
      'SELECT submission_id, candidate_id, candidate_name, status, screening_notes, submitted_by, created_at FROM submissions WHERE job_id = $1 ORDER BY created_at DESC',
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
    const { candidate_id, submitted_by, screening_notes } = req.body;
    if (!candidate_id) return res.status(400).json({ error: 'candidate_id is required' });
    const jRes = await pool.query('SELECT job_id, title, client FROM jobs WHERE job_id = $1', [req.params.jobId]);
    if (!jRes.rows.length) return res.status(404).json({ error: 'Job not found' });
    const job = jRes.rows[0];
    const cRes = await pool.query('SELECT full_name FROM candidates WHERE candidate_id = $1', [candidate_id]);
    if (!cRes.rows.length) return res.status(404).json({ error: 'Candidate not found' });
    // duplicate check
    const dup = await pool.query('SELECT submission_id FROM submissions WHERE candidate_id = $1 AND job_id = $2', [candidate_id, req.params.jobId]);
    if (dup.rows.length) return res.status(409).json({ error: 'duplicate', message: 'This candidate is already submitted to this job.' });
    const submission_id = 'SUB' + Date.now().toString().slice(-10);
    await pool.query(
      'INSERT INTO submissions (submission_id, candidate_id, candidate_name, job_id, job_title, client_name, status, screening_notes, submitted_by, created_at, last_updated) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,NOW(),NOW())',
      [submission_id, candidate_id, cRes.rows[0].full_name, req.params.jobId, job.title, job.client || '', 'SUBMITTED', screening_notes || '', submitted_by || '']
    );
    return res.json({ ok: true, submission_id, candidate_name: cRes.rows[0].full_name });
  } catch (err) {
    return res.status(500).json({ error: 'Could not submit candidate', detail: err.message });
  }
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

    // 2) Submissions for this job
    const sRes = await pool.query(
      `SELECT submission_id, candidate_id, candidate_name, status, screening_notes, submitted_by, created_at
         FROM submissions WHERE job_id = $1 ORDER BY created_at DESC`,
      [req.params.jobId]
    );

    // 3) Interviews (submissions with interview data)
    const iRes = await pool.query(
      `SELECT submission_id, candidate_name, interview_date, interview_notes, status
         FROM submissions WHERE job_id = $1 AND status IN ('INTERVIEW_SCHEDULED','INTERVIEWED')
         ORDER BY interview_date DESC`,
      [req.params.jobId]
    );

    // 4) Placements (submissions that reached placed/offer-accepted)
    const pRes = await pool.query(
      `SELECT submission_id, candidate_name, status, created_at
         FROM submissions WHERE job_id = $1 AND status IN ('OFFER_ACCEPTED','PLACED')
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

// ---------- LIST jobs with the requisition fields (for the admin/HM list) ----------

  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
});


const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log('API on ' + PORT));
