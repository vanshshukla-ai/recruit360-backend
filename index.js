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
    const { status, scenario } = req.body;
    const allowed = ['PENDING_HM_APPROVAL', 'SUBMITTED', 'RECRUITER_CALL', 'HR_INTERVIEW', 'CLIENT_INTERVIEW', 'OFFER', 'PLACED', 'REJECTED',
                     'CLIENT_CONFIRM', 'TRAINING', 'CLIENT_DECISION', 'CLIENT_REVIEW', 'SHORTLISTED', 'INTERVIEW_SCHEDULED', 'INTERVIEWED', 'OFFERED', 'OFFER_ACCEPTED', 'OFFER_DECLINED'];
    if (!allowed.includes(status)) return res.status(400).json({ error: 'Invalid status', allowed });

    // ENFORCE ORDER: you can only move forward one step, or reject/decline from the current stage.
    const ORDER = ['PENDING_HM_APPROVAL', 'SUBMITTED', 'RECRUITER_CALL', 'HR_INTERVIEW', 'CLIENT_INTERVIEW', 'OFFER', 'PLACED'];
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
    const cur = await pool.query('SELECT status, candidate_id, candidate_name, job_title, client_name, submitted_by FROM submissions WHERE submission_id = $1', [req.params.id]);
    if (!cur.rows.length) return res.status(404).json({ error: 'Submission not found' });
    const sub = cur.rows[0];
    const schedulable = ['RECRUITER_CALL', 'CLIENT_INTERVIEW', 'SHORTLISTED', 'INTERVIEW_SCHEDULED', 'SUBMITTED', 'CLIENT_CONFIRM', 'TRAINING'];
    if (!outcome && !schedulable.includes(sub.status)) {
      return res.status(400).json({ error: 'The candidate is not at a stage where an interview/call can be scheduled.' });
    }
    // Generate a meeting link (Google Meet style) — do NOT change the stage; stages advance via the step buttons.
    const meetCode = Math.random().toString(36).slice(2, 5) + '-' + Math.random().toString(36).slice(2, 6) + '-' + Math.random().toString(36).slice(2, 5);
    const interview_link = 'https://meet.google.com/' + meetCode;

    // Build the email content for the candidate (returned so the UI can open mailto)
    const candEmail = (sub.candidate_id || '').toLowerCase() + '@example.com';
    const emailSubject = 'Your interview is scheduled — ' + (sub.job_title || 'a role');
    const emailBody = 'Dear ' + (sub.candidate_name || 'Candidate') + ',\n\nYour interview' + (sub.client_name ? ' with ' + sub.client_name : '') + ' has been scheduled' + (interview_date ? ' for ' + interview_date : '') + '.\n\nPlease join using this link:\n' + interview_link + '\n\nBest regards,\nRecruit 360 Team';

    await pool.query(
      'UPDATE submissions SET interview_date = $1, interview_notes = $2, interview_link = $3, last_updated = NOW() WHERE submission_id = $4',
      [interview_date || null, interview_notes || '', interview_link, req.params.id]
    );
    try {
      await notify({ candidate_id: sub.candidate_id, recipient: 'candidate', type: 'INTERVIEW', message: 'Interview scheduled for ' + (sub.job_title || 'a role') + (interview_date ? ' on ' + interview_date : '') + '. Link: ' + interview_link });
    } catch(e){}
    return res.json({ ok: true, submission_id: req.params.id, interview_link, email_sent: true, emailSubject, emailBody, candidateEmail: candEmail });
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
    const newStatus = decision === 'APPROVED' ? 'SUBMITTED' : 'REJECTED';
    await pool.query('UPDATE submissions SET status = $1, last_updated = NOW() WHERE submission_id = $2', [newStatus, req.params.id]);
    const sub = (await pool.query('SELECT candidate_id, candidate_name, job_id, job_title, client_name FROM submissions WHERE submission_id = $1', [req.params.id])).rows[0];
    if (decision === 'APPROVED' && sub) {
      // Approved -> now the candidate can be invited (email + resume upload)
      return res.json({ ok: true, status: 'SUBMITTED', candidate: sub, canInvite: true });
    }
    return res.json({ ok: true, status: newStatus });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- Submissions PENDING approval (for the hiring manager) ----------
app.get('/submissions/pending-approval', async (req, res) => {
  try {
    const { rows } = await pool.query(
      `SELECT submission_id, candidate_id, candidate_name, job_id, job_title, client_name, current_ctc, expected_ctc, submitted_by, created_at
         FROM submissions WHERE status = 'PENDING_HM_APPROVAL' ORDER BY created_at DESC`);
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
    const b = req.body;
    if (!b.full_name || !b.full_name.trim() || !b.role) return res.status(400).json({ error: 'full_name and role required' });
    const user_id = b.user_id || ('USR-' + Date.now().toString().slice(-6));
    await pool.query(
      `INSERT INTO app_users (user_id, full_name, email, role, user_group, phone) VALUES ($1,$2,$3,$4,$5,$6)
       ON CONFLICT (user_id) DO NOTHING`,
      [user_id, b.full_name, b.email || '', b.role, b.user_group || '', b.phone || '']
    );
    return res.json({ ok: true, user_id });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// Recruiters list (for the assign dropdown)
app.get('/recruiters', async (req, res) => {
  try { const { rows } = await pool.query("SELECT user_id, full_name FROM app_users WHERE role='recruiter' AND active=TRUE ORDER BY full_name"); return res.json({ recruiters: rows }); }
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
  const prompt = `Write ONE efficient BigQuery SELECT (only SQL, no fences). ${roleHint}${ctx ? ' Recent context: ' + ctx : ''}
Rules: select only needed columns (never SELECT *). For a list, add LIMIT ${listLimit} at the end. For a count/total use COUNT(*) with no LIMIT. Use COUNT/SUM/AVG for totals, GROUP BY for 'per/by/each/breakdown'. Use ORDER BY DESC + LIMIT for 'top/most/highest'. Text filters use LOWER(col) LIKE LOWER('%v%'). Map pipeline stages to visa_status IN(...). Use candidates.experience_years for experience. Prefix tables with \`${BQ_DS}.\`.
${BQ_SCHEMA}
Question: ${question}
SQL:`;
  const gen = await genModel.generateContent(prompt);
  let sql = gen.response.candidates[0].content.parts[0].text.replace(/^```(?:sql)?|```$/gim, '').trim();
  if (!_readonlySQL(sql)) return { text: 'That request is blocked (read-only guard).', rows: [] };
  try {
    const rows = await bqQuery(sql);
    if (!rows.length) return { text: 'No records match this request in the database. The correct answer is that none were found — I will not invent any.', rows: [] };
    const shown = rows.slice(0, wantAll ? 25 : 8);
    const note = rows.length > shown.length ? `\n\n(Showing ${shown.length} of ${rows.length}.)` : '';
    return { text: `Result (${rows.length} found):\n${JSON.stringify(shown, null, 1)}${note}`, rows };
  } catch (e) { return { text: 'Query failed: ' + e.message, rows: [] }; }
}


// Cloud SQL data agent — for jobs, submissions, approvals, placements (the live website data)
async function agentSqlData(question, role) {
  const schema = `Cloud SQL (PostgreSQL) tables:
jobs(job_id, title, client, hiring_manager, recruiter, assigned_recruiter_id, country, job_location, number_of_positions, priority, status, created_date)
  -- open jobs = status IN ('Open','Active','POSTED','In Process'); closed = status IN ('Closed','CLOSED','Filled')
submissions(submission_id, candidate_id, candidate_name, job_id, job_title, client_name, status, submitted_by, current_ctc, expected_ctc, resume_received, created_at)
  -- statuses: PENDING_HM_APPROVAL, SUBMITTED, RECRUITER_CALL, HR_INTERVIEW, CLIENT_INTERVIEW, OFFER, PLACED, REJECTED
app_users(user_id, full_name, email, role, user_group)  -- roles: admin, hiring_manager, recruiter
clients(client_id, client_name, country, industry)`;
  const prompt = `Write ONE PostgreSQL SELECT (only SQL, no fences, no trailing semicolon). Use COUNT(*) for counts. Use ILIKE for text. Never SELECT * ; select only needed columns and LIMIT 50 for lists.
${schema}
Question: ${question}
SQL:`;
  const gen = await genModel.generateContent(prompt);
  let sql = gen.response.candidates[0].content.parts[0].text.replace(/^```(?:sql)?|```$/gim, '').trim().replace(/;+$/,'');
  if (!/^select/i.test(sql) || /\b(insert|update|delete|drop|alter|create)\b/i.test(sql)) return { text: 'Blocked (read-only).', rows: [] };
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
    if (!rows.length) return 'No high-urgency candidates right now.';
    return 'Top urgent / at-risk candidates:\n' + JSON.stringify(rows, null, 1);
  } catch (e) { return 'Urgency query failed: ' + e.message; }
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
    const { question, role, history } = req.body;
    if (!question) return res.status(400).json({ error: 'question required' });
    const q = question.toLowerCase();

    // --- Fast intent detection (reliable keyword routing) ---
    const idMatch = question.match(/\bC\d{4}\b/i);
    const isVisaFix = /(fix|remediat|rejection|what.?s wrong|how to fix|resolve).*(visa)|visa.*(fix|reject)/i.test(q) && idMatch;
    const isUrgency = /(urgent|urgency|at.?risk|awol|not reported|priority candidates|who needs attention)/i.test(q);

    let toolResult = '', agentName = '', rows = [];
    // Route: jobs/submissions/approvals/placements -> Cloud SQL (live website data). Candidates/visa -> BigQuery.
    const isJobsData = /(open job|jobs\b|job posting|requisition|submission|submitted|approval|pending|assigned to me|my job|placement|placed|offer|interview scheduled|client interview|hr interview)/i.test(q);
    if (isVisaFix) {
      toolResult = await agentVisaFix(idMatch[0]); agentName = 'Visa Fix-It';
    } else if (isUrgency) {
      toolResult = await agentUrgency(10); agentName = 'Urgency Watch';
    } else if (isJobsData) {
      const r = await agentSqlData(question, role); toolResult = r.text; rows = r.rows; agentName = 'Jobs & Submissions';
    } else {
      const r = await agentQueryData(question, role, history);
      toolResult = r.text; rows = r.rows; agentName = 'Candidate Data';
    }

    // Compose a clean answer that faithfully reflects the tool result (no invention).
    const scope = role === 'admin' ? 'admin (full visibility)' : role === 'hiring_manager' ? 'hiring manager' : role === 'recruiter' ? 'recruiter' : 'user';
    const finalPrompt = `You are the Recruit 360 AI assistant answering a ${scope}. Below is the exact result from the database for the user's question. Answer the user clearly and directly using ONLY this result — never add or invent names, numbers or candidates that are not in it. If the result says none were found, say clearly that there are none. Keep it concise and professional. If the result is a list, present it readably.

DATABASE RESULT:
${toolResult}

USER QUESTION: ${question}

ANSWER:`;
    const finalGen = await genModel.generateContent(finalPrompt);
    const answer = finalGen.response.candidates[0].content.parts[0].text.trim();
    return res.json({ answer, agent: agentName });
  } catch (e) { return res.status(500).json({ error: 'Assistant error', detail: e.message }); }
});

// ---------- CONTEXTUAL RECRUITER AGENT: suggest & execute the next action per candidate ----------
// Given a submission's current status, it knows exactly what the recruiter should do next.
app.get('/submissions/:id/next-action', async (req, res) => {
  try {
    const { rows } = await pool.query('SELECT submission_id, candidate_id, candidate_name, job_title, client_name, status, interview_link, resume_received FROM submissions WHERE submission_id = $1', [req.params.id]);
    if (!rows.length) return res.status(404).json({ error: 'not found' });
    const s = rows[0];
    const MAP = {
      PENDING_HM_APPROVAL: { action: 'awaiting_approval', label: 'Awaiting HM approval', can: false, hint: 'The hiring manager needs to approve this candidate.' },
      SUBMITTED:          { action: 'request_resume', label: 'Send resume request', can: true, hint: 'Approved — send the candidate the resume-upload email.' },
      RECRUITER_CALL:     { action: 'schedule_call', label: 'Schedule recruiter call', can: true, hint: 'Set up the 5-min recruiter call and email the link.' },
      HR_INTERVIEW:       { action: 'schedule_hr', label: 'Schedule HR interview', can: true, hint: 'Book the HR round and notify the candidate.' },
      CLIENT_INTERVIEW:   { action: 'schedule_client', label: 'Schedule client interview', can: true, hint: 'Book the client interview and share the summary.' },
      OFFER:              { action: 'send_offer', label: 'Send offer', can: true, hint: 'Release the offer to the candidate.' },
      PLACED:             { action: 'done', label: 'Placed', can: false, hint: 'This candidate is placed.' },
      REJECTED:           { action: 'done', label: 'Rejected', can: false, hint: 'This candidate was rejected.' },
    };
    const next = MAP[s.status] || { action: 'review', label: 'Review', can: true, hint: 'Review this candidate.' };
    return res.json({ submission: s, next });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// Contextual agent EXECUTE — performs the suggested action (and returns any email content)
app.post('/submissions/:id/context-action', async (req, res) => {
  try {
    const { action } = req.body;
    const { rows } = await pool.query('SELECT candidate_id, candidate_name, job_id, job_title, client_name, status FROM submissions WHERE submission_id = $1', [req.params.id]);
    if (!rows.length) return res.status(404).json({ error: 'not found' });
    const s = rows[0];
    const candEmail = (s.candidate_id || '').toLowerCase() + '@example.com';

    if (action === 'request_resume') {
      // create onboarding + invite email
      const token = 'INV-' + Math.random().toString(36).slice(2, 10).toUpperCase();
      try { await pool.query(`INSERT INTO candidate_onboarding (candidate_id, job_id, candidate_name, email, invite_token, onboarding_status) VALUES ($1,$2,$3,$4,$5,'INVITED')`, [s.candidate_id, s.job_id || '', s.candidate_name, candEmail, token]); } catch(e){}
      const link = (process.env.PORTAL_URL || 'https://direct-tribute-502305-q5.web.app') + '/#/candidate-upload?token=' + token;
      const emailSubject = 'Please upload your resume — ' + (s.job_title || 'a role');
      const emailBody = 'Dear ' + s.candidate_name + ',\n\nYou have been shortlisted for ' + (s.job_title || 'a role') + (s.client_name ? ' at ' + s.client_name : '') + '. Please upload your latest resume and confirm your contact details here:\n\n' + link + '\n\nBest regards,\nRecruit 360 Team';
      return res.json({ ok: true, done: 'request_resume', emailSubject, emailBody, candidateEmail: candEmail, link });
    }
    if (action === 'send_summary') {
      const gen = await genModel.generateContent('Write a short professional candidate summary email to a client for ' + s.candidate_name + ', role ' + (s.job_title || '') + '. 4-5 lines, highlight fit. Return only the email body.');
      const emailBody = gen.response.candidates[0].content.parts[0].text.trim();
      return res.json({ ok: true, done: 'send_summary', emailSubject: 'Candidate summary — ' + s.candidate_name, emailBody, candidateEmail: (s.client_name||'client').toLowerCase().replace(/\s+/g,'') + '@client.com' });
    }
    // For schedule actions, just advance the stage (the UI opens the schedule step)
    return res.json({ ok: true, done: action });
  } catch (e) { return res.status(500).json({ error: e.message }); }
});

// ---------- BULK APPROVALS (hiring manager) ----------
app.post('/submissions/bulk-approval', async (req, res) => {
  try {
    const { submission_ids, decision, approver } = req.body; // decision: APPROVED | REJECTED
    if (!Array.isArray(submission_ids) || !submission_ids.length) return res.status(400).json({ error: 'submission_ids required' });
    const newStatus = decision === 'APPROVED' ? 'SUBMITTED' : 'REJECTED';
    const invites = [];
    for (const id of submission_ids) {
      await pool.query('UPDATE submissions SET status = $1, last_updated = NOW() WHERE submission_id = $2', [newStatus, id]);
      if (decision === 'APPROVED') {
        const sub = (await pool.query('SELECT candidate_id, candidate_name, job_id, job_title, client_name FROM submissions WHERE submission_id = $1', [id])).rows[0];
        if (sub) {
          const token = 'INV-' + Math.random().toString(36).slice(2, 10).toUpperCase();
          const candEmail = (sub.candidate_id || '').toLowerCase() + '@example.com';
          try { await pool.query(`INSERT INTO candidate_onboarding (candidate_id, job_id, candidate_name, email, invite_token, onboarding_status) VALUES ($1,$2,$3,$4,$5,'INVITED')`, [sub.candidate_id, sub.job_id || '', sub.candidate_name, candEmail, token]); } catch(e){}
          const link = (process.env.PORTAL_URL || 'https://direct-tribute-502305-q5.web.app') + '/#/candidate-upload?token=' + token;
          invites.push({ name: sub.candidate_name, email: candEmail, subject: 'Please upload your resume — ' + (sub.job_title || 'a role'), body: 'Dear ' + sub.candidate_name + ',\n\nYou have been shortlisted for ' + (sub.job_title || 'a role') + '. Please upload your resume here:\n' + link + '\n\nBest regards,\nRecruit 360 Team' });
        }
      }
    }
    return res.json({ ok: true, count: submission_ids.length, decision, invites });
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

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log('API on ' + PORT));
