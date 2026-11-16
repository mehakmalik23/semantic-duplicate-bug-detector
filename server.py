import os, re, time, faiss, pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Semantic Duplicate Bug Detection API")

# Allow ANY external website to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Loading NLP Models & FAISS Index into FastAPI Backend...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-4-v2', max_length=128)
corpus_df = pd.read_pickle("corpus_app_df.pkl")
corpus_emb = np.ascontiguousarray(np.load("corpus_embeddings.npy"), dtype=np.float32)

if os.path.exists("bug_corpus_faiss.index"):
    faiss_index = faiss.read_index("bug_corpus_faiss.index")
else:
    faiss_index = faiss.IndexFlatIP(corpus_emb.shape[1])
    faiss_index.add(corpus_emb)

with open("tfidf_vectorizer.pkl", "rb") as f: tfidf_vec = pickle.load(f)
with open("corpus_tfidf.pkl", "rb") as f: corpus_tfidf = pickle.load(f)
meta_clf = pickle.load(open("meta_classifier.pkl", "rb")) if os.path.exists("meta_classifier.pkl") else None

error_pattern = re.compile(r'\b[A-Za-z0-9_]+(?:Exception|Error|Failure)\b')
STOPWORDS = {"the", "and", "for", "with", "when", "that", "this", "from", "into", "should", "would", "could", "have", "does", "not", "are", "was", "were", "while", "using", "instead", "show", "use"}

def highlight_terms(query_text, cand_text):
    q_tokens = set(re.findall(r'[A-Za-z0-9_#]{3,}', query_text.lower())) - STOPWORDS
    def repl(m):
        w = m.group(0)
        if error_pattern.match(w):
            return f'<span class="bg-red-900/80 text-red-200 px-1.5 py-0.5 rounded border border-red-500 font-semibold">{w}</span>'
        if w.lower() in q_tokens:
            return f'<span class="bg-emerald-900/80 text-emerald-200 px-1.5 py-0.5 rounded border border-emerald-500 font-semibold">{w}</span>'
        return w
    return re.sub(r'[A-Za-z0-9_#]{3,}', repl, cand_text)

class BugInput(BaseModel):
    title: str
    description: str
    component: str = "Debug"
    version: str = "2.0"
    top_k: int = 5

@app.post("/api/detect-duplicates")
def detect_duplicates(req: BugInput):
    t0 = time.perf_counter()
    matches = set(error_pattern.findall(f"{req.title} {req.description}"))
    detected_err = ", ".join(sorted(matches)) if matches else "None"

    # Stage 1: FAISS + TF-IDF
    q_emb = np.ascontiguousarray(
        sbert_model.encode([f"{req.title} . {req.title} . {req.description[:1000]}"], normalize_embeddings=True),
        dtype=np.float32
    )
    t_f0 = time.perf_counter()
    f_sims, f_idxs = faiss_index.search(q_emb, len(corpus_df))
    faiss_ms = round((time.perf_counter() - t_f0) * 1000, 2)

    sbert_sims = np.zeros(len(corpus_df), dtype=np.float32)
    sbert_sims[f_idxs[0]] = f_sims[0]
    tfidf_sims = cosine_similarity(tfidf_vec.transform([f"{req.title} . {req.description[:1500]}"]), corpus_tfidf)[0]

    shortlist_idx = np.argsort(-(0.65 * sbert_sims + 0.35 * tfidf_sims))[:15]

    # Stage 2: Cross-Encoder
    q_short = f"{req.title} . {req.description[:250]}"
    ce_pairs = [[q_short, f"{corpus_df.iloc[i]['Clean_Title']} . {corpus_df.iloc[i]['Clean_Description'][:250]}"] for i in shortlist_idx]
    ce_raw = cross_encoder.predict(ce_pairs)
    ce_norm = (ce_raw - np.min(ce_raw)) / (np.max(ce_raw) - np.min(ce_raw) + 1e-8)

    # Stage 3: Meta-Classifier + Metadata Fusion
    results = []
    q_full = f"{req.title} {req.description}"
    for pos, c_idx in enumerate(shortlist_idx):
        row = corpus_df.iloc[c_idx]
        comp_m = 1.0 if row['Component'] == req.component else 0.0
        ver_m = 1.0 if row['Version'] == req.version else 0.0
        err_m = 1.0 if (detected_err != "None" and row['Extracted_Errors'] == detected_err) else 0.0

        if meta_clf is not None:
            feat = np.array([[sbert_sims[c_idx], tfidf_sims[c_idx], ce_norm[pos], comp_m, ver_m, err_m]])
            score = float(meta_clf.predict_proba(feat)[0, 1])
        else:
            score = float(0.5 * sbert_sims[c_idx] + 0.3 * tfidf_sims[c_idx] + 0.2 * ce_norm[pos] + 0.18 * comp_m + 0.07 * ver_m + 0.2 * err_m)

        results.append({
            "issue_id": int(row['Issue_id']),
            "title": row['Clean_Title'],
            "highlighted_title": highlight_terms(q_full, row['Clean_Title']),
            "highlighted_desc": highlight_terms(q_full, row['Clean_Description'][:360] + "..."),
            "component": row['Component'],
            "version": row['Version'],
            "errors": row['Extracted_Errors'],
            "sbert_sim": round(float(sbert_sims[c_idx]) * 100, 1),
            "ce_score": round(float(ce_norm[pos]) * 100, 1),
            "probability": round(min(score * 100, 99.9), 1)
        })

    results = sorted(results, key=lambda x: x['probability'], reverse=True)[:req.top_k]
    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "extracted_errors": detected_err,
        "faiss_latency_ms": faiss_ms,
        "total_latency_ms": total_ms,
        "duplicates": results
    }

@app.get("/", response_class=HTMLResponse)
def serve_website():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>BugTrack AI — Intelligent Issue Portal</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans">
  <nav class="border-b border-slate-800 bg-slate-900/80 px-8 py-4 flex justify-between items-center">
    <div class="flex items-center gap-3">
      <span class="text-2xl">🐞</span>
      <div>
        <h1 class="font-bold text-lg text-white">BugTrack Enterprise Portal</h1>
        <p class="text-xs text-slate-400">Powered by Sentence-BERT + FAISS + Cross-Encoder REST API</p>
      </div>
    </div>
    <div class="flex gap-2">
      <button onclick="loadPreset(1)" class="text-xs bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded border border-slate-700">Preset 1: #toString Debug</button>
      <button onclick="loadPreset(2)" class="text-xs bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded border border-slate-700">Preset 2: NullPointerException</button>
    </div>
  </nav>

  <main class="max-w-7xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
    <!-- Left Column: Website Issue Submission Form -->
    <div class="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl h-fit">
      <h2 class="text-base font-semibold mb-4 text-indigo-400">📝 Submit a New Software Bug Report</h2>
      <div class="space-y-4">
        <div>
          <label class="text-xs text-slate-400 block mb-1">Issue Summary / Title</label>
          <input id="title" type="text" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:border-indigo-500 outline-none" value="Display variable values using toString() in debugger inspector">
        </div>
        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="text-xs text-slate-400 block mb-1">Component</label>
            <select id="component" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm">
              <option value="Debug">Debug</option>
              <option value="Core">Core</option>
              <option value="UI">UI</option>
              <option value="Text">Text</option>
            </select>
          </div>
          <div>
            <label class="text-xs text-slate-400 block mb-1">Version</label>
            <select id="version" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm">
              <option value="2.0">2.0</option>
              <option value="2.1">2.1</option>
              <option value="3.0">3.0</option>
            </select>
          </div>
        </div>
        <div>
          <label class="text-xs text-slate-400 block mb-1">Detailed Description & Stack Trace</label>
          <textarea id="description" rows="5" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:border-indigo-500 outline-none">When inspecting objects during debugging, the variables view should evaluate #toString to show readable details instead of raw type references.</textarea>
        </div>
        <button onclick="checkDuplicates()" id="btn" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-2.5 rounded-lg shadow transition">
          🔍 Check for Duplicate Bugs via AI API
        </button>
      </div>
    </div>

    <!-- Right Column: Live AI Duplicate Detection Feed -->
    <div class="lg:col-span-7 space-y-4">
      <div id="telemetry" class="hidden grid grid-cols-3 gap-3">
        <div class="bg-slate-900 border border-slate-800 p-3 rounded-lg">
          <div class="text-xs text-slate-400">⚡ FAISS Vector Search</div>
          <div id="faiss_ms" class="text-lg font-bold text-emerald-400">-</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 p-3 rounded-lg">
          <div class="text-xs text-slate-400">⏱️ Total API Response</div>
          <div id="total_ms" class="text-lg font-bold text-indigo-400">-</div>
        </div>
        <div class="bg-slate-900 border border-slate-800 p-3 rounded-lg">
          <div class="text-xs text-slate-400">🚨 Extracted Exception</div>
          <div id="err_sig" class="text-sm font-mono text-rose-400 truncate">-</div>
        </div>
      </div>

      <div id="banner" class="hidden p-4 rounded-xl border font-medium text-sm"></div>
      <div id="results" class="space-y-3">
        <div class="text-center text-slate-500 py-16 border border-dashed border-slate-800 rounded-xl">
          Click <b>"Check for Duplicate Bugs via AI API"</b> to query the FastAPI Transformer backend.
        </div>
      </div>
    </div>
  </main>

  <script>
    function loadPreset(n) {
      if (n === 1) {
        document.getElementById('title').value = "Display variable values using toString() in debugger inspector";
        document.getElementById('description').value = "When inspecting objects during debugging, the variables view should evaluate #toString to show readable details instead of raw type references.";
        document.getElementById('component').value = "Debug";
      } else {
        document.getElementById('title').value = "Crash with NullPointerException when renaming method in class";
        document.getElementById('description').value = "While performing a method refactoring in the editor, an unexpected NullPointerException is thrown in JavaModelManager and the operation aborts.";
        document.getElementById('component').value = "Core";
      }
      checkDuplicates();
    }

    async function checkDuplicates() {
      const btn = document.getElementById('btn');
      btn.innerText = "⏳ Querying FastAPI Transformer Pipeline...";
      const res = await fetch('/api/detect-duplicates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: document.getElementById('title').value,
          description: document.getElementById('description').value,
          component: document.getElementById('component').value,
          version: document.getElementById('version').value,
          top_k: 5
        })
      });
      const data = await res.json();
      btn.innerText = "🔍 Check for Duplicate Bugs via AI API";

      document.getElementById('telemetry').classList.remove('hidden');
      document.getElementById('faiss_ms').innerText = data.faiss_latency_ms + " ms";
      document.getElementById('total_ms').innerText = data.total_latency_ms + " ms";
      document.getElementById('err_sig').innerText = data.extracted_errors;

      const top = data.duplicates[0];
      const banner = document.getElementById('banner');
      banner.classList.remove('hidden');
      if (top.probability >= 72) {
        banner.className = "p-4 rounded-xl border bg-rose-950/60 border-rose-700 text-rose-200 text-sm";
        banner.innerHTML = `🔴 <b>HIGH PROBABILITY DUPLICATE (${top.probability}%):</b> Strongly matches Master Bug <b>#${top.issue_id}</b>. Recommend linking instead of opening a new ticket.`;
      } else {
        banner.className = "p-4 rounded-xl border bg-amber-950/60 border-amber-700 text-amber-200 text-sm";
        banner.innerHTML = `🟡 <b>POTENTIAL DUPLICATE (${top.probability}%):</b> Similar to Bug <b>#${top.issue_id}</b>. Please review below.`;
      }

      const container = document.getElementById('results');
      container.innerHTML = data.duplicates.map((d, i) => `
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 hover:border-slate-700 transition">
          <div class="flex justify-between items-start mb-2">
            <div>
              <span class="text-xs font-bold bg-indigo-950 text-indigo-300 border border-indigo-700 px-2 py-0.5 rounded mr-2">Rank #${i+1} • Bug #${d.issue_id}</span>
              <span class="text-xs bg-slate-800 text-slate-300 px-2 py-0.5 rounded">${d.component} v${d.version}</span>
            </div>
            <span class="text-sm font-bold ${d.probability >= 72 ? 'text-rose-400' : 'text-emerald-400'}">${d.probability}% Match</span>
          </div>
          <h3 class="font-semibold text-sm mb-1">${d.highlighted_title}</h3>
          <p class="text-xs text-slate-400 leading-relaxed mb-3">${d.highlighted_desc}</p>
          <div class="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
            <div class="bg-indigo-500 h-full" style="width: ${d.probability}%"></div>
          </div>
        </div>
      `).join('');
    }
  </script>
</body>
</html>
    """
