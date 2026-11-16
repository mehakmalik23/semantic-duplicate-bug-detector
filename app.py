import os
import re
import time
import faiss
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Semantic Duplicate Bug Detector (Pro)",
    page_icon="🐞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS FOR EXPLAINABLE HIGHLIGHTING & UI CARDS ---
st.markdown("""
<style>
.hl-token {
    background-color: #1e4620;
    color: #a3cfbb;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
    border: 1px solid #2e8b57;
}
.hl-err {
    background-color: #5c1e1e;
    color: #f5c2c7;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 700;
    border: 1px solid #dc3545;
}
.box-card {
    background-color: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 14px;
    border-radius: 8px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODELS & ARTIFACTS (CACHED FOR INSTANT PERFORMANCE) ---
@st.cache_resource(show_spinner="Loading Transformer Models, FAISS Vector Index & Learned Meta-Classifier...")
def load_resources():
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    cross_enc = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-4-v2', max_length=128)
    
    corpus_df = pd.read_pickle("corpus_app_df.pkl")
    corpus_emb = np.ascontiguousarray(np.load("corpus_embeddings.npy"), dtype=np.float32)
    
    # Load or build FAISS Inner Product Index
    if os.path.exists("bug_corpus_faiss.index"):
        f_index = faiss.read_index("bug_corpus_faiss.index")
    else:
        f_index = faiss.IndexFlatIP(corpus_emb.shape[1])
        f_index.add(corpus_emb)
        
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vec = pickle.load(f)
    with open("corpus_tfidf.pkl", "rb") as f:
        corpus_tfidf = pickle.load(f)
        
    meta_clf = None
    if os.path.exists("meta_classifier.pkl"):
        with open("meta_classifier.pkl", "rb") as f:
            meta_clf = pickle.load(f)
            
    ablation_df = pd.read_pickle("ablation_df.pkl") if os.path.exists("ablation_df.pkl") else None
    return sbert, cross_enc, corpus_df, corpus_emb, f_index, tfidf_vec, corpus_tfidf, meta_clf, ablation_df

sbert_model, cross_encoder, corpus_df, corpus_embeddings, faiss_index, tfidf_vec, corpus_tfidf, meta_clf, ablation_df = load_resources()

# --- 4. HELPER FUNCTIONS (ERROR EXTRACTION & EXPLAINABLE TOKEN HIGHLIGHTING) ---
error_pattern = re.compile(r'\b[A-Za-z0-9_]+(?:Exception|Error|Failure)\b')

def extract_errors(text):
    matches = set(error_pattern.findall(str(text)))
    return ", ".join(sorted(matches)) if matches else "None"

def min_max_norm(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

STOPWORDS = {
    "the", "and", "for", "with", "when", "that", "this", "from", "into",
    "should", "would", "could", "have", "does", "not", "are", "was",
    "were", "while", "using", "instead", "show", "use"
}

def highlight_shared_terms(query_text, cand_text):
    q_tokens = set(re.findall(r'[A-Za-z0-9_#]{3,}', query_text.lower())) - STOPWORDS
    def replacer(match):
        word = match.group(0)
        if error_pattern.match(word):
            return f'<span class="hl-err">{word}</span>'
        if word.lower() in q_tokens:
            return f'<span class="hl-token">{word}</span>'
        return word
    return re.sub(r'[A-Za-z0-9_#]{3,}', replacer, cand_text)

# --- 5. APPLICATION HEADER ---
st.title("🐞 Semantic Duplicate Bug Report Detection System")
st.caption(
    "Production Architecture: **FAISS Vector Search (SBERT)** ➔ **Cross-Encoder Reranking** ➔ "
    "**Supervised Meta-Classifier (LTR) & Technical Metadata Fusion**"
)

# --- 6. SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Engine Configuration")
scoring_mode = st.sidebar.radio(
    "🧠 Fusion Scoring Mode:",
    ["Learned Meta-Classifier (LTR)", "Heuristic Weighted Fusion"],
    index=0,
    help="Learned Meta-Classifier uses 5-Fold trained Logistic Regression weights; Heuristic uses customizable sliders."
)
top_k = st.sidebar.slider("Top-K Candidates to Display", min_value=3, max_value=10, value=5)
use_metadata = st.sidebar.toggle("🚀 Enable Technical Metadata Signals", value=True)

with st.sidebar.expander("⚖️ Manual Heuristic Sliders", expanded=(scoring_mode == "Heuristic Weighted Fusion")):
    w_sbert = st.slider("SBERT Weight", 0.0, 1.0, 0.50, 0.05)
    w_tfidf = st.slider("TF-IDF Weight", 0.0, 1.0, 0.30, 0.05)
    w_ce = st.slider("Cross-Encoder Weight", 0.0, 1.0, 0.20, 0.05)
    b_comp = st.slider("Component Boost", 0.0, 0.40, 0.18, 0.02)
    b_err = st.slider("Error Signature Boost", 0.0, 0.40, 0.20, 0.02)
    b_ver = st.slider("Version Boost", 0.0, 0.20, 0.07, 0.01)

st.sidebar.divider()
preset = st.sidebar.selectbox(
    "🧪 Quick Test Presets:",
    [
        "Preset 1: Debugger Variable Inspection (#toString)",
        "Preset 2: NullPointerException in Refactoring / AST",
        "Preset 3: Scrapbook Snippet Import Evaluation",
        "Custom Bug Report"
    ]
)

if preset == "Preset 1: Debugger Variable Inspection (#toString)":
    default_title = "Display variable values using toString() in debugger inspector"
    default_desc = "When inspecting objects during debugging, the variables view should evaluate #toString to show readable details instead of raw type references."
    default_comp, default_ver = "Debug", "2.0"
elif preset == "Preset 2: NullPointerException in Refactoring / AST":
    default_title = "Crash with NullPointerException when renaming method in class"
    default_desc = "While performing a method refactoring in the editor, an unexpected NullPointerException is thrown in JavaModelManager and the operation aborts."
    default_comp, default_ver = "Core", "2.0"
elif preset == "Preset 3: Scrapbook Snippet Import Evaluation":
    default_title = "Cannot use import declarations inside scrapbook snippet evaluation"
    default_desc = "Evaluating code snippets fails when trying to include package imports at the top of the snippet file."
    default_comp, default_ver = "Debug", "2.0"
else:
    default_title, default_desc, default_comp, default_ver = "", "", "Core", "2.0"

# --- 7. MULTI-TAB DASHBOARD ---
tab1, tab2, tab3 = st.tabs([
    "🔍 Live Duplicate Bug Detector (FAISS + LTR)",
    "📊 Benchmark & Ablation Study Analytics",
    "🗄️ Historical Bug Corpus Explorer"
])

# === TAB 1: LIVE DUPLICATE BUG DETECTOR ===
with tab1:
    col1, col2 = st.columns([2.2, 1])
    with col1:
        bug_title = st.text_input("📝 Incoming Bug Title / Summary", value=default_title)
        bug_desc = st.text_area("📄 Detailed Description / Stack Trace", value=default_desc, height=130)
    with col2:
        components_list = sorted(corpus_df['Component'].unique().tolist())
        versions_list = sorted(corpus_df['Version'].unique().tolist())
        c_idx_def = components_list.index(default_comp) if default_comp in components_list else 0
        v_idx_def = versions_list.index(default_ver) if default_ver in versions_list else 0

        selected_comp = st.selectbox("🧩 Software Component", options=components_list, index=c_idx_def)
        selected_ver = st.selectbox("🏷️ Software Version", options=versions_list, index=v_idx_def)
        detected_err = extract_errors(f"{bug_title} {bug_desc}")
        st.info(f"**Auto-Extracted Error Signatures:**\n`{detected_err}`")

    if st.button("🚀 Analyze & Retrieve Potential Duplicates", type="primary", use_container_width=True):
        if not bug_title.strip() and not bug_desc.strip():
            st.warning("Please enter a bug title or description.")
        else:
            t_start = time.perf_counter()

            # Stage 1: FAISS Vector Search + TF-IDF Lexical Search
            sbert_input = f"{bug_title} . {bug_title} . {bug_desc[:1000]}"
            tfidf_input = f"{bug_title} . {bug_desc[:1500]}"

            q_emb = np.ascontiguousarray(
                sbert_model.encode([sbert_input], convert_to_numpy=True, normalize_embeddings=True),
                dtype=np.float32
            )

            t_faiss_0 = time.perf_counter()
            faiss_sims_all, faiss_idxs_all = faiss_index.search(q_emb, len(corpus_df))
            faiss_search_ms = (time.perf_counter() - t_faiss_0) * 1000

            sbert_sims = np.zeros(len(corpus_df), dtype=np.float32)
            sbert_sims[faiss_idxs_all[0]] = faiss_sims_all[0]

            q_tfidf = tfidf_vec.transform([tfidf_input])
            tfidf_sims = cosine_similarity(q_tfidf, corpus_tfidf)[0]

            hybrid_stage1 = 0.65 * sbert_sims + 0.35 * tfidf_sims
            shortlist_idx = np.argsort(-hybrid_stage1)[:15]

            # Stage 2: Cross-Encoder Reranking on Top-15 Candidates
            t_ce_0 = time.perf_counter()
            q_short = f"{bug_title} . {bug_desc[:250]}"
            ce_pairs = [
                [q_short, f"{corpus_df.iloc[idx]['Clean_Title']} . {corpus_df.iloc[idx]['Clean_Description'][:250]}"]
                for idx in shortlist_idx
            ]
            ce_raw = cross_encoder.predict(ce_pairs)
            ce_norm = min_max_norm(ce_raw)
            ce_rerank_ms = (time.perf_counter() - t_ce_0) * 1000

            # Stage 3: Learned Meta-Classifier / Metadata Fusion
            rows_out = []
            for rank_pos, c_idx in enumerate(shortlist_idx):
                row = corpus_df.iloc[c_idx]
                comp_match = 1.0 if (use_metadata and row['Component'] == selected_comp) else 0.0
                ver_match = 1.0 if (use_metadata and row['Version'] == selected_ver) else 0.0
                err_match = 1.0 if (use_metadata and detected_err != "None" and row['Extracted_Errors'] == detected_err) else 0.0

                meta_boost = (b_comp * comp_match + b_ver * ver_match + b_err * err_match)
                heur_score = (w_sbert * sbert_sims[c_idx]) + (w_tfidf * tfidf_sims[c_idx]) + (w_ce * ce_norm[rank_pos]) + meta_boost

                if scoring_mode == "Learned Meta-Classifier (LTR)" and meta_clf is not None:
                    feat_vec = np.array([[sbert_sims[c_idx], tfidf_sims[c_idx], ce_norm[rank_pos], comp_match, ver_match, err_match]])
                    final_score = float(meta_clf.predict_proba(feat_vec)[0, 1])
                else:
                    final_score = float(heur_score)

                rows_out.append({
                    "Issue_id": row['Issue_id'],
                    "Title": row['Clean_Title'],
                    "Description": row['Clean_Description'],
                    "Component": row['Component'],
                    "Version": row['Version'],
                    "Extracted_Errors": row['Extracted_Errors'],
                    "SBERT_Score": float(sbert_sims[c_idx]),
                    "TFIDF_Score": float(tfidf_sims[c_idx]),
                    "CE_Score": float(ce_norm[rank_pos]),
                    "Meta_Boost": float(meta_boost),
                    "Final_Score": final_score
                })

            res_df = pd.DataFrame(rows_out).sort_values("Final_Score", ascending=False).head(top_k).reset_index(drop=True)
            res_df["Rank"] = res_df.index + 1
            total_ms = (time.perf_counter() - t_start) * 1000

            # --- Latency Telemetry Metrics Bar ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🗄️ FAISS Indexed Bugs", f"{faiss_index.ntotal:,}")
            m2.metric("⚡ FAISS Stage-1 Search", f"{faiss_search_ms:.2f} ms")
            m3.metric("🎯 Cross-Encoder (Top-15)", f"{ce_rerank_ms:.1f} ms")
            m4.metric("⏱️ Total Pipeline Latency", f"{total_ms:.1f} ms")

            # --- Automated Triage Decision Banner ---
            top_match = res_df.iloc[0]
            if top_match["Final_Score"] >= 0.72:
                st.error(f"🔴 **HIGH PROBABILITY DUPLICATE DETECTED!** Matches Master Bug **#{top_match['Issue_id']}** (Score / Probability: `{top_match['Final_Score']:.3f}`).")
            elif top_match["Final_Score"] >= 0.50:
                st.warning(f"🟡 **POTENTIAL DUPLICATE FOUND:** Moderate similarity with Bug **#{top_match['Issue_id']}** (Score: `{top_match['Final_Score']:.3f}`).")
            else:
                st.success("🟢 **LIKELY NEW UNIQUE BUG:** Safe to open a new Master Bug ticket.")

            # --- Explainable AI Side-by-Side Comparison ---
            st.subheader("🔬 Explainable AI Verification: Shared Technical Terms Highlighted in Green")
            c_left, c_right = st.columns(2)
            q_full = f"{bug_title} {bug_desc}"
            with c_left:
                st.markdown("#### 📥 Incoming Bug Report (Query)")
                st.markdown(f"**Component:** `{selected_comp}` | **Version:** `{selected_ver}` | **Errors:** `{detected_err}`")
                st.markdown(f'<div class="box-card"><b>{bug_title}</b><br><br>{bug_desc}</div>', unsafe_allow_html=True)
            with c_right:
                st.markdown(f"#### 🏆 Top Historical Match: Bug #{top_match['Issue_id']}")
                st.markdown(f"**Component:** `{top_match['Component']}` | **Version:** `{top_match['Version']}` | **Errors:** `{top_match['Extracted_Errors']}`")
                hl_title = highlight_shared_terms(q_full, top_match['Title'])
                hl_desc = highlight_shared_terms(q_full, top_match['Description'][:450])
                st.markdown(f'<div class="box-card"><b>{hl_title}</b><br><br>{hl_desc}</div>', unsafe_allow_html=True)

            # --- Ranked Table ---
            st.subheader(f"📋 Top {top_k} Ranked Duplicate Candidates ({scoring_mode})")
            display_cols = pd.DataFrame({
                "Rank": res_df["Rank"],
                "Bug ID": res_df["Issue_id"].apply(lambda x: f"#{x}"),
                "Bug Title": res_df["Title"],
                "Component": res_df["Component"],
                "Version": res_df["Version"],
                "Extracted Errors": res_df["Extracted_Errors"],
                "SBERT Sim": res_df["SBERT_Score"].round(3),
                "TF-IDF Sim": res_df["TFIDF_Score"].round(3),
                "Cross-Encoder": res_df["CE_Score"].round(3),
                "Final Score / Prob": res_df["Final_Score"].round(3)
            })
            st.dataframe(display_cols, use_container_width=True, hide_index=True)

# === TAB 2: BENCHMARK & ABLATION STUDY ANALYTICS ===
with tab2:
    st.subheader("🏆 Main 4-Stage Architecture Benchmark")
    bench_data = pd.DataFrame([
        {"Model Stage": "1. Baseline: TF-IDF", "Recall@1 (%)": 16.8, "Recall@5 (%)": 32.4, "Recall@10 (%)": 40.0, "MRR (%)": 24.63, "Precision (%)": 89.14, "Recall (%)": 87.14, "F1-Score (%)": 88.13},
        {"Model Stage": "2. Stage 1: Sentence-BERT", "Recall@1 (%)": 21.4, "Recall@5 (%)": 40.4, "Recall@10 (%)": 46.8, "MRR (%)": 30.34, "Precision (%)": 89.82, "Recall (%)": 90.00, "F1-Score (%)": 89.91},
        {"Model Stage": "3. Stage 2: SBERT + Cross-Encoder", "Recall@1 (%)": 26.2, "Recall@5 (%)": 45.0, "Recall@10 (%)": 53.8, "MRR (%)": 34.63, "Precision (%)": 92.69, "Recall (%)": 90.61, "F1-Score (%)": 91.64},
        {"Model Stage": "4. Proposed: Transformer + Tech Metadata", "Recall@1 (%)": 28.8, "Recall@5 (%)": 49.4, "Recall@10 (%)": 56.0, "MRR (%)": 37.39, "Precision (%)": 94.44, "Recall (%)": 90.20, "F1-Score (%)": 92.28}
    ])
    st.dataframe(bench_data, use_container_width=True, hide_index=True)
    if os.path.exists("model_comparison_chart.png"):
        st.image("model_comparison_chart.png", use_container_width=True)

    st.divider()
    st.subheader("🔬 Ablation Study & Learned Meta-Classifier Feature Importance")
    if ablation_df is not None:
        st.dataframe(ablation_df.reset_index().rename(columns={"index": "Ablation Variant"}), use_container_width=True, hide_index=True)
    if os.path.exists("ablation_and_weights.png"):
        st.image("ablation_and_weights.png", use_container_width=True)

# === TAB 3: HISTORICAL BUG CORPUS EXPLORER ===
with tab3:
    st.subheader("🗄️ Explore the Eclipse JDT Historical Bug Database (9,469 Reports)")
    comp_filter = st.multiselect("Filter by Component:", options=sorted(corpus_df['Component'].unique()), default=[])
    only_errors = st.checkbox("Show only reports with Extracted Exceptions/Errors", value=True)
    filtered_corpus = corpus_df.copy()
    if comp_filter:
        filtered_corpus = filtered_corpus[filtered_corpus['Component'].isin(comp_filter)]
    if only_errors:
        filtered_corpus = filtered_corpus[filtered_corpus['Extracted_Errors'] != 'None']
    st.dataframe(filtered_corpus[['Issue_id', 'Component', 'Version', 'Extracted_Errors', 'Clean_Title']].head(100), use_container_width=True, hide_index=True)
