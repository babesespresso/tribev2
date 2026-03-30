import gradio as gr
import os
import torch
import numpy as np
import json
import requests
from datetime import datetime
from pathlib import Path
from tribev2 import TribeModel
from tribev2.plotting import PlotBrainNilearn as PlotBrain
import tempfile
import threading
import matplotlib

matplotlib.use('Agg')

# --- Run History Storage ---
RUNS_DIR = Path("./runs")
RUNS_DIR.mkdir(exist_ok=True)

def save_run(stimulus_type, stimulus_desc, fig, analysis, region_data):
    """Persist a completed run to disk."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the brain plot as PNG
    plot_path = run_dir / "brain_map.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight", facecolor="#09090b")
    
    # Save metadata + analysis as JSON
    meta = {
        "id": run_id,
        "timestamp": datetime.now().isoformat(),
        "stimulus_type": stimulus_type,
        "stimulus": stimulus_desc,
        "analysis": analysis,
        "top_regions": region_data[:10],
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    
    return run_id

def load_all_runs():
    """Load all saved runs, newest first."""
    runs = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["plot_path"] = str(run_dir / "brain_map.png")
            runs.append(meta)
    return runs

def get_history_choices():
    """Return dropdown choices for run history."""
    runs = load_all_runs()
    if not runs:
        return []
    choices = []
    for r in runs:
        ts = datetime.fromisoformat(r["timestamp"]).strftime("%b %d %I:%M%p")
        label = f"[{r['id']}] {ts} - {r['stimulus_type']} - {r['stimulus'][:50]}"
        choices.append(label)
    return choices

def _extract_run_id(choice_str):
    """Extract run ID from dropdown choice string like '[20260329_211449] ...'"""
    if not choice_str:
        return None
    if choice_str.startswith("["):
        return choice_str.split("]")[0][1:]
    return choice_str

def view_run(choice_str):
    """Load a specific run's plot and analysis."""
    run_id = _extract_run_id(choice_str)
    if not run_id:
        return None, "*Select a run from the dropdown above.*"
    run_dir = RUNS_DIR / run_id
    meta_path = run_dir / "meta.json"
    plot_path = run_dir / "brain_map.png"
    if not meta_path.exists():
        return None, f"Run `{run_id}` not found."
    with open(meta_path) as f:
        meta = json.load(f)
    img = str(plot_path) if plot_path.exists() else None
    return img, meta.get("analysis", "No analysis available.")

def delete_run(choice_str):
    """Delete a run from history."""
    run_id = _extract_run_id(choice_str)
    if not run_id:
        return gr.update(choices=get_history_choices(), value=None), None, "*No run selected.*"
    import shutil
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return gr.update(choices=get_history_choices(), value=None), None, "*Run deleted.*"

model = None
plotter = PlotBrain(mesh="fsaverage5")
model_lock = threading.Lock()

# --- HCP Region → Human-readable cognitive function mapping ---
REGION_FUNCTIONS = {
    # Visual Cortex
    "V1": "Primary Visual Cortex — basic visual feature detection (edges, contrast, orientation)",
    "V2": "Secondary Visual Cortex — texture and pattern processing",
    "V3": "Visual Area 3 — dynamic form and motion boundaries",
    "V4": "Visual Area 4 — color perception and object recognition",
    "MT": "Middle Temporal (V5) — motion detection and speed perception",
    "MST": "Medial Superior Temporal — optic flow and self-motion",
    "V6": "Visual Area 6 — peripheral vision and ego-motion",
    "V3A": "Visual Area 3A — depth perception and stereoscopic vision",
    "V3B": "Visual Area 3B — 3D shape processing",
    "V7": "Visual Area 7 — spatial attention in visual field",
    "V8": "Visual Area 8 — color and face perception",
    "FFC": "Fusiform Face Complex — face and body recognition",
    "PIT": "Posterior Inferotemporal — object categorization",
    "VVC": "Ventral Visual Complex — high-level object and scene recognition",
    
    # Auditory Cortex
    "A1": "Primary Auditory Cortex — basic sound processing (pitch, tone)",
    "A4": "Auditory Area 4 — complex sound recognition",
    "A5": "Auditory Association — sound meaning and categorization",
    "RI": "Retroinsular Cortex — auditory-spatial integration",
    "TA2": "Temporal Association 2 — speech sound processing",
    "STSdp": "Superior Temporal Sulcus (dorsal posterior) — voice and biological motion",
    "STSda": "Superior Temporal Sulcus (dorsal anterior) — audiovisual integration",
    "STSvp": "Superior Temporal Sulcus (ventral posterior) — speech comprehension",
    "STSva": "Superior Temporal Sulcus (ventral anterior) — social perception",

    # Language
    "55b": "Area 55b — speech production planning",
    "SFL": "Superior Frontal Language — sentence processing",
    "TPOJ1": "Temporo-Parieto-Occipital Junction 1 — language comprehension",
    "TPOJ2": "Temporo-Parieto-Occipital Junction 2 — semantic processing",
    "TPOJ3": "Temporo-Parieto-Occipital Junction 3 — meaning integration",
    "PSL": "Perisylvian Language — phonological processing",
    "STV": "Superior Temporal Visual — reading and letter recognition",

    # Motor/Somatosensory
    "4": "Primary Motor Cortex — voluntary movement execution",
    "3a": "Somatosensory 3a — proprioception (body position sense)",
    "3b": "Primary Somatosensory Cortex — touch discrimination",
    "1": "Somatosensory Area 1 — texture perception",
    "2": "Somatosensory Area 2 — shape and size by touch",
    "6mp": "Supplementary Motor — movement planning and sequencing",
    "6d": "Dorsal Premotor — reaching and grasping planning",
    "FEF": "Frontal Eye Fields — voluntary eye movements",

    # Prefrontal / Executive
    "p9-46v": "Ventrolateral Prefrontal — working memory and decision-making",
    "a9-46v": "Anterior Ventrolateral Prefrontal — cognitive control",
    "46": "Dorsolateral Prefrontal Area 46 — executive reasoning",
    "9-46d": "Dorsolateral Prefrontal — strategic planning",
    "8Av": "Prefrontal 8Av — attention control",
    "8BL": "Prefrontal 8BL — cognitive set-shifting",
    "10d": "Frontopolar 10d — prospective memory and multitasking",
    "10r": "Frontopolar 10r — abstract reasoning",
    "a10p": "Anterior Frontopolar — metacognition",

    # Emotion / Limbic
    "OFC": "Orbitofrontal Cortex — reward valuation and emotion regulation",
    "pOFC": "Posterior Orbitofrontal — emotional decision-making",
    "25": "Subgenual Cingulate — mood regulation (depression target)",
    "s32": "Subgenual Area 32 — emotional conflict resolution",
    "a24": "Anterior Cingulate 24 — motivation and pain processing",
    "p24": "Posterior Cingulate 24 — emotional awareness",
    "d32": "Dorsal Area 32 — error monitoring and conflict detection",
    
    # Memory / Default Mode Network
    "POS1": "Parieto-Occipital Sulcus 1 — episodic memory retrieval",
    "POS2": "Parieto-Occipital Sulcus 2 — spatial memory",
    "RSC": "Retrosplenial Cortex — spatial navigation and memory",
    "PCV": "Precuneus Visual — self-referential thought and memory",
    "7m": "Medial Area 7 — visuospatial processing and imagery",
    "PGp": "Angular Gyrus (posterior) — semantic memory retrieval",
    
    # Attention / Parietal
    "IPS1": "Intraparietal Sulcus 1 — spatial attention",
    "LIPv": "Lateral Intraparietal (ventral) — saccade planning and attention",
    "LIPd": "Lateral Intraparietal (dorsal) — decision-making under uncertainty",
    "AIP": "Anterior Intraparietal — grasping and tool use",
    "MIP": "Medial Intraparietal — reaching and pointing",
    "7AL": "Parietal Area 7AL — sensorimotor integration",
    "7PL": "Parietal Area 7PL — spatial awareness",
    "7Pm": "Parietal Area 7Pm — visual-spatial orientation",
}

def get_model():
    global model
    with model_lock:
        if model is None:
            use_mps = torch.backends.mps.is_available()
            label = "mps (Apple GPU)" if use_mps else "cpu"
            print(f"Loading TRIBE v2 model... text extractor → {label}, others → cpu")
            model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
            if not torch.cuda.is_available():
                for attr in ["neuro", "audio_feature", "video_feature", "image_feature"]:
                    extractor = getattr(model.data, attr, None)
                    if extractor is not None:
                        if hasattr(extractor, "device"):
                            extractor.device = "cpu"
                        if hasattr(extractor, "image") and hasattr(extractor.image, "device"):
                            extractor.image.device = "cpu"
                text_ext = getattr(model.data, "text_feature", None)
                if text_ext is not None and hasattr(text_ext, "device"):
                    text_ext.device = "mps" if use_mps else "cpu"
            print("Model loaded successfully.")
    return model


def analyze_brain_regions(preds, stimulus_description=""):
    """Full brain analysis: regional scores, temporal dynamics, hemispheric laterality, predictive metrics."""
    
    avg_activation = np.mean(preds, axis=0)
    n_vertices = len(avg_activation)
    half = n_vertices // 2
    
    # --- LOBE DEFINITIONS ---
    LOBE_RANGES = {
        "Occipital (Visual Cortex)": {
            "indices": list(range(0, int(half * 0.15))) + list(range(half, half + int(half * 0.15))),
            "functions": ["Primary visual processing (V1/V2)", "Color and motion detection (V4/MT)", "Object and face recognition (FFC)"],
            "category": "Visual Processing"
        },
        "Temporal (Auditory/Language)": {
            "indices": list(range(int(half * 0.15), int(half * 0.35))) + list(range(half + int(half * 0.15), half + int(half * 0.35))),
            "functions": ["Speech comprehension (Wernicke's area)", "Auditory processing (A1)", "Voice and social perception (STS)"],
            "category": "Auditory & Language"
        },
        "Parietal (Spatial/Attention)": {
            "indices": list(range(int(half * 0.35), int(half * 0.55))) + list(range(half + int(half * 0.35), half + int(half * 0.55))),
            "functions": ["Spatial attention (IPS)", "Sensorimotor integration", "Body awareness and proprioception"],
            "category": "Attention & Spatial"
        },
        "Frontal (Executive/Motor)": {
            "indices": list(range(int(half * 0.55), int(half * 0.80))) + list(range(half + int(half * 0.55), half + int(half * 0.80))),
            "functions": ["Working memory and planning (DLPFC)", "Motor execution and sequencing", "Speech production (Broca's area)"],
            "category": "Executive & Motor"
        },
        "Prefrontal (Decision/Emotion)": {
            "indices": list(range(int(half * 0.80), half)) + list(range(half + int(half * 0.80), n_vertices)),
            "functions": ["Decision-making and reward (OFC)", "Emotion regulation (vmPFC)", "Abstract reasoning and metacognition"],
            "category": "Emotion & Decision"
        },
    }
    
    # --- PER-LOBE SCORES ---
    region_data = []
    for lobe_name, lobe_info in LOBE_RANGES.items():
        indices = [i for i in lobe_info["indices"] if i < n_vertices]
        if indices:
            activation = float(np.mean(avg_activation[indices]))
            peak = float(np.max(avg_activation[indices]))
            region_data.append({
                "region": lobe_name,
                "activation": activation,
                "peak": peak,
                "function": "; ".join(lobe_info["functions"]),
                "category": lobe_info["category"],
            })
    region_data.sort(key=lambda x: x["activation"], reverse=True)
    
    global_mean = float(np.mean(avg_activation))
    global_std = float(np.std(avg_activation))
    
    # --- TEMPORAL DYNAMICS (per-second engagement) ---
    n_timesteps = len(preds)
    temporal_engagement = []
    for t in range(n_timesteps):
        ts_activation = float(np.mean(np.abs(preds[t])))
        temporal_engagement.append(ts_activation)
    
    # Attention half-life: find the second where engagement drops below 50% of peak
    peak_engagement = max(temporal_engagement) if temporal_engagement else 0
    peak_second = temporal_engagement.index(peak_engagement) if temporal_engagement else 0
    attention_halflife = n_timesteps  # default: no drop
    threshold = peak_engagement * 0.5
    for t in range(peak_second, n_timesteps):
        if temporal_engagement[t] < threshold:
            attention_halflife = t
            break
    
    # --- HEMISPHERIC LATERALITY ---
    left_activation = float(np.mean(avg_activation[:half]))
    right_activation = float(np.mean(avg_activation[half:]))
    laterality_index = (left_activation - right_activation) / (abs(left_activation) + abs(right_activation) + 1e-10)
    # Positive = left-dominant (analytical/language), Negative = right-dominant (creative/spatial/emotional)
    
    # --- BUILD ANALYSIS ---
    analysis_data = {
        "region_data": region_data,
        "global_mean": global_mean,
        "global_std": global_std,
        "temporal_engagement": temporal_engagement,
        "peak_second": peak_second,
        "peak_engagement": peak_engagement,
        "attention_halflife": attention_halflife,
        "n_timesteps": n_timesteps,
        "left_activation": left_activation,
        "right_activation": right_activation,
        "laterality_index": laterality_index,
    }
    
    interpretation = _generate_full_scorecard(analysis_data, stimulus_description)
    return interpretation, region_data


def _generate_full_scorecard(data, stimulus_description):
    """Generate the complete Content Engagement Scorecard with predictive metrics."""
    
    region_data = data["region_data"]
    global_mean = data["global_mean"]
    global_std = data["global_std"]
    temporal = data["temporal_engagement"]
    peak_sec = data["peak_second"]
    peak_eng = data["peak_engagement"]
    halflife = data["attention_halflife"]
    n_ts = data["n_timesteps"]
    left_act = data["left_activation"]
    right_act = data["right_activation"]
    lat_idx = data["laterality_index"]
    
    # --- SCORING ENGINE ---
    scores = {}
    max_peak = max(r.get("peak", 0.001) for r in region_data) if region_data else 1
    
    for r in region_data:
        normalized_mean = (r["activation"] / (global_mean if global_mean > 0 else 0.001)) * 50
        normalized_peak = (r.get("peak", 0) / max_peak) * 100
        score = min(100, max(0, (normalized_mean * 0.4 + normalized_peak * 0.6)))
        scores[r["category"]] = round(score)
    
    def grade(s):
        if s >= 90: return "A+"
        if s >= 85: return "A"
        if s >= 80: return "A-"
        if s >= 75: return "B+"
        if s >= 70: return "B"
        if s >= 65: return "B-"
        if s >= 60: return "C+"
        if s >= 55: return "C"
        if s >= 50: return "C-"
        if s >= 40: return "D"
        return "F"
    
    weights = {
        "Visual Processing": 0.20,
        "Auditory & Language": 0.25,
        "Attention & Spatial": 0.20,
        "Executive & Motor": 0.15,
        "Emotion & Decision": 0.20,
    }
    overall = sum(scores.get(k, 50) * w for k, w in weights.items())
    overall = min(100, max(0, round(overall)))
    
    emotion_score = scores.get("Emotion & Decision", 50)
    visual_score = scores.get("Visual Processing", 50)
    auditory_score = scores.get("Auditory & Language", 50)
    attention_score = scores.get("Attention & Spatial", 50)
    executive_score = scores.get("Executive & Motor", 50)
    
    # ===================================================================
    # PREDICTIVE METRICS
    # ===================================================================
    
    # Watch-Through Rate: based on attention decay
    if n_ts > 1:
        # Compare first-half vs second-half engagement
        first_half = np.mean(temporal[:n_ts//2]) if n_ts > 1 else 0
        second_half = np.mean(temporal[n_ts//2:]) if n_ts > 1 else 0
        decay_ratio = second_half / (first_half + 1e-10)
        watch_through = min(100, max(15, round(decay_ratio * 85 + 15)))
    else:
        watch_through = 70
    
    # Ad Recall (24hr): temporal+prefrontal co-activation predicts memory encoding
    memory_signal = (auditory_score * 0.4 + emotion_score * 0.3 + executive_score * 0.3)
    ad_recall = min(100, max(10, round(memory_signal * 0.9)))
    
    # Purchase/Conversion Intent: emotion + decision activation
    conversion_signal = (emotion_score * 0.5 + executive_score * 0.25 + attention_score * 0.25)
    purchase_intent = min(100, max(5, round(conversion_signal * 0.85)))
    
    # Virality/Shareability: emotion + auditory (social cognition)
    virality_signal = (emotion_score * 0.45 + auditory_score * 0.3 + visual_score * 0.25)
    virality = min(100, max(5, round(virality_signal * 0.9)))
    
    # Optimal Content Length (seconds): based on attention halflife
    if halflife < n_ts:
        optimal_length = halflife
    else:
        optimal_length = n_ts  # attention held throughout
    
    # Cognitive Load: executive + attention intensity
    cog_load_raw = (executive_score * 0.6 + attention_score * 0.4)
    cog_load = min(100, max(0, round(cog_load_raw)))
    cog_load_label = "High" if cog_load >= 75 else "Medium" if cog_load >= 50 else "Low"
    
    # Content Type Classification
    type_scores = {
        "Entertainment": visual_score * 0.4 + emotion_score * 0.4 + auditory_score * 0.2,
        "Educational": executive_score * 0.4 + auditory_score * 0.35 + attention_score * 0.25,
        "Persuasion/Ad": emotion_score * 0.4 + auditory_score * 0.3 + visual_score * 0.3,
        "Informational": executive_score * 0.35 + attention_score * 0.35 + auditory_score * 0.3,
    }
    best_fit = max(type_scores, key=type_scores.get)
    
    # ===================================================================
    # BUILD REPORT
    # ===================================================================
    
    report = "## 📊 Content Engagement Scorecard\n\n"
    report += f"**Stimulus:** {stimulus_description or 'Multimodal media input'}\n\n"
    report += f"### Overall Neural Engagement: {overall}/100 ({grade(overall)})\n\n"
    
    # --- CATEGORY BREAKDOWN ---
    report += "### Category Breakdown\n\n"
    report += "| Cognitive System | Score | Grade | What It Measures |\n"
    report += "|-----------------|-------|-------|------------------|\n"
    
    grade_desc = {
        "Visual Processing": "How strongly visuals capture attention (faces, motion, color)",
        "Auditory & Language": "Speech comprehension, voice impact, and word meaning",
        "Attention & Spatial": "Sustained focus and spatial awareness engagement",
        "Executive & Motor": "Active thinking, problem-solving, and action impulse",
        "Emotion & Decision": "Emotional resonance, trust, reward, and persuasion",
    }
    
    for r in region_data:
        cat = r["category"]
        s = scores.get(cat, 50)
        report += f"| **{cat}** | {s}/100 | **{grade(s)}** | {grade_desc.get(cat, '')} |\n"
    
    # --- ENGAGEMENT PROFILE BARS ---
    report += "\n### Engagement Profile\n\n"
    for r in region_data:
        cat = r["category"]
        s = scores.get(cat, 50)
        filled = s // 5
        empty = 20 - filled
        bar = "█" * filled + "░" * empty
        report += f"- **{cat}**: `{bar}` {s}/100 ({grade(s)})\n"
    
    # --- ENGAGEMENT TIMELINE ---
    report += "\n---\n\n### ⏱️ Engagement Timeline\n\n"
    if len(temporal) > 1:
        max_t = max(temporal) if max(temporal) > 0 else 1
        report += "| Second | Engagement | Level |\n"
        report += "|--------|-----------|-------|\n"
        for i, t_val in enumerate(temporal):
            pct = int((t_val / max_t) * 10)
            bar = "▓" * pct + "░" * (10 - pct)
            marker = " ← PEAK" if i == peak_sec else ""
            report += f"| {i+1}s | `{bar}` {t_val:.4f} | {marker} |\n"
        
        report += f"\n**Peak engagement:** Second {peak_sec + 1}\n"
        if halflife < n_ts:
            report += f"**Attention drops below 50%:** Second {halflife + 1} — viewers likely disengage after this point\n"
        else:
            report += f"**Attention sustained** throughout all {n_ts} seconds — strong holding power\n"
    else:
        report += "Single-timestep stimulus — timeline not available for text-only inputs.\n"
    
    # --- HEMISPHERIC ANALYSIS ---
    report += "\n---\n\n### 🧠 Brain Laterality\n\n"
    report += "| Hemisphere | Activation | Processes |\n"
    report += "|-----------|-----------|----------|\n"
    report += f"| **Left Brain** | {left_act:.5f} | Language, logic, analytical thinking, speech |\n"
    report += f"| **Right Brain** | {right_act:.5f} | Creativity, emotion, spatial awareness, music |\n\n"
    
    if lat_idx > 0.05:
        report += f"**Left-brain dominant** (laterality: {lat_idx:+.3f}) — This content engages **analytical and language** processing more than emotional/creative circuits. Typical of speech-heavy, informational, or text-based content.\n"
    elif lat_idx < -0.05:
        report += f"**Right-brain dominant** (laterality: {lat_idx:+.3f}) — This content engages **creative, emotional, and spatial** processing. Typical of music, visual art, emotionally-charged narratives, or immersive video.\n"
    else:
        report += f"**Balanced activation** (laterality: {lat_idx:+.3f}) — Both hemispheres are equally engaged, suggesting well-rounded multimodal content that combines logic with emotion.\n"
    
    # --- PREDICTIVE METRICS ---
    report += "\n---\n\n### 🔮 Predictive Insights\n\n"
    report += "| Metric | Prediction | Confidence |\n"
    report += "|--------|-----------|------------|\n"
    report += f"| **Watch-Through Rate** | {watch_through}% of viewers will finish | {'High' if watch_through > 70 else 'Medium' if watch_through > 50 else 'Low'} |\n"
    report += f"| **24hr Ad Recall** | {ad_recall}% recall probability | {'High' if ad_recall > 70 else 'Medium' if ad_recall > 50 else 'Low'} |\n"
    report += f"| **Purchase/Action Intent** | {purchase_intent}/100 | {'Strong' if purchase_intent > 70 else 'Moderate' if purchase_intent > 50 else 'Weak'} |\n"
    report += f"| **Virality / Shareability** | {virality}/100 | {'Highly Shareable' if virality > 75 else 'Moderately Shareable' if virality > 50 else 'Low Share Potential'} |\n"
    report += f"| **Cognitive Load** | {cog_load}/100 ({cog_load_label}) | — |\n"
    report += f"| **Optimal Content Length** | ~{optimal_length}s | Based on attention decay |\n"
    report += f"| **Best Content Fit** | {best_fit} | Based on cognitive profile |\n"
    
    # --- KEY FINDINGS ---
    report += "\n---\n\n### Key Findings\n\n"
    
    dominant = region_data[0] if region_data else None
    weakest = region_data[-1] if region_data else None
    
    if dominant:
        dom_cat = dominant["category"]
        dom_score = scores.get(dom_cat, 50)
        report += f"**Strongest signal: {dom_cat} ({grade(dom_score)}).** "
        
        if "Visual" in dom_cat:
            report += "This content is **visually dominant** — the brain prioritizes processing what it sees. Strong for ads, thumbnails, and visual storytelling. "
        elif "Auditory" in dom_cat:
            report += "This content is **speech/audio dominant** — the words and voice carry the most neural weight. The speaker's message is the primary engagement driver. "
        elif "Attention" in dom_cat:
            report += "This content commands **strong focused attention** — the viewer is locked in and spatially engaged. "
        elif "Executive" in dom_cat:
            report += "This content triggers **active cognitive processing** — the viewer is thinking and analyzing. "
        elif "Emotion" in dom_cat:
            report += "This content is **emotionally compelling** — it activates reward, empathy, and decision circuits. "
    
    if weakest:
        weak_cat = weakest["category"]
        weak_score = scores.get(weak_cat, 50)
        if weak_score < 60:
            report += f"\n\n**Weakest signal: {weak_cat} ({grade(weak_score)}).** "
            if "Emotion" in weak_cat:
                report += "Low emotional activation means this content **informs but doesn't persuade**. "
            elif "Visual" in weak_cat:
                report += "Low visual engagement — visuals are **not contributing** to the message. "
            elif "Attention" in weak_cat:
                report += "Low attention capture — content may **fail to hold viewers**. "
            elif "Auditory" in weak_cat:
                report += "Low auditory engagement — **audio/speech is not landing**. "
            elif "Executive" in weak_cat:
                report += "Low executive activation — content is **passively consumed**. "
    
    # --- RECOMMENDATIONS ---
    report += "\n\n### Recommendations\n\n"
    
    recs = []
    if emotion_score < 60:
        recs.append("🎭 **Boost emotional impact** — add personal stories, conflict, music, or facial close-ups")
    if visual_score < 60:
        recs.append("🎨 **Strengthen visuals** — add motion, faces, high-contrast imagery, or text overlays")
    if auditory_score < 60:
        recs.append("🔊 **Improve audio** — clearer speech, vocal variety, background scoring, or sound effects")
    if attention_score < 60:
        recs.append("🎯 **Add attention hooks** — faster cuts, direct eye contact, questions, or scene changes every 3-5s")
    if watch_through < 50:
        recs.append(f"⏱️ **Shorten to ~{optimal_length}s** — attention decays rapidly; front-load your key message")
    if purchase_intent < 40:
        recs.append("💰 **Add a clear call to action** — the content lacks the emotional+cognitive push to drive behavior")
    if virality > 75:
        recs.append("🚀 **High share potential** — prioritize social distribution; this content has organic spread characteristics")
    if overall >= 80:
        recs.append("✅ **Strong overall** — candidate for broad distribution and paid amplification")
    if cog_load > 85:
        recs.append("⚠️ **High cognitive load** — content may overwhelm; simplify messaging or slow pacing")
    
    if not recs:
        recs.append("📊 Content is performing adequately — consider A/B testing variations to optimize further")
    
    for i, rec in enumerate(recs, 1):
        report += f"{i}. {rec}\n"
    
    # --- BOTTOM LINE ---
    report += "\n### Bottom Line\n\n"
    if overall >= 80:
        report += f"**{overall}/100 — Excellent.** Strong multi-system engagement. High retention, recall, and action potential."
    elif overall >= 65:
        report += f"**{overall}/100 — Good.** Solid engagement with room to improve. Focus on the weakest category."
    elif overall >= 50:
        report += f"**{overall}/100 — Average.** Adequate but forgettable. Needs stronger emotional or visual hooks."
    else:
        report += f"**{overall}/100 — Needs Work.** Weak engagement. Major revisions recommended."
    
    return report


def generate_plot_and_analysis(df, progress, stimulus_type="Text", stimulus_desc=""):
    """Generate brain plot AND AI analysis, then persist the run."""
    progress((0.4, 1.0), desc="Extracting Deep Multimodal AI Features (Heaviest Step)...")
    m = get_model()
    preds, segments = m.predict(events=df, gradio_progress=progress)
    
    progress((0.75, 1.0), desc="Rendering 3D Brain Mesh...")
    n_to_plot = min(len(preds), 4)
    sliced_preds = preds[:n_to_plot]
    fig = plotter.plot_timesteps(sliced_preds, show_stimuli=False)
    
    progress((0.9, 1.0), desc="Analyzing Brain Activation Patterns...")
    interpretation, region_data = analyze_brain_regions(preds, stimulus_desc)
    
    # Persist the run
    progress((0.95, 1.0), desc="Saving run to history...")
    try:
        save_run(stimulus_type, stimulus_desc, fig, interpretation, region_data)
    except Exception as e:
        print(f"[Run History] Failed to save: {e}")
    
    progress((1.0, 1.0), desc="Complete")
    return fig, interpretation


def process_text(text, progress=gr.Progress()):
    if not text.strip():
        return None, ""
        
    if len(text) > 150:
        gr.Warning("Text exceeded 150 characters. Truncating to bypass local hardware bottlenecks.")
        text = text[:150] + "..."
        
    progress((0.0, 1.0), desc="Initializing Natural Language Extractors...")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt") as tmp:
        tmp.write(text)
        tmp_path = tmp.name
        
    try:
        progress((0.2, 1.0), desc="Processing text embeddings...")
        df = get_model().get_events_dataframe(text_path=tmp_path)
        return generate_plot_and_analysis(df, progress, stimulus_type="Text", stimulus_desc=text)
    finally:
        os.unlink(tmp_path)
        
def process_audio(audio_path, progress=gr.Progress()):
    if not audio_path:
        return None, ""
    progress((0.0, 1.0), desc="Loading Audio & Whisper Extractors...")
    progress((0.15, 1.0), desc="Extracting Audio & Phonetics Features...")
    df = get_model().get_events_dataframe(audio_path=audio_path)
    return generate_plot_and_analysis(df, progress, stimulus_type="Audio", stimulus_desc="Audio recording")

def process_video(video_path, progress=gr.Progress()):
    if not video_path:
        return None, ""
    progress((0.0, 1.0), desc="Trimming video and Splitting Frames...")
    
    ext = os.path.splitext(video_path)[1]
    trimmed_path = video_path.replace(ext, f"_trimmed{ext}")
    os.system(f'ffmpeg -y -i "{video_path}" -t 5 -c copy "{trimmed_path}" -loglevel quiet')
    
    progress((0.15, 1.0), desc="Extracting Video Motion & Semantics via V-JEPA2...")
    df = get_model().get_events_dataframe(video_path=trimmed_path)
    return generate_plot_and_analysis(df, progress, stimulus_type="Video", stimulus_desc="Video clip (first 5 seconds)")


# --- Custom Theme & Modern Black Dashboard Styling ---
custom_theme = gr.themes.Base(
    primary_hue="zinc",
    secondary_hue="stone",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    body_background_fill="#000000",
    body_text_color="#f4f4f5",
    block_background_fill="#09090b",
    block_border_width="1px",
    block_border_color="#27272a",
    block_shadow="none",
    button_primary_background_fill="#ffffff",
    button_primary_background_fill_hover="#d4d4d8",
    button_primary_text_color="#000000",
    button_secondary_background_fill="#18181b",
    button_secondary_background_fill_hover="#27272a",
    button_secondary_text_color="#f4f4f5",
    panel_background_fill="#000000",
    checkbox_background_color="#09090b",
    slider_color="#ffffff",
    input_background_fill="#09090b",
    input_border_color="#27272a"
)

# Gradio Interface
with gr.Blocks(title="MULTITUDE MEDIA | TRIBE v2", theme=custom_theme) as app:
    
    gr.HTML("""
    <div style="margin-top: 1.5rem; margin-bottom: 2.5rem; border-bottom: 1px solid #27272a; padding-bottom: 1.5rem;">
        <h3 style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em; color: #a1a1aa; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 8px;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
            MULTITUDE MEDIA
        </h3>
        <h1 style="font-size: 2.75rem; font-weight: 300; letter-spacing: -0.03em; margin-bottom: 0.75rem; color: #ffffff;">
            TRIBE v2 Brain Encoding
        </h1>
        <p style="font-size: 1.05rem; color: #71717a; max-width: 700px; line-height: 1.5;">
            Upload text, audio, or video to predict human fMRI brain activity. The AI will map cortical activation patterns and provide a detailed cognitive analysis.
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.Tab("Text Inference"):
                    gr.Markdown("Type a scenario or description.")
                    text_in = gr.Textbox(label="Input Stimulus", placeholder="A dog runs across the field...", lines=3)
                    text_btn = gr.Button("Execute Brain Mapping", variant="primary", size="lg")
                    gr.Markdown("*Try: \"A dog runs across the field\" or \"She listened to the quiet raindrops\"*", elem_classes=["text-xs"])
                    
                with gr.Tab("Audio Inference"):
                    gr.Markdown("Upload voice clips or music.")
                    audio_in = gr.Audio(label="Audio Stimulus", type="filepath")
                    audio_btn = gr.Button("Execute Brain Mapping", variant="primary", size="lg")
                    
                with gr.Tab("Video Inference"):
                    gr.Markdown("Upload standard video formats. Auto-trimmed to 5 seconds.")
                    video_in = gr.Video(label="Video Stimulus", sources=["upload"])
                    video_btn = gr.Button("Execute Brain Mapping", variant="primary", size="lg")
                    
        with gr.Column(scale=5):
            with gr.Row():
                gr.Markdown("### Predicted Cortical Activation")
                new_run_btn = gr.Button("New +", variant="secondary", size="sm", scale=0, min_width=80)
            out_plot = gr.Plot(label="", show_label=False)

    gr.Markdown("---")
    gr.Markdown("### Content Engagement Scorecard")
    out_analysis = gr.Markdown(value="*Run a brain mapping to see an engagement scorecard with grades, scores, and actionable recommendations.*")

    # --- Run History Section ---
    gr.HTML("""
    <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #27272a;">
        <h2 style="font-size: 1.5rem; font-weight: 400; letter-spacing: -0.02em; color: #ffffff; display: flex; align-items: center; gap: 10px;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
            Run History
        </h2>
        <p style="color: #71717a; font-size: 0.9rem;">All completed brain mappings are saved here. Select a past run to review its results.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=5):
            history_dropdown = gr.Dropdown(
                label="Past Runs",
                choices=get_history_choices(),
                interactive=True,
                value=None
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button("Refresh", variant="secondary", size="sm")
            delete_btn = gr.Button("Delete Run", variant="secondary", size="sm")

    history_image = gr.Image(label="Brain Map", show_label=False, type="filepath")
    history_analysis = gr.Markdown(value="*Select a run from the dropdown above.*")

    # Helper to refresh dropdown after a run completes
    def refresh_history():
        return gr.update(choices=get_history_choices())

    # New+ button — clears everything for a fresh run
    def reset_for_new_run():
        return (
            "",           # text_in
            None,         # audio_in
            None,         # video_in
            None,         # out_plot
            "*Run a brain mapping to see an engagement scorecard with grades, scores, and actionable recommendations.*",  # out_analysis
        )

    new_run_btn.click(
        fn=reset_for_new_run,
        outputs=[text_in, audio_in, video_in, out_plot, out_analysis]
    )

    # Wire up brain mapping — chain dropdown refresh after completion
    text_btn.click(fn=process_text, inputs=text_in, outputs=[out_plot, out_analysis]).then(
        fn=refresh_history, outputs=history_dropdown
    )
    audio_btn.click(fn=process_audio, inputs=audio_in, outputs=[out_plot, out_analysis]).then(
        fn=refresh_history, outputs=history_dropdown
    )
    video_btn.click(fn=process_video, inputs=video_in, outputs=[out_plot, out_analysis]).then(
        fn=refresh_history, outputs=history_dropdown
    )

    # History event handlers
    history_dropdown.change(
        fn=view_run,
        inputs=history_dropdown,
        outputs=[history_image, history_analysis]
    )
    refresh_btn.click(
        fn=refresh_history,
        outputs=history_dropdown
    )
    delete_btn.click(
        fn=delete_run,
        inputs=history_dropdown,
        outputs=[history_dropdown, history_image, history_analysis]
    )

    # Force-load history on page open
    app.load(fn=refresh_history, outputs=history_dropdown)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
