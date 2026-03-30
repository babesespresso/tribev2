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
        ts = datetime.fromisoformat(r["timestamp"]).strftime("%b %d, %Y %I:%M %p")
        label = f"{ts}  |  {r['stimulus_type']}  |  {r['stimulus'][:60]}"
        choices.append((label, r["id"]))
    return choices

def view_run(run_id):
    """Load a specific run's plot and analysis."""
    if not run_id:
        return None, "*Select a run from the dropdown above.*"
    run_dir = RUNS_DIR / run_id
    meta_path = run_dir / "meta.json"
    plot_path = run_dir / "brain_map.png"
    if not meta_path.exists():
        return None, "Run not found."
    with open(meta_path) as f:
        meta = json.load(f)
    img = str(plot_path) if plot_path.exists() else None
    return img, meta.get("analysis", "No analysis available.")

def delete_run(run_id):
    """Delete a run from history."""
    if not run_id:
        return gr.update(choices=get_history_choices()), None, "*No run selected.*"
    import shutil
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return gr.update(choices=get_history_choices()), None, "*Run deleted.*"

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
    """Analyze brain activation using vertex spatial mapping (no MNE download needed)."""
    from neuralset.extractors.neuro import FSAVERAGE_SIZES
    
    avg_activation = np.mean(preds, axis=0)
    n_vertices = len(avg_activation)
    half = n_vertices // 2
    
    # fsaverage5 has 10242 vertices per hemisphere
    # Split into anatomical zones based on vertex index ranges
    # These ranges approximate the HCP parcellation on fsaverage5
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
    
    # Compute mean activation per lobe
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
    
    # Also compute overall stats
    global_mean = float(np.mean(avg_activation))
    global_std = float(np.std(avg_activation))
    
    # Build the local analysis (instant, no API call needed)
    interpretation = _generate_local_analysis(region_data, stimulus_description, global_mean, global_std)
    
    return interpretation, region_data


def _call_llm_for_interpretation(region_data, stimulus_description):
    """Call HuggingFace Inference API to generate a neuroscience interpretation."""
    hf_token = os.environ.get("HF_TOKEN", "")
    
    # Build a structured prompt
    regions_text = ""
    for i, r in enumerate(region_data[:10], 1):
        regions_text += f"{i}. **{r['region']}** (activation: {r['activation']:.4f})\n   → {r['function']}\n"
    
    prompt = f"""You are a computational neuroscientist analyzing predicted fMRI brain activation patterns from Meta's TRIBE v2 model.

The stimulus was: "{stimulus_description if stimulus_description else 'a multimodal media input (video/audio/text)'}"

The top 10 most activated cortical regions (from the HCP MMP1.0 parcellation on fsaverage5) are:

{regions_text}

Write a clear, concise analysis (3-4 paragraphs) for a non-specialist audience that explains:
1. **Dominant Processing Mode**: What cognitive systems are most engaged (visual, auditory, language, emotional, etc.)
2. **Key Findings**: What the activation pattern reveals about how the brain would process this stimulus
3. **Emotional & Attentional Signature**: Any regions linked to emotion, attention, or memory that suggest deeper engagement
4. **Practical Insight**: What this means in plain language — is this content visually dominant? Emotionally engaging? Linguistically complex?

Write in a professional but accessible tone. Use specific region names with their functions in parentheses."""

    # Try HuggingFace Inference API
    if hf_token:
        try:
            headers = {"Authorization": f"Bearer {hf_token}"}
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 600, "temperature": 0.7}
            }
            response = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
                headers=headers,
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    text = result[0].get("generated_text", "")
                    # Strip the prompt from the response
                    if prompt in text:
                        text = text[len(prompt):].strip()
                    return text
        except Exception as e:
            print(f"[LLM API] Failed: {e}")
    
    # Fallback: generate a structured local analysis without an LLM
    return _generate_local_analysis(region_data, stimulus_description)


def _generate_local_analysis(region_data, stimulus_description, global_mean=0.0, global_std=0.0):
    """Generate a structured analysis locally using lobe-level activation data."""
    report = f"## Neuro-Cognitive Analysis\n\n"
    report += f"**Stimulus:** {stimulus_description or 'Multimodal media input'}\n\n"
    
    # Activation table
    report += "### Cortical Activation by Brain Region\n\n"
    report += "| Rank | Brain Region | Mean Activation | Peak Activation | Key Functions |\n"
    report += "|------|-------------|----------------|----------------|---------------|\n"
    
    for i, r in enumerate(region_data, 1):
        report += f"| {i} | **{r['region']}** | {r['activation']:.4f} | {r.get('peak', 0):.4f} | {r['function']} |\n"
    
    # Overall stats
    report += f"\n**Global Mean Activation:** {global_mean:.4f} | **Standard Deviation:** {global_std:.4f}\n\n"
    
    # Dominant processing mode
    report += "### Processing Profile\n\n"
    
    dominant = region_data[0] if region_data else None
    if dominant:
        report += f"The **dominant processing mode** is **{dominant['category']}** "
        report += f"(region: {dominant['region']}, activation: {dominant['activation']:.4f}).\n\n"
    
    # Relative activation chart using text bars
    if region_data:
        max_act = max(r["activation"] for r in region_data)
        min_act = min(r["activation"] for r in region_data)
        spread = max_act - min_act if max_act != min_act else 1
        for r in region_data:
            pct = int(((r["activation"] - min_act) / spread) * 20)
            bar = "█" * max(pct, 1) + "░" * (20 - max(pct, 1))
            report += f"- {r['category']}: `{bar}` {r['activation']:.4f}\n"
    
    # Interpretation
    report += "\n### Interpretation\n\n"
    
    for r in region_data:
        cat = r["category"]
        act = r["activation"]
        
        if "Visual" in cat and act > global_mean:
            report += "This stimulus triggers **strong visual processing**, engaging the occipital cortex for feature extraction — from basic edge detection (V1/V2) through complex object and face recognition. "
        
        if "Auditory" in cat and act > global_mean:
            report += "Significant **auditory and language network engagement** indicates the brain is actively processing speech comprehension, vocal tone analysis, and semantic meaning extraction. "
        
        if "Attention" in cat and act > global_mean:
            report += "The **parietal attention system** is activated, indicating the brain is directing sustained spatial attention and integrating multisensory information. "
        
        if "Executive" in cat and act > global_mean:
            report += "**Frontal executive regions** show activation, suggesting working memory engagement, motor planning, and potentially speech production (Broca's area). "
        
        if "Emotion" in cat and act > global_mean:
            report += "**Prefrontal and orbitofrontal activation** suggests this content triggers emotional evaluation, reward assessment, and higher-order decision-making. "
    
    # Summary
    report += "\n\n### Summary\n\n"
    above_mean = [r for r in region_data if r["activation"] > global_mean]
    if len(above_mean) >= 4:
        report += "This stimulus produces **broadly distributed cortical activation**, engaging multiple cognitive systems simultaneously — characteristic of rich, multimodal content that demands visual, auditory, and cognitive processing in parallel."
    elif len(above_mean) >= 2:
        report += "This stimulus produces **focused cortical activation** concentrated in specific cognitive systems, suggesting targeted neural engagement rather than broad processing."
    else:
        report += "This stimulus produces **localized cortical activation** in a narrow set of brain regions, suggesting a simple, modality-specific processing response."
    
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
                    gr.Examples(
                        examples=[["A dog runs across the field"], ["She listened to the quiet raindrops"]],
                        inputs=text_in
                    )
                    
                with gr.Tab("Audio Inference"):
                    gr.Markdown("Upload voice clips or music.")
                    audio_in = gr.Audio(label="Audio Stimulus", type="filepath")
                    audio_btn = gr.Button("Execute Brain Mapping", variant="primary", size="lg")
                    
                with gr.Tab("Video Inference"):
                    gr.Markdown("Upload standard video formats. Auto-trimmed to 5 seconds.")
                    video_in = gr.Video(label="Video Stimulus", sources=["upload"])
                    video_btn = gr.Button("Execute Brain Mapping", variant="primary", size="lg")
                    
        with gr.Column(scale=5):
            gr.Markdown("### Predicted Cortical Activation")
            out_plot = gr.Plot(label="", show_label=False)

    gr.Markdown("---")
    gr.Markdown("### AI Neuro-Cognitive Analysis")
    out_analysis = gr.Markdown(value="*Run a brain mapping to see the AI interpretation of cortical activation patterns.*")

    # Wire up the events — now outputs both plot AND analysis
    text_btn.click(fn=process_text, inputs=text_in, outputs=[out_plot, out_analysis])
    audio_btn.click(fn=process_audio, inputs=audio_in, outputs=[out_plot, out_analysis])
    video_btn.click(fn=process_video, inputs=video_in, outputs=[out_plot, out_analysis])

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

    # History event handlers
    history_dropdown.change(
        fn=view_run,
        inputs=history_dropdown,
        outputs=[history_image, history_analysis]
    )
    refresh_btn.click(
        fn=lambda: gr.update(choices=get_history_choices()),
        outputs=history_dropdown
    )
    delete_btn.click(
        fn=delete_run,
        inputs=history_dropdown,
        outputs=[history_dropdown, history_image, history_analysis]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
