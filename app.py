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
import multiprocessing

# Maximize PyTorch CPU threading for CPU-bound video extraction
# (MacOS defaults to limiting PyTorch to 4 threads dynamically)
try:
    cpu_cores = multiprocessing.cpu_count()
    torch.set_num_threads(cpu_cores)
    print(f"PyTorch CPU threads uncapped to {cpu_cores}")
except Exception as e:
    print(f"Could not uncap PyTorch threads: {e}")

try:
    import spaces
except ImportError:
    class SpacesPlaceholder:
        def GPU(self, *args, **kwargs):
            def decorator(func): return func
            return decorator
    spaces = SpacesPlaceholder()

matplotlib.use('Agg')

# --- Run History Storage ---
RUNS_DIR = Path("./runs")
RUNS_DIR.mkdir(exist_ok=True)

def save_run(stimulus_type, stimulus_desc, fig, analysis, region_data, media_path=None):
    """Persist a completed run to disk and generate a professional multi-page PDF."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract snapshot
    snapshot_path = None
    if media_path and stimulus_type == "Video":
        snapshot_path = run_dir / "snapshot.jpg"
        import os
        # Try capturing at 1s, fallback to frame 1
        os.system(f'ffmpeg -y -i "{media_path}" -ss 00:00:01.000 -vframes 1 "{snapshot_path}" -loglevel quiet')
        if not snapshot_path.exists():
            os.system(f'ffmpeg -y -i "{media_path}" -vframes 1 "{snapshot_path}" -loglevel quiet')

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
    
    # --- Professional Multi-Page PDF Generation ---
    pdf_path_str = None
    try:
        from fpdf import FPDF
        import re

        # Color palette
        BLACK = (9, 9, 11)
        DARK_BG = (24, 24, 27)
        CARD_BG = (39, 39, 42)
        WHITE = (244, 244, 245)
        MUTED = (161, 161, 170)
        ACCENT = (99, 102, 241)  # indigo
        
        def grade_color(g):
            if g.startswith('A'): return (34, 197, 94)   # green
            if g.startswith('B'): return (59, 130, 246)  # blue
            if g.startswith('C'): return (250, 204, 21)  # yellow
            if g.startswith('D'): return (249, 115, 22)  # orange
            return (239, 68, 68)                          # red / F

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

        class ScorecardPDF(FPDF):
            def header(self):
                self.set_fill_color(*BLACK)
                self.rect(0, 0, 210, 297, 'F')
                self.set_font('Helvetica', '', 8)
                self.set_text_color(*MUTED)
                self.set_y(8)
                self.cell(0, 4, 'MULTITUDE MEDIA', align='L')
                self.cell(0, 4, datetime.now().strftime('%B %d, %Y'), align='R')
                self.ln(8)

            def footer(self):
                self.set_y(-12)
                self.set_font('Helvetica', 'I', 7)
                self.set_text_color(*MUTED)
                self.cell(0, 5, f'TRIBE v2 Brain Encoding Report  |  Page {self.page_no()}', align='C')

            def section_title(self, title):
                if self.get_y() > 255:
                    self.add_page()
                self.ln(4)
                self.set_font('Helvetica', 'B', 13)
                self.set_text_color(*WHITE)
                self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
                self.set_draw_color(*ACCENT)
                self.set_line_width(0.6)
                self.line(10, self.get_y(), 100, self.get_y())
                self.ln(4)

            def safe_write(self, h, txt):
                clean = txt.encode('ascii', 'ignore').decode('ascii')
                self.write(h, clean)

        pdf = ScorecardPDF()
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.add_page()

        # === PAGE 1: Title + Brain Map ===
        pdf.set_font('Helvetica', 'B', 22)
        pdf.set_text_color(*WHITE)
        pdf.cell(0, 10, 'Content Engagement Scorecard', new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(*MUTED)
        stim_clean = stimulus_desc.encode('ascii', 'ignore').decode('ascii') if stimulus_desc else 'Multimodal input'
        pdf.cell(0, 6, f'Stimulus: {stim_clean}', new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.ln(6)

        from PIL import Image as PILImage

        # Video Snapshot (Optional)
        if snapshot_path and snapshot_path.exists():
            try:
                with PILImage.open(str(snapshot_path)) as snap:
                    s_w, s_h = snap.size
                snap_w = 70
                snap_h = snap_w * (s_h / s_w)
                # Cap the snapshot height to prevent tall vertical videos from consuming the page
                if snap_h > 60:
                    snap_h = 60
                    snap_w = snap_h * (s_w / s_h)
                    
                pdf.image(str(snapshot_path), x=(210 - snap_w) / 2, y=pdf.get_y(), w=snap_w, h=snap_h)
                pdf.set_y(pdf.get_y() + snap_h + 6)
            except Exception as e:
                pass

        # Brain map image — fit to width and let it flow
        img_w = 180
        from PIL import Image as PILImage
        try:
            with PILImage.open(str(plot_path)) as im:
                w_px, h_px = im.size
            img_h = img_w * (h_px / w_px)
            # Cap height dynamically so it NEVER hits the footer (y=285)
            max_img_h = 275 - pdf.get_y()
            if img_h > max_img_h:
                img_h = max_img_h
                img_w = img_h * (w_px / h_px)
        except Exception:
            img_h = 120
        pdf.image(str(plot_path), x=(210 - img_w) / 2, y=pdf.get_y(), w=img_w, h=img_h)
        pdf.set_y(pdf.get_y() + img_h + 6)

        # === Extract scores from the markdown analysis for PDF tables ===
        scores_map = {}
        grade_desc_map = {
            "Auditory & Language": "Speech comprehension, voice impact, and word meaning",
            "Visual Processing": "How strongly visuals capture attention (faces, motion, color)",
            "Executive & Motor": "Active thinking, problem-solving, and action impulse",
            "Attention & Spatial": "Sustained focus and spatial awareness engagement",
            "Emotion & Decision": "Emotional resonance, trust, reward, and persuasion",
        }
        # Parse scores from the markdown table rows
        for line in analysis.split('\n'):
            for cat in grade_desc_map:
                if cat in line and '/100' in line:
                    m = re.search(r'(\d+)/100', line)
                    if m:
                        scores_map[cat] = int(m.group(1))
        # Parse the overall score
        overall_match = re.search(r'Overall Neural Engagement:\s*(\d+)/100', analysis)
        overall_score = int(overall_match.group(1)) if overall_match else 0
        overall_grade = grade(overall_score)

        # === Overall Score Badge ===
        if pdf.get_y() > 250:
            pdf.add_page()
        gc = grade_color(overall_grade)
        pdf.set_fill_color(*gc)
        badge_w = 60
        badge_x = (210 - badge_w) / 2
        pdf.set_xy(badge_x, pdf.get_y())
        pdf.set_font('Helvetica', 'B', 18)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(badge_w, 12, f'{overall_score}/100  ({overall_grade})', align='C', fill=True)
        pdf.ln(4)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 5, 'Overall Neural Engagement', align='C', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # === PAGE 2+: Category Breakdown Table ===
        pdf.section_title('Category Breakdown')
        col_widths = [55, 18, 16, 101]
        headers = ['Cognitive System', 'Score', 'Grade', 'What It Measures']
        # Table header
        pdf.set_fill_color(*CARD_BG)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(*WHITE)
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 7, h, border=0, fill=True)
        pdf.ln()
        # Table rows
        pdf.set_font('Helvetica', '', 9)
        row_toggle = False
        for cat in ["Auditory & Language", "Visual Processing", "Executive & Motor", "Attention & Spatial", "Emotion & Decision"]:
            s = scores_map.get(cat, 0)
            g = grade(s)
            desc = grade_desc_map.get(cat, '')
            bg = DARK_BG if row_toggle else BLACK
            pdf.set_fill_color(*bg)
            pdf.set_text_color(*WHITE)
            pdf.cell(col_widths[0], 7, cat, border=0, fill=True)
            pdf.cell(col_widths[1], 7, f'{s}/100', border=0, fill=True)
            gc2 = grade_color(g)
            pdf.set_text_color(*gc2)
            pdf.cell(col_widths[2], 7, g, border=0, fill=True)
            pdf.set_text_color(*MUTED)
            pdf.cell(col_widths[3], 7, desc, border=0, fill=True)
            pdf.ln()
            row_toggle = not row_toggle

        # === Engagement Bars ===
        pdf.section_title('Engagement Profile')
        pdf.set_font('Helvetica', '', 9)
        bar_max_w = 100
        for cat in ["Auditory & Language", "Visual Processing", "Executive & Motor", "Attention & Spatial", "Emotion & Decision"]:
            s = scores_map.get(cat, 0)
            g = grade(s)
            gc2 = grade_color(g)
            if pdf.get_y() > 270:
                pdf.add_page()
            pdf.set_text_color(*WHITE)
            pdf.cell(55, 6, cat, border=0)
            # Draw bar background
            bar_x = pdf.get_x()
            bar_y = pdf.get_y() + 1
            pdf.set_fill_color(*CARD_BG)
            pdf.rect(bar_x, bar_y, bar_max_w, 4, 'F')
            # Draw filled portion
            fill_w = max(1, bar_max_w * s / 100)
            pdf.set_fill_color(*gc2)
            pdf.rect(bar_x, bar_y, fill_w, 4, 'F')
            pdf.set_x(bar_x + bar_max_w + 3)
            pdf.set_text_color(*gc2)
            pdf.cell(30, 6, f'{s}/100 ({g})', border=0)
            pdf.ln()

        # === Predictive Insights ===
        # Parse from markdown
        pred_metrics = []
        for metric_name in ["Watch-Through Rate", "24hr Ad Recall", "Purchase/Action Intent",
                            "Virality / Shareability", "Cognitive Load", "Optimal Content Length", "Best Content Fit"]:
            for line in analysis.split('\n'):
                if metric_name in line and '|' in line:
                    parts = [p.strip().strip('*') for p in line.split('|')]
                    parts = [p for p in parts if p]
                    if len(parts) >= 3:
                        pred_metrics.append((parts[0].strip('* '), parts[1], parts[2] if len(parts) > 2 else ''))
                    break

        if pred_metrics:
            pdf.section_title('Predictive Insights')
            col_w_pred = [55, 75, 60]
            pdf.set_fill_color(*CARD_BG)
            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_text_color(*WHITE)
            pdf.cell(col_w_pred[0], 7, 'Metric', fill=True)
            pdf.cell(col_w_pred[1], 7, 'Prediction', fill=True)
            pdf.cell(col_w_pred[2], 7, 'Confidence', fill=True)
            pdf.ln()
            pdf.set_font('Helvetica', '', 9)
            row_toggle = False
            for name, pred, conf in pred_metrics:
                bg = DARK_BG if row_toggle else BLACK
                pdf.set_fill_color(*bg)
                pdf.set_text_color(*WHITE)
                safe_name = name.encode('ascii', 'ignore').decode('ascii')
                safe_pred = pred.encode('ascii', 'ignore').decode('ascii')
                safe_conf = conf.encode('ascii', 'ignore').decode('ascii')
                pdf.cell(col_w_pred[0], 7, safe_name, fill=True)
                pdf.cell(col_w_pred[1], 7, safe_pred, fill=True)
                pdf.set_text_color(*MUTED)
                pdf.cell(col_w_pred[2], 7, safe_conf, fill=True)
                pdf.ln()
                row_toggle = not row_toggle

        # === Key Findings + AI Strategic Action Plan ===
        # Extract sections from the markdown
        for section_marker, section_title_str in [
            ("### Key Findings", "Key Findings"),
            ("AI Strategic Action Plan", "AI Strategic Action Plan"),
            ("### Bottom Line", "Bottom Line"),
        ]:
            idx = analysis.find(section_marker)
            if idx == -1:
                continue
            # Find end of section (next ### or end of string)
            rest = analysis[idx:]
            lines = rest.split('\n')
            section_lines = []
            for i, line in enumerate(lines):
                if i == 0:
                    continue  # skip the header line itself
                if line.strip().startswith('### ') or line.strip().startswith('---'):
                    break
                section_lines.append(line)
            body = '\n'.join(section_lines).strip()
            if not body:
                continue

            pdf.section_title(section_title_str)
            # Clean markdown formatting
            clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', body)  # bold
            clean = re.sub(r'[#]', '', clean)
            clean = clean.encode('ascii', 'ignore').decode('ascii')
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(*WHITE)
            pdf.multi_cell(0, 5, clean)

        pdf_path = run_dir / "report.pdf"
        pdf.output(str(pdf_path))
        pdf_path_str = str(pdf_path.absolute())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to generate PDF: {e}")
        
    return run_id, pdf_path_str

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

def create_pdf_button_html(pdf_path_str):
    if not pdf_path_str: return gr.update(visible=False)
    # Passed directly to gr.DownloadButton which natively resolves the UUID caching bug 
    return gr.update(value=pdf_path_str, visible=True)


def view_run(choice_str):
    """Load a specific run's plot and analysis."""
    run_id = _extract_run_id(choice_str)
    if not run_id:
        return None, "*Select a run from the dropdown above.*", gr.update(visible=False)
    run_dir = RUNS_DIR / run_id
    meta_path = run_dir / "meta.json"
    plot_path = run_dir / "brain_map.png"
    if not meta_path.exists():
        return None, f"Run `{run_id}` not found.", gr.update(visible=False)
    with open(meta_path) as f:
        meta = json.load(f)
    img = str(plot_path) if plot_path.exists() else None
    pdf_path = run_dir / "report.pdf"
    pdf_str = str(pdf_path.absolute()) if pdf_path.exists() else None
    return img, meta.get("analysis", "No analysis available."), create_pdf_button_html(pdf_str)

def delete_run(choice_str):
    """Delete a run from history."""
    run_id = _extract_run_id(choice_str)
    if not run_id:
        return gr.update(choices=get_history_choices(), value=None), None, "*No run selected.*", gr.update(visible=False)
    import shutil
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return gr.update(choices=get_history_choices(), value=None), None, "*Run deleted.*", gr.update(visible=False)

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

import torch

def get_model():
    global model
    with model_lock:
        if model is None:
            # We ONLY use CUDA if it exists (i.e., on Modal). 
            # Macs (MPS) still crash on these attention layers, so they must use CPU.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading TRIBE v2 model... computing device → {device}")
            
            # Pass device to the HuggingFace model if supported natively
            try:
                model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache", device=device)
            except TypeError:
                # Fallback if TribeModel does not accept device via kwargs
                model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
            
            # Explicitly force extractors to the selected device
            for attr in ["neuro", "audio_feature", "video_feature", "image_feature", "text_feature"]:
                extractor = getattr(model.data, attr, None)
                if extractor is not None:
                    if hasattr(extractor, "device"):
                        extractor.device = device
                    if hasattr(extractor, "image") and hasattr(extractor.image, "device"):
                        extractor.image.device = device
            print(f"Model loaded successfully on {device}.")
    return model


def analyze_brain_regions(preds, stimulus_description=""):
    """Full brain analysis: regional scores, temporal dynamics, hemispheric laterality, predictive metrics."""
    
    # Use absolute values — brain model outputs signed activations where
    # negative values represent suppression. Both positive and negative
    # responses indicate neural engagement, so we measure total energy.
    avg_activation = np.mean(np.abs(preds), axis=0)
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
    
    # --- AI STRATEGIC ACTION PLAN ---
    report += "\n\n### 💡 AI Strategic Action Plan\n\n"
    report += "Based on your neural encoding profile, here is exactly how to optimize this content for higher engagement:\n\n"
    recs = []
    
    if emotion_score < 60:
        recs.append("**Elevate Emotional Resonance:** Your emotional activation is lagging. Introduce a human element earlier (faces, eyes), raise the stakes with a clear conflict or problem-statement, and consider using a subtle background music track that swells during key points.")
    if visual_score < 60:
        recs.append("**Stimulate the Visual Cortex:** The brain is not actively tracking your visuals. Break visual monotony by changing camera angles every 3-5 seconds, adding dynamic text overlays (B-roll, captions), or ensuring high color contrast in your framing.")
    if auditory_score < 60:
        recs.append("**Enhance Auditory Processing:** The auditory cortex is under-stimulated. Use vocal intonation and pacing shifts to emphasize important words. Eliminate background noise and consider subtle sound effects (whooshes, pops) during visual transitions.")
    if attention_score < 60:
        recs.append("**Command Focused Attention:** Viewers are passively watching. Force active attention by breaking the fourth wall (direct eye contact), asking a rhetorical question, or using a 'pattern interrupt' (a sudden visual/audio shift) in the first 3 seconds.")
    if executive_score < 60:
        recs.append("**Trigger Deep Thinking:** The content is being consumed too passively. Challenge the viewer's assumptions, present a surprising statistic, or frame your message as a 'secret' or 'counter-intuitive truth' to force the executive network to engage.")
        
    if watch_through < 50:
        recs.append(f"**Combat Attention Decay:** Our models predict high drop-off. Cut this content down to ~{optimal_length}s. Front-load your absolute strongest hook into the first 2 seconds, and ruthlessly cut any 'fluff' or slow introductions.")
    if purchase_intent < 40:
        recs.append("**Drive Action Intent:** To push the brain from 'watching' to 'acting', you must explicitly state what the user should do next. Pair a clear, urgent Call-To-Action (CTA) with a visual of the desired outcome.")
    if virality > 75:
        recs.append("**Capitalize on Virality:** This content already exhibits high organic shareability signatures. Do not put this behind a paywall—use it as top-of-funnel content optimized for TikTok, Reels, and Shorts algorithms.")
    elif virality < 40:
        recs.append("**Engineer Shareability:** To make this spread organically, it needs a higher 'Social Currency' trigger. Frame the core message as something the viewer would look smart for sharing with a friend.")

    if not recs:
        recs.append("**Scale and Distribute:** Your neural engagement is exceptionally well-balanced. Do not change the core creative. Lock this in for A/B testing on paid ad campaigns to find the best audience match.")
    
    for i, rec in enumerate(recs, 1):
        report += f"{i}. {rec}\n\n"
    
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


# Each view angle reveals different brain systems — give unique, meaningful labels
VIEW_LABELS = {
    "left":     ("Left Hemisphere",  "Language, speech, and analytical processing"),
    "right":    ("Right Hemisphere", "Creativity, emotion, and spatial awareness"),
    "dorsal":   ("Top-Down View",    "Motor planning, attention, and focus"),
    "anterior": ("Front View",       "Decision-making, social cognition, and impulse control"),
    "posterior":("Rear View",        "Visual processing and sensory integration"),
    "ventral":  ("Bottom View",      "Object recognition and face processing"),
}


def generate_plot_and_analysis(df, progress, stimulus_type="Text", stimulus_desc="", media_path=None):
    """Generate vertical brain plot AND AI analysis, then persist the run."""
    import matplotlib.pyplot as plt
    from tribev2.plotting.utils import robust_normalize
    
    progress((0.4, 1.0), desc="Extracting Deep Multimodal AI Features (Heaviest Step)...")
    m = get_model()
    preds, segments = m.predict(events=df, gradio_progress=progress)
    
    progress((0.75, 1.0), desc="Rendering 3D Brain Mesh (Vertical Layout)...")
    
    # Calculate the global average activation across the entire stimulus
    avg_pred = np.mean(preds, axis=0)
    views_seq = ["left", "right", "dorsal", "anterior"]
    n_to_plot = len(views_seq)
    
    # Normalize data globally for consistent colorbar
    norm_pred = robust_normalize(avg_pred, percentile=95)
    
    import textwrap
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(10, 3.5 * n_to_plot))
    fig.patch.set_facecolor('#09090b')
    gs = gridspec.GridSpec(n_to_plot, 2, width_ratios=[1, 1.2], figure=fig)
    
    sm = None
    for i in range(n_to_plot):
        view_name = views_seq[i]
        view_title, view_desc = VIEW_LABELS.get(view_name, ("Brain View", "Neural activation"))
        raw_act = float(np.mean(np.abs(avg_pred)))
        
        # Wrap the description text so it doesn't bleed off the edge of the figure
        wrapped_desc = textwrap.fill(view_desc, width=45)
        
        # Brain plot on the left
        ax_brain = fig.add_subplot(gs[i, 0], projection="3d")
        ax_brain.set_facecolor('#09090b')
        
        sm = plotter.plot_surf(
            norm_pred,
            views=view_name,
            axes=[ax_brain], 
            colorbar=False,
            cmap="hot"
        )
        
        # Text on the right
        ax_text = fig.add_subplot(gs[i, 1])
        ax_text.axis("off")
        ax_text.set_facecolor('#09090b')
        label_text = f"{view_title}\n\n{wrapped_desc}\n\nActivation: {raw_act * 100:.1f}%"
        ax_text.text(0.1, 0.5, label_text, fontsize=13, color="#e4e4e7", va="center", ha="left", wrap=True)

    # Add a global colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.05, 0.25, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Relative Activation Intensity", color="#a1a1aa", fontsize=10)
    cbar.ax.tick_params(colors="#a1a1aa", labelsize=9)
    # remove ticks
    cbar.set_ticks([])
    
    fig.subplots_adjust(top=0.95, bottom=0.12, wspace=0.1, hspace=0.1)

    progress((0.9, 1.0), desc="Analyzing Brain Activation Patterns...")
    interpretation, region_data = analyze_brain_regions(preds, stimulus_desc)
    
    progress((0.95, 1.0), desc="Saving run to history...")
    pdf_path = None
    try:
        _, pdf_path = save_run(stimulus_type, stimulus_desc, fig, interpretation, region_data, media_path)
    except Exception as e:
        print(f"[Run History] Failed to save: {e}")
    
    progress((1.0, 1.0), desc="Complete")
    return fig, interpretation, create_pdf_button_html(str(pdf_path) if pdf_path else None)


@spaces.GPU(duration=180)
def process_text(text, progress=gr.Progress()):
    if not text.strip():
        import gradio as gr
        return None, "", gr.update(visible=False, value=None)
        
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
        
@spaces.GPU(duration=180)
def process_audio(audio_path, progress=gr.Progress()):
    if not audio_path:
        import gradio as gr
        return None, "", gr.update(visible=False, value=None)
    progress((0.0, 1.0), desc="Loading Audio & Whisper Extractors...")
    progress((0.15, 1.0), desc="Extracting Audio & Phonetics Features...")
    df = get_model().get_events_dataframe(audio_path=audio_path)
    return generate_plot_and_analysis(df, progress, stimulus_type="Audio", stimulus_desc="Audio recording")

@spaces.GPU(duration=180)
def process_video(video_path, progress=gr.Progress()):
    if not video_path:
        import gradio as gr
        return None, "", gr.update(visible=False, value=None)
    progress((0.0, 1.0), desc="Trimming video and Splitting Frames...")
    
    # Build trimmed path safely — do NOT use str.replace() as it corrupts
    # paths when the extension appears in parent directory names.
    base, ext = os.path.splitext(video_path)
    trimmed_path = f"{base}_trimmed_6s{ext}"
    
    # Enforce 6.0 seconds with clean re-encoding. 3.0s starved the sequence batcher.
    # Re-encode (libx264) to guarantee clean keyframes for the vision encoder.
    os.system(f'ffmpeg -y -i "{video_path}" -t 6 -c:v libx264 -preset ultrafast -c:a aac "{trimmed_path}" -loglevel quiet')
    
    progress((0.15, 1.0), desc="Extracting Video Motion & Semantics via V-JEPA2...")
    df = get_model().get_events_dataframe(video_path=trimmed_path)
    return generate_plot_and_analysis(df, progress, stimulus_type="Video", stimulus_desc="Video clip (first 6 seconds)", media_path=video_path)


# --- Custom Theme & Modern Black Dashboard Styling ---
custom_theme = gr.themes.Base(
    primary_hue="zinc",
    secondary_hue="stone",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    background_fill_primary="#000000",
    background_fill_secondary="#09090b",
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
            TRIBE v2 Brain Encoding (Live v2)
        </h1>
        <p style="font-size: 1.05rem; color: #71717a; max-width: 700px; line-height: 1.5;">
            Upload text, audio, or video to predict human fMRI brain activity. The AI will map cortical activation patterns and provide a detailed cognitive analysis.
        </p>
    </div>
    """)
    
    with gr.Accordion("📚 How to Read Your Analysis & Grades (Click to expand)", open=False):
        gr.Markdown("""
### 🎯 What Your Score Means
The **Overall Neural Engagement Score (0-100)** displays how strongly your content grips the human brain. A higher score means audiences are organically locked in, feeling emotion, and actively paying attention.

### 📊 The 5 Grading Categories
We grade your content across distinct cognitive systems. To get better engagement, look at your weakest grade and adjust your content:
* **🗣️ Auditory & Language:** Processing speech and words. *To improve:* Speak clearly, use strong vocabulary, or add sound effects.
* **👀 Visual Processing:** Processing faces, motion, and color. *To improve:* Add more dynamic fast edits, bold text overlays, or expressive facial close-ups.
* **⏱️ Attention & Spatial:** Focusing on the screen without losing interest. *To improve:* Add cuts or slight camera movements every 3-5 seconds.
* **🧠 Executive & Deep Thinking:** Processing complex ideas or logic. *To improve:* Ask questions or introduce a "hook", mystery, or puzzle early.
* **❤️ Emotion & Decision:** Feeling empathy, excitement, or trust. *To improve:* Tell a personal story, use strong emotional delivery, and build a cohesive narrative.

### 🧠 How to Read the 3D Brain Maps
The 3D glowing brains show *exactly* when and where the human brain activated during your video, audio, or text. 
Above each brain, our AI notes the precise cognitive system being engaged at that second (e.g., "👀 Processing Visuals"), alongside the raw Activation `%`. If activation drops in the middle of your video, you are likely losing the viewer's attention.
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
                    audio_upload = gr.File(label="Upload Audio File", file_types=["audio"])
                    audio_preview = gr.Audio(label="Preview", interactive=False, visible=False)
                    audio_in = gr.State()
                    audio_btn = gr.Button("Execute Brain Mapping", variant="primary", size="lg")
                    
                    def process_audio_upload(f):
                        if not f: return None, gr.update(visible=False)
                        return f, gr.update(value=f, visible=True)
                    audio_upload.change(process_audio_upload, inputs=audio_upload, outputs=[audio_in, audio_preview])
                    
                with gr.Tab("Video Inference"):
                    gr.Markdown("Upload standard video formats. Auto-trimmed to 6 seconds.")
                    video_upload = gr.File(label="Upload Video File", file_types=["video"])
                    video_preview = gr.Video(label="Preview", interactive=False, visible=False)
                    video_in = gr.State()
                    video_btn = gr.Button("Execute Brain Mapping", variant="primary", size="lg")
                    
                    def process_video_upload(f):
                        if not f: return None, gr.update(visible=False)
                        return f, gr.update(value=f, visible=True)
                    video_upload.change(process_video_upload, inputs=video_upload, outputs=[video_in, video_preview])
                    
        with gr.Column(scale=5):
            with gr.Row():
                gr.Markdown("### Predicted Cortical Activation")
                new_run_btn = gr.Button("New +", variant="secondary", size="sm", scale=0, min_width=80)
            out_plot = gr.Plot(label="", show_label=False)

    gr.Markdown("---")
    with gr.Row():
        gr.Markdown("### Content Engagement Scorecard")
    out_analysis = gr.Markdown(value="*Run a brain mapping to see an engagement scorecard with grades, scores, and actionable recommendations.*")
    out_pdf = gr.DownloadButton("⬇ Download PDF Report", visible=False, variant="secondary", size="lg")

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
    history_pdf = gr.DownloadButton("⬇ Download PDF Report", visible=False, variant="secondary", size="lg")

    # Helper to refresh dropdown after a run completes
    def refresh_history():
        return gr.update(choices=get_history_choices())

    # New+ button — clears everything for a fresh run
    def reset_for_new_run():
        return (
            "",           # text_in
            None,         # audio_upload
            None,         # audio_in
            gr.update(visible=False), # audio_preview
            None,         # video_upload
            None,         # video_in
            gr.update(visible=False), # video_preview
            None,         # out_plot
            "*Run a brain mapping to see an engagement scorecard with grades, scores, and actionable recommendations.*",  # out_analysis
            gr.update(visible=False), # out_pdf
        )

    new_run_btn.click(
        fn=reset_for_new_run,
        outputs=[text_in, audio_upload, audio_in, audio_preview, video_upload, video_in, video_preview, out_plot, out_analysis, out_pdf]
    )

    # Wire up brain mapping — chain dropdown refresh after completion
    text_btn.click(fn=process_text, inputs=text_in, outputs=[out_plot, out_analysis, out_pdf]).then(
        fn=refresh_history, outputs=history_dropdown
    )
    audio_btn.click(fn=process_audio, inputs=audio_in, outputs=[out_plot, out_analysis, out_pdf]).then(
        fn=refresh_history, outputs=history_dropdown
    )
    video_btn.click(fn=process_video, inputs=video_in, outputs=[out_plot, out_analysis, out_pdf]).then(
        fn=refresh_history, outputs=history_dropdown
    )

    # History event handlers
    history_dropdown.change(
        fn=view_run,
        inputs=history_dropdown,
        outputs=[history_image, history_analysis, history_pdf]
    )
    refresh_btn.click(
        fn=refresh_history,
        outputs=history_dropdown
    )
    delete_btn.click(
        fn=delete_run,
        inputs=history_dropdown,
        outputs=[history_dropdown, history_image, history_analysis, history_pdf]
    )

    # Force-load history on page open
    app.load(fn=refresh_history, outputs=history_dropdown)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[str(RUNS_DIR.absolute())])
