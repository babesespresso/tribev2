import gradio as gr
import os
import torch
from tribev2 import TribeModel
from tribev2.plotting import PlotBrainNilearn as PlotBrain
import tempfile
import threading
import matplotlib
import time

matplotlib.use('Agg')

model = None
plotter = PlotBrain(mesh="fsaverage5")
model_lock = threading.Lock()

def get_model():
    global model
    with model_lock:
        if model is None:
            print("Loading TRIBE v2 model... this may take a minute.")
            model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
            if not torch.cuda.is_available():
                for attr in ["neuro", "text_feature", "audio_feature", "video_feature", "image_feature"]:
                    extractor = getattr(model.data, attr, None)
                    if extractor is not None:
                        if hasattr(extractor, "device"):
                            extractor.device = "cpu"
                        if hasattr(extractor, "image") and hasattr(extractor.image, "device"):
                            extractor.image.device = "cpu"
            print("Model loaded successfully.")
    return model

def generate_plot(df, progress):
    progress((0.4, 1.0), desc="Extracting Deep Multimodal AI Features (Heaviest Step)...")
    m = get_model()
    # Smooth progress mapping passed into the core engine
    preds, segments = m.predict(events=df, gradio_progress=progress)
    
    progress((0.8, 1.0), desc="Rendering Topographical 3D Mesh Output...")
    
    # By default, predictions is 1 second = 1 frame. Let's cap at 5 seconds for visual layout speeds.
    n_to_plot = min(len(preds), 4)
    sliced_preds = preds[:n_to_plot]
    
    # Create the figures
    fig = plotter.plot_timesteps(sliced_preds, show_stimuli=False)
    
    progress((1.0, 1.0), desc="Complete")
    return fig

def process_text(text, progress=gr.Progress()):
    if not text.strip():
        return None
        
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
        return generate_plot(df, progress)
    finally:
        os.unlink(tmp_path)
        
def process_audio(audio_path, progress=gr.Progress()):
    if not audio_path:
        return None
    progress((0.0, 1.0), desc="Loading Audio & Whisper Extractors...")
    progress((0.15, 1.0), desc="Extracting Audio & Phonetics Features...")
    df = get_model().get_events_dataframe(audio_path=audio_path)
    return generate_plot(df, progress)

def process_video(video_path, progress=gr.Progress()):
    if not video_path:
        return None
    progress((0.0, 1.0), desc="Trimming video and Splitting Frames...")
    
    # Auto-trim the video to the first 5 seconds to guarantee fast local execution
    ext = os.path.splitext(video_path)[1]
    trimmed_path = video_path.replace(ext, f"_trimmed{ext}")
    os.system(f'ffmpeg -y -i "{video_path}" -t 5 -c copy "{trimmed_path}" -loglevel quiet')
    
    progress((0.15, 1.0), desc="Extracting Video Motion & Semantics via V-JEPA2...")
    df = get_model().get_events_dataframe(video_path=trimmed_path)
    return generate_plot(df, progress)

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
            A Multimodal Foundation Model for In-Silico Neuroscience. Upload naturalistic stimuli to structurally predict human fMRI activity directly mapped onto the cortical mesh.
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
                    gr.Markdown("Upload standard video formats. Note: Length is truncated to 5 seconds by default to bypass CPU processing bottlenecks.")
                    video_in = gr.Video(label="Video Stimulus", sources=["upload"])
                    video_btn = gr.Button("Execute Brain Mapping", variant="primary", size="lg")
                    
        with gr.Column(scale=5):
            gr.Markdown("### Predicted Cortical Activation")
            out_plot = gr.Plot(label="", show_label=False)

    # Wire up the events
    text_btn.click(fn=process_text, inputs=text_in, outputs=out_plot)
    audio_btn.click(fn=process_audio, inputs=audio_in, outputs=out_plot)
    video_btn.click(fn=process_video, inputs=video_in, outputs=out_plot)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
