import os
import sys
import tempfile
import urllib.request
from typing import Dict

import modal

# ── Modal Configuration ──────────────────────────────────────────────
# We define the Docker container image, installing all heavy ML dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch", 
        "torchvision", 
        "torchaudio", 
        index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "matplotlib",
        "pandas",
        "fpdf2",
        "huggingface_hub",
        "slack-sdk",
        "scikit-learn",
        "neuralset==0.0.2",
        "neuraltrain==0.0.2",
        "x_transformers==1.27.20",
        "einops",
        "moviepy>=2.2.1",
        "spaces",
        "gtts",
        "langdetect",
        "spacy",
        "soundfile",
        "Levenshtein",
        "nibabel>=5.4",
        "nilearn>=0.13",
        "colorcet",
        "seaborn",
        "gradio>=5.0",
        "requests",
        "tqdm",
        "transformers",
        "julius"
    )
    .add_local_dir(
        ".", 
        remote_path="/root/tribev2", 
        ignore=[".git", "venv", "__pycache__", "runs", ".env"]
    )
)

app = modal.App("tribe-v2-backend")

# ── Serverless ML Inference Function ─────────────────────────────────
@app.function(
    image=image,
    gpu="a10g",
    timeout=1800,                # 30 min — cold starts download ~8GB of models
    scaledown_window=300,        # Keep container warm 5 min between requests
    secrets=[
        modal.Secret.from_name("slack-bot-secret"),
        modal.Secret.from_name("huggingface-token")
    ]
)
def run_tribev2_gpu(payload: Dict):
    """
    Downloads the video from Slack, runs the heavy PyTorch ML models,
    generates the brain scorecard, and uploads the PDF back to Slack.
    """
    import logging
    from slack_sdk import WebClient
    
    # Insert mounted directory to PATH so we can import app.py
    sys.path.insert(0, "/root/tribev2")
    os.chdir("/root/tribev2")
    
    from app import get_model, generate_plot_and_analysis, RUNS_DIR
    import matplotlib.pyplot as plt
    import re
    from pathlib import Path

    # 1. Setup Logging and Slack Client
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("modal-tribev2")
    logger.info(f"Started GPU Inference for payload: {payload['filename']}")
    
    slack_token = os.environ.get("SLACK_BOT_TOKEN")
    if not slack_token:
        logger.error("SLACK_BOT_TOKEN missing from Modal Secrets!")
        return

    client = WebClient(token=slack_token)
    
    channel = payload["channel"]
    thread_ts = payload["thread_ts"]
    proc_ts = payload.get("proc_ts")
    filename = payload["filename"]
    download_url = payload["download_url"]

    # Simple logger progress tracker to replace Gradio's
    def progress_tracker(progress_tuple=None, desc=""):
        if desc: logger.info(f"Progress [{filename}]: {desc}")

    try:
        # 2. Update Slack Thread Message
        if proc_ts:
            client.chat_update(
                channel=channel,
                ts=proc_ts,
                text=(f"🧠 *TRIBE v2 Brain Encoding* — `{filename}`\n"
                      f"⚡ _GPU Booted. Extracting Neural Features..._")
            )

        # 3. Download the Video into Modal Container Temp space
        ext = os.path.splitext(filename)[1].lower()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_vid:
            req = urllib.request.Request(download_url, headers={"Authorization": f"Bearer {slack_token}"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                tmp_vid.write(resp.read())
            video_path = tmp_vid.name
            
        logger.info(f"Downloaded video to {video_path}. Trimming...")

        # 4. Trim Video to 6 seconds (mirrors app.py process_video)
        trimmed_path = video_path.replace(ext, f"_trimmed{ext}")
        os.system(
            f'ffmpeg -y -i "{video_path}" -t 6 '
            f'-c:v libx264 -preset ultrafast -c:a aac '
            f'"{trimmed_path}" -loglevel quiet'
        )
        actual_path = trimmed_path if os.path.exists(trimmed_path) else video_path

        # 5. ML INFERENCE 
        logger.info("Extracting events dataframe (V-JEPA)...")
        df = get_model().get_events_dataframe(video_path=actual_path)
        
        fig, interpretation, _ = generate_plot_and_analysis(
            df,
            progress_tracker,
            stimulus_type="Video",
            stimulus_desc=f"{filename} (Slack Upload)",
            media_path=video_path,
        )
        plt.close(fig)

        # 6. Locate generated PDF
        pdf_path = None
        runs_dir = Path("/root/tribev2/runs")
        if runs_dir.exists():
            logger.info(f"Runs directory contents: {[d.name for d in runs_dir.iterdir()]}")
            for run_dir in sorted(runs_dir.iterdir(), reverse=True):
                candidate = run_dir / "report.pdf"
                if run_dir.is_dir() and candidate.exists():
                    pdf_path = str(candidate.absolute())
                    logger.info(f"Found PDF at: {pdf_path}")
                    break
        else:
            logger.error("Runs directory does not exist!")
                
        if not pdf_path:
            raise Exception("PDF was not created in the runs directory!")

        # Extra: parse one paragraph summary
        score_match = re.search(r"Overall Neural Engagement:\s*(\d+)/100\s*\(([^)]+)\)", interpretation)
        if score_match:
            summary = f"📊 *Overall Neural Engagement: {score_match.group(1)}/100 ({score_match.group(2)})*"
        else:
            summary = "📊 *Analysis complete!*"

        # 7. Upload PDF back to Slack
        logger.info("Uploading PDF scorecard back to Slack...")
        client.files_upload_v2(
            channel=channel,
            thread_ts=thread_ts,
            file=pdf_path,
            filename=f"Neuro_Scorecard_{filename}.pdf",
            title=f"Scorecard: {filename}",
            initial_comment=(
                f"✅ *GPU Analysis Complete!*\n\n{summary}\n\n"
                f"_Download the PDF to view the second-by-second brain analytics._"
            ),
        )

        # Cleanup UI
        if proc_ts:
            try:
                client.chat_update(channel=channel, ts=proc_ts, text=f"🧠 *TRIBE v2 Brain Encoding* — `{filename}` ✅ Complete!")
            except Exception:
                pass
        try:
            client.reactions_remove(channel=channel, timestamp=thread_ts, name="hourglass_flowing_sand")
            client.reactions_add(channel=channel, timestamp=thread_ts, name="white_check_mark")
        except Exception:
            pass  # reactions:write scope not granted — non-critical

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        if proc_ts:
            try:
                client.chat_update(
                    channel=channel, ts=proc_ts,
                    text=f"❌ *GPU Analysis Failed* for `{filename}`\n```{str(e)}```"
                )
            except Exception:
                pass
        
        try:
            client.reactions_remove(channel=channel, timestamp=thread_ts, name="hourglass_flowing_sand")
            client.reactions_add(channel=channel, timestamp=thread_ts, name="warning")
        except Exception as reaction_err:
            logger.warning(f"Failed to update reactions (likely missing reactions:write scope): {reaction_err}")


# ── Webhook Endpoint (Instantly returns 200 OK while processing) ─────
@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
async def slack_webhook(payload: Dict):
    """
    Render listener hits this URL instantly. We respond immediately so Render
    doesn't time out, and we spawn the heavy GPU task in the background.
    """
    print(f"Received webhook for file: {payload.get('filename')}")
    run_tribev2_gpu.spawn(payload) # Non-blocking kick-off
    return {"status": "ok", "message": "GPU Task Spawned"}
