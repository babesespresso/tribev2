#!/usr/bin/env python3
"""
Deploy TRIBE v2 to Hugging Face Spaces.

Usage:
    python deploy_to_hf.py

Requirements:
    - A HuggingFace token with WRITE permissions
    - Set it via: export HF_TOKEN=hf_xxxxx
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

def main():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: Set HF_TOKEN env var with a WRITE-enabled token.")
        print("Go to: https://huggingface.co/settings/tokens")
        print("Create a new token with 'Write' permission.")
        sys.exit(1)

    from huggingface_hub import HfApi, upload_folder

    api = HfApi(token=token)
    whoami = api.whoami()
    username = whoami["name"]
    space_id = f"{username}/tribev2"

    print(f"Deploying to: https://huggingface.co/spaces/{space_id}")

    # 1. Create the Space
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True,
        )
        print(f"✅ Space created: {space_id}")
    except Exception as e:
        print(f"❌ Failed to create Space: {e}")
        sys.exit(1)

    # 2. Set HF_TOKEN as a secret in the Space
    try:
        api.add_space_secret(repo_id=space_id, key="HF_TOKEN", value=token)
        print("✅ HF_TOKEN secret added")
    except Exception as e:
        print(f"⚠️  Could not set secret (may need manual setup): {e}")

    # 3. Create a temporary staging directory with Space files
    project_root = Path(__file__).parent.resolve()
    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)

        # Copy essential files
        files_to_copy = [
            "app.py",
            "requirements.txt",
            "pyproject.toml",
            "LICENSE",
        ]
        for f in files_to_copy:
            src = project_root / f
            if src.exists():
                shutil.copy2(src, staging / f)
                print(f"  📄 {f}")

        # Copy the tribev2 package
        src_pkg = project_root / "tribev2"
        if src_pkg.exists():
            shutil.copytree(src_pkg, staging / "tribev2", 
                          ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            print("  📦 tribev2/")

        # Create Space README with frontmatter
        readme_content = """---
title: TRIBE v2 Brain Encoding
emoji: 🧠
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "6.10.0"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Predict human brain activity from text, audio, or video
---

# 🧠 TRIBE v2 — Content Engagement Scorecard

**Powered by Meta's TRIBE v2 Brain Encoding Model**

Upload text, audio, or video to predict human fMRI brain activity. The AI maps cortical activation patterns and generates a comprehensive Content Engagement Scorecard.

## Features

- **Neural Engagement Scoring** — Overall score (0-100) with letter grades (A+ to F) across 5 cognitive systems
- **Engagement Timeline** — Second-by-second engagement visualization
- **Brain Laterality** — Left vs Right hemisphere analysis
- **Predictive Insights** — Watch-through rate, 24hr recall, purchase intent, virality, cognitive load
- **Run History** — All completed brain mappings saved for comparison
- **Annotated Brain Maps** — Each timestep labeled with dominant cognitive process

## How It Works

TRIBE v2 combines three state-of-the-art AI models:
- **LLaMA 3.2** for text/language understanding
- **V-JEPA2** for video/visual processing  
- **Wav2Vec-BERT** for audio/speech analysis

These are fed into a Transformer that maps multimodal representations onto the cortical surface, predicting how the human brain would respond to the content.

## Built by Multitude Media
"""
        (staging / "README.md").write_text(readme_content)
        print("  📄 README.md (Space config)")

        # 4. Upload to HF
        print("\n🚀 Uploading to Hugging Face Spaces...")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=space_id,
            repo_type="space",
        )
        print(f"\n✅ Deployed! Your Space is live at:")
        print(f"   https://huggingface.co/spaces/{space_id}")
        print(f"\n⏳ It will take a few minutes to build and start.")
        print(f"   The Space will auto-install dependencies and launch.")


if __name__ == "__main__":
    main()
