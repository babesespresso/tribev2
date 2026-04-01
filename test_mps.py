import torch
import warnings
warnings.filterwarnings('ignore')

from tribev2.models.tribe import TribeModel

print("Loading model...")
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

print("Testing Video Extractor on MPS...")
video_ext = getattr(model.data, "video_feature", None)
if video_ext:
    video_ext.device = "mps"
    print("Video extractor device set to MPS.")

# Get a dummy video
try:
    df = model.get_events_dataframe(video_path="venv/lib/python3.13/site-packages/gradio/templates/frontend/assets/logo.svg") # just anything, wait, get_events expects a real video. Let me find a video.
except Exception as e:
    print(f"Failed dummy extraction setup {e}")
