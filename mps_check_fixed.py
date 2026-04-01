import torch
import warnings
warnings.filterwarnings('ignore')
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
video_ext = getattr(model.data, "video_feature", None)
if video_ext:
    video_ext.device = "mps"
    print("Testing Video on MPS...")
    try:
        df = model.get_events_dataframe(video_path="dummy.mp4")
        print("Success! Inference on MPS worked.")
    except Exception as e:
        print(f"Failed cleanly: {e}")
