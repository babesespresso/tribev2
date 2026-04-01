import torch
import warnings
warnings.filterwarnings('ignore')
from tribev2.models.tribe import TribeModel

# Create a dummy 1-second white video
import os
os.system('ffmpeg -y -f lavfi -i color=c=white:s=320x240:d=1 -c:v libx264 -pix_fmt yuv420p dummy.mp4 -loglevel quiet')

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
video_ext = getattr(model.data, "video_feature", None)
if video_ext:
    video_ext.device = "mps"
    print("Testing Video on MPS...")
    try:
        df = model.get_events_dataframe(video_path="dummy.mp4")
        print("Success!")
    except Exception as e:
        print(f"Failed cleanly: {e}")
