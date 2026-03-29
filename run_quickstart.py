from tribev2 import TribeModel
import torch

if __name__ == "__main__":
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

    if not torch.cuda.is_available():
        for attr in ["neuro", "text_feature", "audio_feature", "video_feature", "image_feature"]:
            extractor = getattr(model.data, attr, None)
            if extractor is not None and hasattr(extractor, "device"):
                extractor.device = "cpu"

    df = model.get_events_dataframe(text_path="dummy.txt")
    preds, segments = model.predict(events=df)
    print(preds.shape)
