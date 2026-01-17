# i3d_feature_extractor.py (Updated)

import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
import cv2

class I3DFeatureExtractor(nn.Module):
    def __init__(self, model_path, device):
        super(I3DFeatureExtractor, self).__init__()
        from pytorch_i3d import InceptionI3d
        self.device = device
        self.model = InceptionI3d(400, in_channels=3)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.replace_logits(1024)
        self.model.to(device).eval()

    def forward(self, clip_tensor):
        with torch.no_grad():
            features = self.model.extract_features(clip_tensor)
        features = features.squeeze(-1).squeeze(-1).squeeze(0).permute(1, 0)  # (32, 1024)
        return features


def load_and_preprocess_video(video_path):
    transform = Compose([
        ToPILImage(),
        Resize((256, 340)),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(frame))
        success = cap.read()[0]
    cap.release()
    return frames
