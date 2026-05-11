# ============================================================================
# R3D-18 — Single Video Inference Script
# Just set the paths below and run
# ============================================================================

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models.video as video_models
import time

# ============================================================================
# SET YOUR PATHS HERE
# ============================================================================
CHECKPOINT_PATH = r'R3D-18\checkpoint_best.pth'
VIDEO_PATH      = r'R3D-18\major.mp4'
# ============================================================================

NUM_FRAMES  = 16
FRAME_SIZE  = 112
CLASSES     = ['no accident', 'minor', 'moderate', 'major']
MEAN        = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD         = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class R3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = video_models.r3d_18(pretrained=False)
        inf = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(inf, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        return self.backbone(x)

def r_preprocess_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise ValueError(f"Could not read video: {video_path}")
    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            frames.append(frame)
        else:
            frames.append(frames[-1].copy() if frames
                          else np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8))
    cap.release()
    frames = np.array(frames, dtype=np.float32) / 255.0
    frames = (frames - MEAN) / STD
    tensor = torch.from_numpy(
        np.ascontiguousarray(frames.transpose(3, 0, 1, 2))
    ).float().unsqueeze(0)
    return tensor

def r_infer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = R3DModel().to(device)
    ck = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    print(f"Model loaded from {CHECKPOINT_PATH}")

    print(f"Processing video: {VIDEO_PATH}")
    tensor = r_preprocess_video(VIDEO_PATH).to(device)

    t0 = time.time()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred   = int(probs.argmax())
    elapsed = (time.time() - t0) * 1000

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Predicted class : {CLASSES[pred].upper()}")
    print(f"Inference time  : {elapsed:.1f} ms")
    print("\nClass probabilities:")
    for i, (cls, prob) in enumerate(zip(CLASSES, probs)):
        bar = '#' * int(prob * 30)
        marker = ' <--' if i == pred else ''
        print(f"  {cls:>12}: {prob*100:5.1f}%  {bar}{marker}")
    is_accident = pred > 0
    print(f"\nAccident detected : {'YES' if is_accident else 'NO'}")
    if is_accident:
        print(f"Severity          : {CLASSES[pred].upper()}")
