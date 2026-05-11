# ============================================================================
# SlowFast R50 — Single Video Inference Script
# Just set the paths below and run
# ============================================================================

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

try:
    from pytorchvideo.models.hub import slowfast_r50
except ImportError:
    os.system("pip install pytorchvideo --quiet")
    from pytorchvideo.models.hub import slowfast_r50

# ============================================================================
# SET YOUR PATHS HERE
# ============================================================================
CHECKPOINT_PATH = r'Slowfast\slowfast_r3_best.pth'
VIDEO_PATH      = r'Slowfast\major.mp4'
# ============================================================================

SLOW_FRAMES   = 8
FAST_FRAMES   = 32
IMG_SIZE      = 224
SEV_CLASSES   = ['minor', 'moderate', 'major']
MEAN          = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
STD           = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
ACC_THRESHOLD = 0.5

class SlowFastModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = slowfast_r50(pretrained=False)
        in_features = backbone.blocks[-1].proj.in_features
        backbone.blocks[-1].proj = nn.Identity()
        if hasattr(backbone.blocks[-1], 'dropout'):
            backbone.blocks[-1].dropout = nn.Identity()
        self.backbone    = backbone
        self.feature_dim = in_features
        self.accident_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        self.severity_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 3)
        )

    def forward(self, slow, fast):
        features = self.backbone([slow, fast])
        return {
            'accident_logits': self.accident_head(features),
            'severity_logits': self.severity_head(features)
        }

def slowfast_preprocess_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise ValueError(f"Could not read video: {video_path}")
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        all_frames.append(frame)
    cap.release()
    all_frames   = np.array(all_frames)
    T            = len(all_frames)
    fast_indices = np.linspace(0, T - 1, FAST_FRAMES, dtype=int)
    fast_arr     = all_frames[fast_indices]
    slow_arr     = fast_arr[::4]

    def to_tensor(arr):
        t = torch.from_numpy(arr).float() / 255.0
        t = t.permute(3, 0, 1, 2)
        return ((t - MEAN) / STD).unsqueeze(0)

    return to_tensor(slow_arr), to_tensor(fast_arr)

def slowfast_infer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = SlowFastModel().to(device)
    ck    = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state = {k.replace('module.', ''): v
             for k, v in ck['model_state_dict'].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Model loaded from {CHECKPOINT_PATH}")

    print(f"Processing video: {VIDEO_PATH}")
    slow, fast = slowfast_preprocess_video(VIDEO_PATH)
    slow = slow.to(device)
    fast = fast.to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs   = model(slow, fast)
        acc_prob  = torch.sigmoid(outputs['accident_logits']).item()
        sev_probs = torch.softmax(outputs['severity_logits'], dim=1).cpu().numpy()[0]
        sev_pred  = int(sev_probs.argmax())
    elapsed = (time.time() - t0) * 1000

    is_accident = acc_prob > ACC_THRESHOLD

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Accident detected  : {'YES' if is_accident else 'NO'}")
    print(f"Accident prob      : {acc_prob*100:.1f}%")
    print(f"Inference time     : {elapsed:.1f} ms")

    if is_accident:
        print(f"\nSeverity prediction: {SEV_CLASSES[sev_pred].upper()}")
        print("\nSeverity probabilities:")
        for cls, prob in zip(SEV_CLASSES, sev_probs):
            bar    = '#' * int(prob * 30)
            marker = ' <--' if cls == SEV_CLASSES[sev_pred] else ''
            print(f"  {cls:>10}: {prob*100:5.1f}%  {bar}{marker}")
    else:
        print("\nSeverity not computed (no accident detected)")

    