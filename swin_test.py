# ============================================================================
# Video Swin Transformer — Single Video Inference Script
# Just set the paths below and run
# ============================================================================

import cv2
import numpy as np
import torch
import torch.nn as nn
import time

from torchvision.models.video import swin3d_t

# ============================================================================
# SET YOUR PATHS HERE
# ============================================================================
CHECKPOINT_PATH = r'video swin transformer\swin_scratch_best_balanced.pth'
VIDEO_PATH      = r'video swin transformer\major.mp4'
# ============================================================================

NUM_FRAMES    = 16
IMG_SIZE      = 224
SEV_CLASSES   = ['minor', 'moderate', 'major']
MEAN          = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD           = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
ACC_THRESHOLD = 0.5

class SwinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone      = swin3d_t(weights=None)
        self.feature_dim   = 768
        self.backbone.head = nn.Identity()
        self.accident_head = nn.Sequential(
            nn.Linear(768, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        self.severity_head = nn.Sequential(
            nn.Linear(768, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return {
            'accident_logits': self.accident_head(features),
            'severity_logits': self.severity_head(features)
        }

def swin_preprocess_video(video_path):
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
        all_frames.append(frame)
    cap.release()
    all_frames = np.array(all_frames)
    T          = len(all_frames)
    indices    = np.linspace(0, T - 1, NUM_FRAMES, dtype=int)
    frames     = all_frames[indices]
    resized    = []
    for f in frames:
        f = cv2.resize(f, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        resized.append(f)
    frames = np.stack(resized)
    tensor = torch.from_numpy(frames).float() / 255.0
    tensor = tensor.permute(3, 0, 1, 2)
    tensor = (tensor - MEAN) / STD
    return tensor.unsqueeze(0)

def swin_infer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = SwinModel().to(device)
    ck    = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state = {k.replace('module.', ''): v
             for k, v in ck['model_state_dict'].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Model loaded from {CHECKPOINT_PATH}")

    print(f"Processing video: {VIDEO_PATH}")
    tensor = swin_preprocess_video(VIDEO_PATH).to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs   = model(tensor)
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

