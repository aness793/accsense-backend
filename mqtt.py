import cv2
import torch
import torch.nn as nn
import numpy as np
import paho.mqtt.client as mqtt
import json
import time
import base64
from torchvision import transforms
import torchvision.models.video as video_models

# --- CONFIG ---
MQTT_BROKER = "your-broker-address"
MQTT_PORT = 1883
MQTT_TOPIC = "accident/detection"
VIDEO_PATH = "accident_video.mp4"      # your prerecorded video
MODEL_PATH = "checkpoint_best.pth"     # your R3D-18 checkpoint
CLIP_LENGTH = 16                        # frames per clip
INFERENCE_EVERY = 16                    # run inference every N frames
CONFIDENCE_THRESHOLD = 0.60            # below this → uncertain

SEVERITY_CLASSES = ['no_accident', 'minor', 'moderate', 'major']

# --- MODEL ---
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

def load_model(model_path):
    device = torch.device('cpu')  # edge device = CPU
    model = R3DModel().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")
    return model, device

# --- PREPROCESSING ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.43216, 0.394666, 0.37645],
        std=[0.22803, 0.22145, 0.216989]
    )
])

def preprocess_clip(frames):
    tensors = [transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    # (C, T, H, W) → add batch dim → (1, C, T, H, W)
    return torch.stack(tensors, dim=1).unsqueeze(0)

# --- INFERENCE ---
def run_inference(model, device, frames):
    clip_tensor = preprocess_clip(frames).to(device)
    with torch.no_grad():
        logits = model(clip_tensor)
        probs = torch.softmax(logits / 2.0, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
        confidence = float(probs[pred])

    severity = SEVERITY_CLASSES[pred]
    is_accident = pred > 0

    # Low confidence fallback
    if confidence < CONFIDENCE_THRESHOLD:
        severity = "uncertain"
        is_accident = False

    return severity, confidence, is_accident

# --- MQTT PUBLISH ---
def publish(client, severity, confidence, is_accident, frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    frame_b64 = base64.b64encode(buffer).decode('utf-8')

    payload = {
        "severity": severity,
        "confidence": round(confidence * 100, 2),
        "is_accident": is_accident,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame": frame_b64
    }

    client.publish(MQTT_TOPIC, json.dumps(payload))
    print(f"[{payload['timestamp']}] Published → {severity} ({payload['confidence']}%)")

# --- MAIN LOOP ---
def main():
    # MQTT setup
    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    # Model
    model, device = load_model(MODEL_PATH)

    # Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_buffer = []
    frame_count = 0

    print(f"Starting simulation on: {VIDEO_PATH}")
    print(f"Video FPS: {fps} | Inference every {INFERENCE_EVERY} frames\n")

    while True:
        ret, frame = cap.read()

        # Loop video when it ends — simulates continuous camera
        if not ret:
            print("Video ended, looping...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_buffer.clear()
            frame_count = 0
            continue

        frame_buffer.append(frame.copy())
        frame_count += 1

        if len(frame_buffer) > CLIP_LENGTH:
            frame_buffer.pop(0)

        if frame_count % INFERENCE_EVERY == 0 and len(frame_buffer) == CLIP_LENGTH:
            severity, confidence, is_accident = run_inference(model, device, frame_buffer)
            publish(client, severity, confidence, is_accident, frame)

        # Control playback speed to match real video FPS
        time.sleep(1 / fps)

    cap.release()
    client.loop_stop()

if __name__ == "__main__":
    main()