# edge_simulator.py  (standalone process on your machine; do NOT import server)
import base64, json, os, time
from collections import deque
from datetime import datetime, timezone

import cv2
import paho.mqtt.client as mqtt
import torch
import torch.nn as nn
import torchvision.models.video as video_models
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from torchvision import transforms

load_dotenv()

MQTT_TOPIC = "accident/detection"
MQTT_BROKER = os.environ["MQTT_HOST"]
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USER = os.environ["MQTT_USER"]
MQTT_PASS = os.environ["MQTT_PASS"]

VIDEO_PATH = os.getenv("SIM_VIDEO", r"accsense-backend\major.mp4")
WEIGHTS_PATH = os.getenv("SIM_WEIGHTS")              # optional local r3d_weights.pth
CLIP_LENGTH = 16
FRAME_STRIDE = int(os.getenv("SIM_FRAME_STRIDE", "1"))   # match r_preprocess_video's sampling
PUBLISH_EVERY_S = float(os.getenv("SIM_PUBLISH_EVERY_S", "3"))
CONFIDENCE_THRESHOLD = 0.40
SEVERITY_CLASSES = ["no_accident", "minor", "moderate", "major"]


class R3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = video_models.r3d_18(weights=None)
        inf = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(inf, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, 4))

    def forward(self, x):
        return self.backbone(x)


def load_model():
    path = WEIGHTS_PATH or hf_hub_download(          # cached after the first call
        repo_id=os.environ["HF_REPO_ID"], filename="r3d_weights.pth",
        token=os.getenv("HF_TOKEN"))
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model = R3DModel()
    model.load_state_dict(sd)
    print("Model loaded from", path)
    return model.eval()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989]),
])


def preprocess_clip(frames):
    t = [transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    return torch.stack(t, dim=1).unsqueeze(0)


def run_inference(model, frames):
    with torch.inference_mode():
        probs = torch.softmax(model(preprocess_clip(frames)) / 2.0, dim=1)[0].numpy()
    pred = int(probs.argmax())
    conf = float(probs[pred])
    severity, is_accident = SEVERITY_CLASSES[pred], pred > 0
    if conf < CONFIDENCE_THRESHOLD:
        severity, is_accident = "uncertain", False
    return severity, conf, is_accident


def publish(client, severity, conf, is_accident, frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (320, int(h * 320 / w)))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    payload = {
        "severity": severity,
        "confidence": round(conf * 100, 2),
        "is_accident": is_accident,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame": base64.b64encode(buf).decode("utf-8"),
    }
    client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
    print(f"[{payload['timestamp']}] {severity} ({payload['confidence']}%)")


def main():
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.tls_set()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()                              # required: acks, keepalive, reconnect

    model = load_model()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    buf = deque(maxlen=CLIP_LENGTH * FRAME_STRIDE)
    last_pub = 0.0
    print(f"Simulating on {VIDEO_PATH} | {fps:.0f} fps | publish every {PUBLISH_EVERY_S}s")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                buf.clear()
                continue
            buf.append(frame)

            now = time.monotonic()
            if len(buf) == buf.maxlen and now - last_pub >= PUBLISH_EVERY_S:
                clip = list(buf)[::FRAME_STRIDE]
                severity, conf, is_acc = run_inference(model, clip)
                publish(client, severity, conf, is_acc, frame)
                last_pub = time.monotonic()
            time.sleep(1 / fps)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()