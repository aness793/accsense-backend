import base64, glob, json, os, sys
from collections import deque
import cv2
import edge_simulator as sim   # reuses load_model, run_inference, constants

STEP_S = float(os.getenv("DEMO_STEP_S", "1.0"))

def encode_frame(frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (320, int(h * 320 / w)))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf).decode("utf-8")

def process(model, path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps * STEP_S))
    buf = deque(maxlen=sim.CLIP_LENGTH * sim.FRAME_STRIDE)
    events, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(frame)
        i += 1
        if len(buf) == buf.maxlen and i % step == 0:
            sev, conf, is_acc = sim.run_inference(model, list(buf)[::sim.FRAME_STRIDE])
            events.append({"severity": sev, "confidence": round(conf * 100, 2),
                           "is_accident": is_acc, "frame": encode_frame(frame),
                           "video": os.path.basename(path)})
    cap.release()
    return events

if __name__ == "__main__":
    model = sim.load_model()
    events = []
    for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.mp4"))):
        print("processing", p)
        events += process(model, p)
    with open("demo_events.json", "w", encoding="utf-8") as f:
        json.dump(events, f)
    print(len(events), "events written")