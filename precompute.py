from dotenv import load_dotenv
load_dotenv()

import json
import os
import subprocess
import tempfile

import cv2

from core import ACC_THRESHOLD, TEMPERATURE, infer

# EDIT THESE
CLIPS = {
    "crash": "clips/crash.mp4",
    "near_miss": "clips/near_miss.mp4",
    "normal": "clips/normal.mp4",
    "night_rain": "clips/night_rain.mp4",
}
W = 3.0   # window length in seconds, set to your training clip length
S = 1.5   # stride in seconds
OUT = "out"
os.makedirs(OUT, exist_ok=True)


def duration(path):
    cap = cv2.VideoCapture(path)
    d = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return d


def cut(src, t0, dst):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(t0), "-i", src, "-t", str(W), "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", dst],
        check=True, capture_output=True,
    )


def window_starts(dur):
    out, t = [], 0.0
    while t + W <= dur + 1e-6:
        out.append(round(t, 2))
        t += S
    return out or [0.0]


data = {
    cid: {
        "video": f"/clips/{cid}.mp4",
        "duration_s": round(duration(p), 2),
        "window_s": W,
        "stride_s": S,
        "temperature": TEMPERATURE,
        "threshold": ACC_THRESHOLD,
        "models": {},
    }
    for cid, p in CLIPS.items()
}

for name in ["r3d", "slowfast", "swin"]:  # models outer loop so each loads once
    for cid, path in CLIPS.items():
        print(f"{name} / {cid}")
        # same call the live endpoint uses, so presets match live results
        clip_level = infer(name, path)

        windows = []
        for t0 in window_starts(data[cid]["duration_s"]):
            with tempfile.TemporaryDirectory() as d:
                seg = os.path.join(d, "w.mp4")
                cut(path, t0, seg)
                r = infer(name, seg)
            windows.append({
                "t0": t0,
                "t1": round(t0 + W, 2),
                "p_accident": r["p_accident"],
                "sev_probs": r["sev_probs"],
            })

        best = max(windows, key=lambda w: w["p_accident"])
        entry = {
            "clip_level": clip_level,
            "windows": windows,
            "peak_p": best["p_accident"],
            "mean_p": round(sum(w["p_accident"] for w in windows) / len(windows), 4),
            "peak_window": [best["t0"], best["t1"]],
        }

        # Grad-CAM on the highest-scoring window, R3D only
        if name == "r3d" and best["p_accident"] > ACC_THRESHOLD:
            with tempfile.TemporaryDirectory() as d:
                seg = os.path.join(d, "best.mp4")
                cut(path, best["t0"], seg)
                entry["gradcam"] = infer("r3d", seg, gradcam=True).get("gradcam")

        data[cid]["models"][name] = entry

for cid, entry in data.items():
    with open(os.path.join(OUT, f"{cid}.json"), "w") as f:
        json.dump(entry, f)
print("done")