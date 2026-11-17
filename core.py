import os
import gc
import time
import base64
import threading

import numpy as np
import torch
import torch.nn as nn
import torchvision.models.video as video_models
from torchvision.models.video import swin3d_t
from pytorchvideo.models.hub import slowfast_r50
from huggingface_hub import hf_hub_download

import slowfast_test as slowfast
import swin_test as swin
import r3d_test as r3d

torch.set_num_threads(int(os.getenv("TORCH_THREADS", "1")))
device = torch.device("cpu")

HF_REPO = os.getenv("HF_REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

# Temperature is a hack until you fit it on a validation set (temperature scaling).
TEMPERATURE = float(os.getenv("TEMPERATURE", "2.0"))
ACC_THRESHOLD = float(os.getenv("ACC_THRESHOLD", "0.5"))
# |p - threshold| below this band gives the "uncertain" decision
UNCERTAIN_BAND = float(os.getenv("UNCERTAIN_BAND", "0.10"))
SEV_CLASSES = ["minor", "moderate", "major"]


# ---------------- MODEL DEFINITIONS ----------------

class SlowFastModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = slowfast_r50(pretrained=False)
        in_features = backbone.blocks[-1].proj.in_features
        backbone.blocks[-1].proj = nn.Identity()
        if hasattr(backbone.blocks[-1], "dropout"):
            backbone.blocks[-1].dropout = nn.Identity()
        self.backbone = backbone
        self.feature_dim = in_features
        self.accident_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1)
        )
        self.severity_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 3)
        )

    def forward(self, slow, fast):
        features = self.backbone([slow, fast])
        return {
            "accident_logits": self.accident_head(features),
            "severity_logits": self.severity_head(features),
        }


class R3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = video_models.r3d_18(weights=None)
        inf = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(inf, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        return self.backbone(x)


class SwinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = swin3d_t(weights=None)
        self.feature_dim = 768
        self.backbone.head = nn.Identity()
        self.accident_head = nn.Sequential(
            nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 1)
        )
        self.severity_head = nn.Sequential(
            nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 3)
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return {
            "accident_logits": self.accident_head(features),
            "severity_logits": self.severity_head(features),
        }


# ---------------- SINGLE-SLOT MODEL LOADER ----------------

SPECS = {
    "r3d": (R3DModel, "r3d_weights.pth"),
    "slowfast": (SlowFastModel, "slowfast_weights.pth"),
    "swin": (SwinModel, "swin_weights.pth"),
}
_slot = {"name": None, "model": None}
_load_lock = threading.Lock()


def loaded_model_name():
    return _slot["name"]


def get_model(name):
    """Keeps at most one model in RAM. Switching models evicts the previous one."""
    if name not in SPECS:
        raise ValueError(f"unknown model: {name}")
    with _load_lock:
        if _slot["name"] == name:
            return _slot["model"]

        _slot["name"], _slot["model"] = None, None
        gc.collect()

        cls, filename = SPECS[name]
        m = cls().to(device)
        path = hf_hub_download(repo_id=HF_REPO, filename=filename, token=HF_TOKEN)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}

        if name == "slowfast":
            res = m.load_state_dict(state, strict=False)
            # If the head keys show up here, you are running a randomly initialized head.
            print("slowfast missing keys:", res.missing_keys)
            print("slowfast unexpected keys:", res.unexpected_keys)
        else:
            m.load_state_dict(state)

        m.eval()
        del ckpt, state
        gc.collect()
        _slot["name"], _slot["model"] = name, m
        print(f"model loaded: {name}")
        return m


# ---------------- RESULT BUILDING ----------------

def _build_result(p_acc, sev_probs, pre_ms, inf_ms, is_acc=None):
    # is_acc: pass it explicitly when the model's own rule differs (R3D argmax)
    if is_acc is None:
        is_acc = p_acc > ACC_THRESHOLD
    # display flag only, not a model output
    uncertain = abs(p_acc - ACC_THRESHOLD) < UNCERTAIN_BAND

    sev_idx = int(np.argmax(sev_probs))
    label = SEV_CLASSES[sev_idx] if is_acc else "no accident"
    conf = p_acc if is_acc else 1.0 - p_acc   # confidence in the accident / no accident call

    return {
        "label": label,                         # no accident | minor | moderate | major
        "uncertain": bool(uncertain),
        "is_accident": bool(is_acc),
        "p_accident": round(float(p_acc), 4),
        "confidence": round(float(conf), 4),
        "probability": f"{conf * 100:.1f}%",
        "severity": label if is_acc else "N/A",
        "severity_confidence": round(float(sev_probs[sev_idx]), 4) if is_acc else None,
        "sev_probs": [round(float(x), 4) for x in sev_probs],
        "preprocess_ms": round(pre_ms, 1),
        "inference_ms": round(inf_ms, 1),
        "time_ms": round((pre_ms + inf_ms) / 1000.0, 1),   # legacy, seconds
    }


# ---------------- GRAD-CAM (R3D only, opt-in) ----------------

def _gradcam_overlay(model, tensor, target_idx, video_path):
    import cv2
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(total // 2, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    cam = GradCAM(model=model, target_layers=[model.backbone.layer4[-1]])
    try:
        with torch.enable_grad():
            cams = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(target_idx)])
    finally:
        cam.activations_and_grads.release()

    g = np.asarray(cams[0], dtype=np.float32)
    if g.ndim == 3:
        g = g.mean(axis=0)
    g = cv2.resize(np.clip(g, 0, 1), (112, 112))
    heat = cv2.applyColorMap(np.uint8(255 * g), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.resize(frame, (112, 112)), 0.6, heat, 0.4, 0)
    _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


# ---------------- INFERENCE ----------------

def r3d_infer(model, video_path, gradcam=False):
    t0 = time.perf_counter()
    tensor = r3d.r_preprocess_video(video_path).to(device)
    pre_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(tensor) / TEMPERATURE, dim=1).cpu().numpy()[0]
    inf_ms = (time.perf_counter() - t1) * 1000

    # classes: [no accident, minor, moderate, major]
    # NOTE: decision uses P(accident) = 1 - p[0] vs threshold, not argmax over 4 classes.
        # classes: [no accident, minor, moderate, major]
    pred = int(np.argmax(probs))              # your trained decision rule
    is_acc = pred > 0
    p_acc = float(1.0 - probs[0])
    sev = probs[1:] / max(float(probs[1:].sum()), 1e-9)
    res = _build_result(p_acc, sev, pre_ms, inf_ms, is_acc=is_acc)
    res["class_probs"] = [round(float(x), 4) for x in probs]

    if gradcam and res["is_accident"]:
        try:
            target = int(np.argmax(sev)) + 1
            res["gradcam"] = _gradcam_overlay(model, tensor, target, video_path)
        except Exception as e:
            print(f"gradcam error: {e}")
            res["gradcam"] = None
    return res


def _two_head(model, inputs, pre_ms):
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model(*inputs)
        p_acc = torch.sigmoid(out["accident_logits"] / TEMPERATURE).item()
        sev = torch.softmax(out["severity_logits"], dim=1).cpu().numpy()[0]
    inf_ms = (time.perf_counter() - t1) * 1000
    return _build_result(p_acc, sev, pre_ms, inf_ms)


def slowfast_infer(model, video_path, gradcam=False):
    t0 = time.perf_counter()
    slow, fast = slowfast.slowfast_preprocess_video(video_path)
    slow, fast = slow.to(device), fast.to(device)
    pre_ms = (time.perf_counter() - t0) * 1000
    return _two_head(model, (slow, fast), pre_ms)


def swin_infer(model, video_path, gradcam=False):
    t0 = time.perf_counter()
    tensor = swin.swin_preprocess_video(video_path).to(device)
    pre_ms = (time.perf_counter() - t0) * 1000
    return _two_head(model, (tensor,), pre_ms)


INFER = {"r3d": r3d_infer, "slowfast": slowfast_infer, "swin": swin_infer}


def infer(name, video_path, gradcam=False):
    model = get_model(name)
    return INFER[name](model, video_path, gradcam=gradcam)