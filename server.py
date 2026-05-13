# this is the server
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import shutil
import os
import tempfile
import time
import numpy as np
from datetime import datetime
from supabase import create_client
import requests
# Import your local test scripts
import slowfast_test as slowfast
import swin_test as swin
import r3d_test as r3d
from huggingface_hub import hf_hub_download
# Torchvision imports
import torchvision.models.video as video_models
from torchvision.models.video import swin3d_t

try:
    from pytorchvideo.models.hub import slowfast_r50
except ImportError:
    os.system("pip install pytorchvideo --quiet")
    from pytorchvideo.models.hub import slowfast_r50

# --- DATABASE CONFIG ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np
import base64

def generate_gradcam(model, clip_tensor, predicted_class, original_frame_bgr):
    target_layers = [model.backbone.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=clip_tensor, targets=None)
    grayscale_cam = grayscale_cam[0]  # could be (T, H, W) or (H, W, T)

    print(f"raw cam shape: {grayscale_cam.shape}")

    # Collapse temporal dimension → get 2D (H, W)
    if grayscale_cam.ndim == 3:
        grayscale_cam = grayscale_cam.mean(axis=0)  # average across temporal dim

    grayscale_cam = np.array(grayscale_cam, dtype=np.float32)
    grayscale_cam = np.clip(grayscale_cam, 0, 1)
    grayscale_cam = cv2.resize(grayscale_cam, (112, 112))

    print(f"final cam shape: {grayscale_cam.shape}")  # should be (112, 112)

    heatmap_uint8 = np.uint8(255 * grayscale_cam)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    frame_resized = cv2.resize(original_frame_bgr, (112, 112))
    overlay = cv2.addWeighted(frame_resized, 0.6, heatmap_colored, 0.4, 0)

    _, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    gradcam_b64 = base64.b64encode(buffer).decode('utf-8')

    return gradcam_b64
# --- 1. MODEL CLASS DEFINITIONS ---
NTFY_TOPIC = "btvad-accident-alerts-aness"  # change to something unique

def send_ntfy_notification(severity: str, confidence: str, model_name: str, video_name: str):
    if not severity or severity == "N/A":
        return

    severity_config = {
        "minor": {
            "priority": "default",
            "tags": "warning,car",
            "title": "Minor Accident Detected"
        },
        "moderate": {
            "priority": "high",
            "tags": "rotating_light,car",
            "title": "Moderate Accident Detected"
        },
        "major": {
            "priority": "urgent",
            "tags": "sos,rotating_light",
            "title": "MAJOR ACCIDENT DETECTED"
        }
    }

    config = severity_config.get(severity.lower())
    if not config:
        return

    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=f"Severity: {severity.upper()}\nConfidence: {confidence}\nModel: {model_name}\nFile: {video_name}",
            headers={
                "Title": config["title"],
                "Priority": config["priority"],
                "Tags": config["tags"]
            },
            timeout=5
        )
    except Exception as e:
        print(f"ntfy Error: {e}")
class SlowFastModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = slowfast_r50(pretrained=False)
        in_features = backbone.blocks[-1].proj.in_features
        backbone.blocks[-1].proj = nn.Identity()
        if hasattr(backbone.blocks[-1], 'dropout'):
            backbone.blocks[-1].dropout = nn.Identity()
        self.backbone = backbone
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

class SwinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = swin3d_t(weights=None)
        self.feature_dim = 768
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

# --- 2. GLOBAL SETTINGS & LOADING ---

models = {}
HF_REPO = os.getenv("HF_REPO_ID") 
HF_TOKEN = os.getenv("HF_TOKEN") 
# def load_all_models():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Load SlowFast
#     sf = SlowFastModel().to(device)
#     # ckpt_sf = torch.load('slowfast_r3_best.pth', map_location=device, weights_only=False)
#     ckpt_sf = torch.load(hf_hub_download(repo_id=HF_REPO, filename="slowfast_r3_best.pth", token=HF_TOKEN),map_location=device, weights_only=False)
#     state_dict_sf = ckpt_sf['model_state_dict']
#     new_state_dict_sf = {k.replace("module.", ""): v for k, v in state_dict_sf.items()}
#     sf.load_state_dict(new_state_dict_sf, strict=False)
#     sf.eval()
#     models["slowfast"] = sf
    
#     # Load R3D
#     r3d_m = R3DModel().to(device)
#     # ckpt_r3d = torch.load('checkpoint_best.pth', map_location=device, weights_only=False)
#     ckpt_r3d = torch.load(hf_hub_download(repo_id=HF_REPO, filename="checkpoint_best.pth", token=HF_TOKEN), map_location=device, weights_only=False)
#     state_dict_r3d = ckpt_r3d['model_state_dict'] if 'model_state_dict' in ckpt_r3d else ckpt_r3d
#     r3d_m.load_state_dict(state_dict_r3d)
#     r3d_m.eval()
#     models["r3d"] = r3d_m
    
#     # Load Swin
#     sw = SwinModel().to(device)
#     # ckpt_sw = torch.load('swin_scratch_best_balanced.pth', map_location=device, weights_only=False)
#     ckpt_sw = torch.load(hf_hub_download(repo_id=HF_REPO, filename="swin_scratch_best_balanced.pth", token=HF_TOKEN), map_location=device, weights_only=False)
#     state_dict_sw = ckpt_sw['model_state_dict']
#     new_state_dict_sw = {k.replace("module.", ""): v for k, v in state_dict_sw.items()}
#     sw.load_state_dict(new_state_dict_sw)
#     sw.eval()
#     models['swin'] = sw
    
#     print(f"All models loaded successfully on {device}!")
#     return device
def load_all_models(model_name):
    if model_name not in models:
        HF_REPO = os.getenv("HF_REPO_ID")
    
    if model_name == "r3d":
        m = R3DModel().to(device)
        ckpt = torch.load(hf_hub_download(repo_id=HF_REPO, filename="checkpoint_best.pth"), map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        m.load_state_dict(state_dict)
        m.eval()
        models["r3d"] = m

    elif model_name == "slowfast":
        m = SlowFastModel().to(device)
        ckpt = torch.load(hf_hub_download(repo_id=HF_REPO, filename="slowfast_r3_best.pth"), map_location=device, weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in ckpt['model_state_dict'].items()}
        m.load_state_dict(state_dict, strict=False)
        m.eval()
        models["slowfast"] = m

    elif model_name == "swin":
        m = SwinModel().to(device)
        ckpt = torch.load(hf_hub_download(repo_id=HF_REPO, filename="swin_scratch_best_balanced.pth"), map_location=device, weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in ckpt['model_state_dict'].items()}
        m.load_state_dict(state_dict)
        m.eval()
        models["swin"] = m

    return models[model_name]

# --- 3. INFERENCE FUNCTIONS ---

def log_to_supabase(data):
    try:
        supabase.table("accidents_log").insert(data).execute()
    except Exception as e:
        print(f"DB Logging Error: {e}")

def slowfast_infer(model, device, video_path):
    SEV_CLASSES = ['minor', 'moderate', 'major']
    slow, fast = slowfast.slowfast_preprocess_video(video_path)
    
    t0 = time.time()
    with torch.no_grad():
        outputs = model(slow.to(device), fast.to(device))
        # Temperature scaling (T=2.0) helps avoid 0% or 100% saturation
        acc_prob = torch.sigmoid(outputs['accident_logits'] / 2.0).item()
        sev_probs = torch.softmax(outputs['severity_logits'], dim=1).cpu().numpy()[0]
        sev_pred = int(sev_probs.argmax())
    elapsed = (time.time() - t0) * 1000

    is_accident = acc_prob > 0.5
    confidence = float(sev_probs[sev_pred]) if is_accident else (1.0 - acc_prob)

    return {
        "is_accident": is_accident,
        "probability": f"{confidence*100:.1f}%",
        "severity": SEV_CLASSES[sev_pred] if is_accident else "N/A",
        "time_ms": round(elapsed/1000, 1),
        "elapsed":elapsed
    }
import torchvision.transforms as transforms
from PIL import Image

def r_infer_with_gradcam(model, device, video_path):
    CLASSES = ['no accident', 'minor', 'moderate', 'major']

    # Preprocess video — reuse your existing function
    tensor = r3d.r_preprocess_video(video_path).to(device)

    t0 = time.time()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits / 2.0, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    elapsed = (time.time() - t0) * 1000

    is_accident = pred > 0
    conf = float(probs[pred])
    severity = CLASSES[pred] if is_accident else "N/A"

    gradcam_b64 = None

    # Only generate GradCAM if accident detected
    if is_accident:
        # Extract middle frame from the video for display
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, middle_frame = cap.read()
        cap.release()

        if ret:
            # GradCAM needs gradients — run outside no_grad
            gradcam_b64 = generate_gradcam(
                model=model,
                clip_tensor=tensor,
                predicted_class=pred,
                original_frame_bgr=middle_frame
            )

    return {
        "is_accident": is_accident,
        "probability": f"{conf*100:.1f}%",
        "severity": severity,
        "time_ms": round(elapsed / 1000, 1) * 10,
        "elapsed": elapsed,
        "gradcam": gradcam_b64  # None if no accident
    }
def r_infer(model, device, video_path):
    CLASSES = ['no accident', 'minor', 'moderate', 'major']
    tensor = r3d.r_preprocess_video(video_path).to(device)
    
    t0 = time.time()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits / 2.0, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
        elapsed = (time.time() - t0) * 1000
        elapsed_seconds = (time.time() - t0)  

    is_accident = pred > 0
    conf = float(probs[pred])

    return {
        "is_accident": is_accident,
        "probability": f"{conf*100:.1f}%",
        "severity": CLASSES[pred] if is_accident else "N/A",
        "time_ms":round(elapsed_seconds,1)*10 ,
        "elapsed":elapsed
    }

def swin_infer(model, device, video_path):
    SEV_CLASSES = ['minor', 'moderate', 'major']
    tensor = swin.swin_preprocess_video(video_path).to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(tensor)
        acc_prob = torch.sigmoid(outputs['accident_logits'] / 2.0).item()
        sev_probs = torch.softmax(outputs['severity_logits'], dim=1).cpu().numpy()[0]
        sev_pred = int(sev_probs.argmax())
    elapsed = (time.time() - t0) * 1000

    is_accident = acc_prob > 0.5
    confidence = float(sev_probs[sev_pred]) if is_accident else (1.0 - acc_prob)

    return {
        "is_accident": is_accident,
        "probability": f"{confidence*100:.1f}%",
        "severity": SEV_CLASSES[sev_pred] if is_accident else "N/A",
        "time_ms": round(elapsed/1000, 1),
        "elapsed":elapsed
    }

# --- 4. FASTAPI APP SETUP ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.on_event("startup")
# async def startup_event():
#     global device
#     device = load_all_models()
@app.on_event("startup")
async def startup_event():
    global loop
    loop = asyncio.get_event_loop()
    print("Server started. Models will load on first request.")


from fastapi import WebSocket
import paho.mqtt.client as mqtt
import asyncio
import json

connected_websockets = []
def on_mqtt_message(client, userdata, message):
    try:
        payload = json.loads(message.payload.decode())

        if payload.get("is_accident"):
            send_ntfy_notification(
                payload["severity"],
                f"{payload['confidence']}%",
                "edge_simulator",
                "live_feed"
            )

        log_data = {
            "video_name": "live_feed",
            "analyzed_at": payload["timestamp"],
            "inference_ms": 0,
            "model_name": "r3d_edge",
            "is_accident": payload["is_accident"],
            "severity": payload["severity"],
            "confidence": f"{payload['confidence']}%"
        }
        log_to_supabase(log_data)

        # Fix — schedule coroutine on the event loop instead of asyncio.run()
        for ws in connected_websockets:
            asyncio.run_coroutine_threadsafe(ws.send_json(payload), loop)  

    except Exception as e:
        print(f"MQTT message error: {e}")

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_mqtt_message
# mqtt_client.connect("localhost", 1883, 60)
mqtt_client.tls_set()
mqtt_client.username_pw_set(os.getenv("MQTT_USER"), os.getenv("MQTT_PASS"))
mqtt_client.connect(os.getenv("MQTT_HOST"), 8883, 60)
mqtt_client.subscribe("accident/detection")
mqtt_client.loop_start()

loop = None

@app.on_event("startup")
async def startup_event():
    global device, loop
    loop = asyncio.get_event_loop()
    # device = load_all_models()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except:
        connected_websockets.remove(websocket)


@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, video: UploadFile = File(...), model_name: str = Form(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name

    try:
        # if model_name == "slowfast":
        #     res = slowfast_infer(models["slowfast"], device, tmp_path)
        # elif model_name == "r3d":
        #     res = r_infer(models["r3d"], device, tmp_path)
        # elif model_name == "swin":
        #     res = swin_infer(models["swin"], device, tmp_path)
        # else:
        #     return {"status": "error", "message": "Invalid model selection"}
        if model_name == "slowfast":
                res = slowfast_infer(load_all_models("slowfast"), device, tmp_path)
        elif model_name == "r3d":
            res = r_infer(load_all_models("r3d"), device, tmp_path)
        elif model_name == "swin":
            res = swin_infer(load_all_models("swin"), device, tmp_path)
        # Prepare payload for Supabase
        log_data = {
            "video_name": video.filename,
            "analyzed_at": datetime.utcnow().isoformat(),
            "inference_ms": float(res["time_ms"]),
            "model_name": model_name,
            "is_accident": res["is_accident"],
            "severity": res["severity"],
            "confidence": res["probability"]
        }
        
        # Save to DB in the background so the user doesn't wait
        background_tasks.add_task(log_to_supabase, log_data)
        if res["is_accident"]:
            background_tasks.add_task(
                send_ntfy_notification,
                res["severity"],
                res["probability"],
                model_name,
                video.filename
            )

        return {
            "status": "success",
            "model": model_name,
            "results": res
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
from fastapi.responses import StreamingResponse
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import cm
import io
@app.get("/history")
async def get_history(limit: int = 10):
    try:
        response = supabase.table("accidents_log")\
            .select("*")\
            .order("analyzed_at", desc=True)\
            .limit(limit)\
            .execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.post("/generate-report")
async def generate_report(request: dict):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = ParagraphStyle('title', fontSize=20, fontName='Helvetica-Bold',
                                  textColor=colors.HexColor('#dc2626'), spaceAfter=6)
    elements.append(Paragraph("ACCIDENT DETECTION REPORT", title_style))
    elements.append(Paragraph("AccSense — Automatic Road Accident Detection System", styles['Normal']))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#dc2626'), spaceAfter=12))

    # Incident summary table
    severity = request.get("severity", "N/A").upper()
    confidence = request.get("confidence", "N/A")
    model_name = request.get("model_name", "N/A")
    video_name = request.get("video_name", "N/A")
    timestamp = request.get("timestamp", datetime.utcnow().isoformat())

    data = [
        ["Field", "Value"],
        ["Timestamp", timestamp],
        ["Video File", video_name],
        ["Model Used", model_name.upper()],
        ["Accident Detected", "YES"],
        ["Severity Level", severity],
        ["Confidence Score", confidence],
        ["Status", " REQUIRES IMMEDIATE RESPONSE"],
    ]

    table = Table(data, colWidths=[6*cm, 10*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc2626')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fef2f2')),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#fca5a5')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fef2f2')]),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.5*cm))

    # Recommended actions
    elements.append(Paragraph("Recommended Actions", ParagraphStyle(
        'heading', fontSize=13, fontName='Helvetica-Bold', spaceAfter=6, spaceBefore=12
    )))

    actions = [
        "1. Dispatch emergency services to the incident location immediately.",
        "2. Alert traffic management center to redirect vehicles.",
        "3. Notify nearest hospital and medical response team.",
        "4. Document the incident for insurance and legal purposes.",
        "5. Review footage for forensic analysis if required.",
    ]
    for action in actions:
        elements.append(Paragraph(action, styles['Normal']))
        elements.append(Spacer(1, 0.2*cm))

    elements.append(Spacer(1, 0.5*cm))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Paragraph(
        f"Generated automatically by AccSense on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        ParagraphStyle('footer', fontSize=8, textColor=colors.grey, spaceBefore=6)
    ))

    doc.build(elements)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=accident_report_{timestamp[:10]}.pdf"}
    )