from dotenv import load_dotenv
load_dotenv()  # must run before importing core (it reads env vars at import)

import asyncio
import io
import json
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import paho.mqtt.client as mqtt
import requests
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from supabase import create_client

from core import INFER, get_model, infer, loaded_model_name

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "60"))

# ---------------- SUPABASE ----------------
supabase = None
_url, _key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
if _url and _key:
    try:
        supabase = create_client(_url, _key)
    except Exception as e:
        print(f"Supabase init error: {e}")


def log_to_supabase(data):
    if supabase is None:
        return
    try:
        supabase.table("accidents_log").insert(data).execute()
    except Exception as e:
        print(f"DB Logging Error: {e}")


# ---------------- NTFY ----------------
def send_ntfy_notification(severity, confidence, model_name, video_name):
    topic = os.getenv("NTFY_TOPIC")  # use a long random topic, never commit it
    if not topic or not severity or severity == "N/A":
        return

    config = {
        "minor": {"priority": "default", "tags": "warning,car", "title": "Minor Accident Detected"},
        "moderate": {"priority": "high", "tags": "rotating_light,car", "title": "Moderate Accident Detected"},
        "major": {"priority": "urgent", "tags": "sos,rotating_light", "title": "MAJOR ACCIDENT DETECTED"},
    }.get(str(severity).lower())
    if not config:
        return

    try:
        requests.post(
            f"https://ntfy.sh/{topic}",
            data=f"Severity: {str(severity).upper()}\nConfidence: {confidence}\nModel: {model_name}\nFile: {video_name}",
            headers={"Title": config["title"], "Priority": config["priority"], "Tags": config["tags"]},
            timeout=5,
        )
    except Exception as e:
        print(f"ntfy Error: {e}")


# ---------------- ERRORS ----------------
class ApiError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message


# ---------------- WEBSOCKET + MQTT ----------------
connected_websockets = []
loop = None
mqtt_client = None


async def broadcast(payload):
    for ws in list(connected_websockets):
        try:
            await ws.send_json(payload)
        except Exception:
            if ws in connected_websockets:
                connected_websockets.remove(ws)


def on_mqtt_connect(client, userdata, flags, rc):
    print(f"MQTT connected, rc={rc}")
    client.subscribe("accident/detection")


def on_mqtt_message(client, userdata, message):
    try:
        payload = json.loads(message.payload.decode())

        if payload.get("is_accident"):
            send_ntfy_notification(
                payload.get("severity"), f"{payload.get('confidence')}%", "edge_simulator", "live_feed"
            )

        log_to_supabase({
            "video_name": "live_feed",
            "analyzed_at": payload.get("timestamp"),
            "inference_ms": 0,
            "model_name": "r3d_edge",
            "is_accident": payload.get("is_accident"),
            "severity": payload.get("severity"),
            "confidence": f"{payload.get('confidence')}%",
        })

        if loop is not None:
            asyncio.run_coroutine_threadsafe(broadcast(payload), loop)
    except Exception as e:
        print(f"MQTT message error: {e}")


def make_mqtt_client():
    if hasattr(mqtt, "CallbackAPIVersion"):  # paho-mqtt 2.x
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    return mqtt.Client()


def start_mqtt():
    global mqtt_client
    host = os.getenv("MQTT_HOST")
    if not host:
        print("MQTT disabled (MQTT_HOST not set)")
        return
    try:
        c = make_mqtt_client()
        c.tls_set()
        c.username_pw_set(os.getenv("MQTT_USER"), os.getenv("MQTT_PASS"))
        c.on_connect = on_mqtt_connect
        c.on_message = on_mqtt_message
        c.connect_async(host, int(os.getenv("MQTT_PORT", "8883")), 60)
        c.loop_start()
        mqtt_client = c
    except Exception as e:
        print(f"MQTT start error (server keeps running): {e}")


def stop_mqtt():
    if mqtt_client is not None:
        try:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        except Exception:
            pass


# ---------------- APP ----------------
@asynccontextmanager
async def lifespan(app):
    global loop
    loop = asyncio.get_running_loop()
    start_mqtt()
    print("Server started. Models load on first request or /warmup.")
    yield
    stop_mqtt()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ApiError)
async def api_error_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"status": "error", "message": exc.message})


# one inference at a time (free tier RAM/CPU)
_infer_lock = threading.Lock()


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": loaded_model_name(), "busy": _infer_lock.locked()}


def _warm(name):
    if _infer_lock.acquire(blocking=False):  # never evict a model that is mid-inference
        try:
            get_model(name)
        except Exception as e:
            print(f"warmup error: {e}")
        finally:
            _infer_lock.release()


@app.post("/warmup")
def warmup(background_tasks: BackgroundTasks, model_name: str = "r3d"):
    if model_name not in INFER:
        raise ApiError(400, "invalid model")
    background_tasks.add_task(_warm, model_name)
    return {"status": "warming", "model": model_name}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)


# ---------------- UPLOAD HELPERS ----------------
def save_upload(video: UploadFile, max_bytes: int):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    size = 0
    try:
        while True:
            chunk = video.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise ApiError(413, f"file too large, max {MAX_UPLOAD_MB} MB")
            tmp.write(chunk)
    except Exception:
        tmp.close()
        os.unlink(tmp.name)
        raise
    tmp.close()
    return tmp.name


def check_duration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not fps or not frames:
        raise ApiError(400, "unreadable video")
    if frames / fps > MAX_VIDEO_SECONDS:
        raise ApiError(413, f"video too long, max {MAX_VIDEO_SECONDS} s")


# ---------------- PREDICT ----------------
# plain def on purpose: runs in the threadpool so /health and /ws stay responsive during inference
@app.post("/predict")
def predict(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    model_name: str = Form(...),
    gradcam: bool = Form(False),
):
    if model_name not in INFER:
        raise ApiError(400, "invalid model selection")
    if not _infer_lock.acquire(blocking=False):
        raise ApiError(429, "server busy, retry in a few seconds")

    tmp_path = None
    try:
        tmp_path = save_upload(video, MAX_UPLOAD_MB * 1024 * 1024)
        check_duration(tmp_path)
        res = infer(model_name, tmp_path, gradcam=(gradcam and model_name == "r3d"))
    except ApiError:
        raise
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise ApiError(500, str(e))
    finally:
        _infer_lock.release()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    background_tasks.add_task(log_to_supabase, {
        "video_name": video.filename,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "inference_ms": float(res["inference_ms"]),  # real milliseconds now
        "model_name": model_name,
        "is_accident": res["is_accident"],
        "severity": res["severity"],
        "confidence": res["probability"],
    })
    if res["is_accident"]:
        background_tasks.add_task(
            send_ntfy_notification, res["severity"], res["probability"], model_name, video.filename
        )

    return {"status": "success", "model": model_name, "results": res}


# ---------------- HISTORY ----------------
@app.get("/history")
def get_history(limit: int = 10):
    if supabase is None:
        return {"status": "error", "message": "database not configured"}
    try:
        response = (
            supabase.table("accidents_log")
            .select("*")
            .order("analyzed_at", desc=True)
            .limit(min(limit, 100))
            .execute()
        )
        return {"status": "success", "data": response.data}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------- PDF REPORT ----------------
@app.post("/generate-report")
def generate_report(request: dict):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2 * cm, leftMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    elements = []

    red = colors.HexColor("#dc2626")
    elements.append(Paragraph("ACCIDENT DETECTION REPORT",
                              ParagraphStyle("title", fontSize=20, fontName="Helvetica-Bold",
                                             textColor=red, spaceAfter=6)))
    elements.append(Paragraph("AccSense - Automatic Road Accident Detection System", styles["Normal"]))
    elements.append(HRFlowable(width="100%", thickness=2, color=red, spaceAfter=12))

    severity = str(request.get("severity") or "N/A").upper()
    confidence = request.get("confidence", "N/A")
    model_name = str(request.get("model_name") or "N/A")
    video_name = request.get("video_name", "N/A")
    timestamp = str(request.get("timestamp") or datetime.now(timezone.utc).isoformat())

    data = [
        ["Field", "Value"],
        ["Timestamp", timestamp],
        ["Video File", video_name],
        ["Model Used", model_name.upper()],
        ["Accident Detected", "YES"],
        ["Severity Level", severity],
        ["Confidence Score", confidence],
        ["Status", "REQUIRES IMMEDIATE RESPONSE"],
    ]
    table = Table(data, colWidths=[6 * cm, 10 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), red),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 12),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#fca5a5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fef2f2")]),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.5 * cm))

    elements.append(Paragraph("Recommended Actions", ParagraphStyle(
        "heading", fontSize=13, fontName="Helvetica-Bold", spaceAfter=6, spaceBefore=12)))
    for action in [
        "1. Dispatch emergency services to the incident location immediately.",
        "2. Alert traffic management center to redirect vehicles.",
        "3. Notify nearest hospital and medical response team.",
        "4. Document the incident for insurance and legal purposes.",
        "5. Review footage for forensic analysis if required.",
    ]:
        elements.append(Paragraph(action, styles["Normal"]))
        elements.append(Spacer(1, 0.2 * cm))

    elements.append(Spacer(1, 0.5 * cm))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Paragraph(
        f"Generated automatically by AccSense on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        ParagraphStyle("footer", fontSize=8, textColor=colors.grey, spaceBefore=6)))

    doc.build(elements)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=accident_report_{timestamp[:10]}.pdf"},
    )