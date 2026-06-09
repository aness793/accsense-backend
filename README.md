# AcciSense Backend

FastAPI backend for AcciSense — an automatic road accident detection and severity estimation system using deep learning on surveillance video.

## Overview

This backend exposes a REST API and WebSocket server that:
- Runs inference on uploaded video clips using three deep learning models (R3D-18, SlowFast R50, Video Swin Transformer)
- Receives real-time accident detections from an edge simulator via MQTT
- Logs results to a Supabase database
- Sends push notifications via ntfy.sh
- Generates PDF incident reports

## Tech Stack

- **Framework**: FastAPI
- **Models**: R3D-18, SlowFast R50, Video Swin Transformer (PyTorch)
- **Database**: Supabase (PostgreSQL)
- **Messaging**: MQTT (HiveMQ Cloud)
- **Model Storage**: HuggingFace Hub
- **Server**: Uvicorn

## Project Structure

```
AcciSense-backend/
├── server.py           # Main FastAPI app
├── r3d_test.py         # R3D-18 preprocessing
├── slowfast_test.py    # SlowFast preprocessing
├── swin_test.py        # Swin Transformer preprocessing
├── requirements.txt
└── .env                # Environment variables (not committed)
```

## Environment Variables

Create a `.env` file in the root:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
MQTT_HOST=your_hivemq_host
MQTT_USER=your_hivemq_username
MQTT_PASS=your_hivemq_password
HF_REPO_ID=your_hf_username/AcciSense-models
HF_TOKEN=your_hf_token
```

## Running Locally

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Upload video and run inference |
| GET | `/history` | Fetch recent accident logs |
| POST | `/generate-report` | Generate PDF incident report |
| WS | `/ws` | WebSocket for live MQTT feed |

## Models

Models are loaded lazily on first request and downloaded automatically from HuggingFace Hub. The three `.pth` checkpoint files are not stored in this repository.

| Model | Task | Accuracy |
|-------|------|----------|
| R3D-18 | 4-class unified | 89.30% detection |
| SlowFast R50 | Dual-head | 87.63% detection |
| Video Swin Transformer | Dual-head | 88.96% detection |

## Author

Aness Rahmani — Master's Thesis, Université Djilali Bounaama Khemis Miliana  
Supervisor: Slim Rouabah
