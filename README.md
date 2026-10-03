# Speed ANPR System

A production-ready starter for an intelligent vehicle speed detection and automatic number plate recognition (ANPR) system powered by YOLOv8, EasyOCR, FastAPI, and Firebase.

## Overview

This project combines:
- Vehicle detection using YOLOv8
- License plate extraction and OCR using EasyOCR
- API services with FastAPI
- Data persistence and event logging with Firebase
- Speed estimation utilities for camera-based enforcement workflows

## Features

- Vehicle detection from uploaded images or video frames
- Number plate recognition from detected vehicles
- Speed estimation from calibration data
- REST API for inference and reporting
- Firebase-ready data storage and audit logging
- Modular design for adding streaming/video support

## Project Structure

```text
speed-anpr-system/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── schemas.py
│   └── services/
│       ├── __init__.py
│       ├── anpr_service.py
│       ├── firebase_service.py
│       ├── speed_service.py
│       └── yolo_service.py
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── tests/
    └── test_health.py
```

## Local Setup

1. Clone the repository
2. Create a virtual environment
3. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

4. Configure environment

```bash
cp .env.example .env
```

5. Start the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Example

Check service health:

```bash
curl http://localhost:8000/health
```

Submit an image for ANPR:

```bash
curl -X POST "http://localhost:8000/api/v1/anpr" \
  -F "file=@sample.jpg"
```

Estimate speed from calibration parameters:

```bash
curl -X POST "http://localhost:8000/api/v1/speed" \
  -H "Content-Type: application/json" \
  -d '{
    "distance_m": 20,
    "time_seconds": 2.5,
    "vehicle_length_m": 4.5
  }'
```

## Configuration

Environment variables:

- `APP_TITLE`
- `APP_VERSION`
- `APP_HOST`
- `APP_PORT`
- `FIREBASE_CREDENTIALS_PATH`
- `FIREBASE_DATABASE_URL`

## Notes

This is a starter implementation designed to provide a clean foundation for a real-world ANPR and speed enforcement platform. The detection and OCR pipeline can be extended with:
- video stream ingestion
- tracking across frames
- database event storage
- surveillance dashboard
- region-of-interest (ROI) configuration

## License

This project is open for educational and engineering experimentation.
