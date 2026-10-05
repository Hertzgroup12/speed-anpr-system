# JNRD PRO - Intelligent Speed Enforcement Demo

A prototype that follows the requested flow: **detect speed → capture vehicle → read plate → create a review case**. It uses a FastAPI web application, YOLOv8 vehicle tracking, EasyOCR text recognition, optional Firebase Firestore persistence, and a simulated-by-default SMS workflow.

> **Important:** A flagged speed is an estimate, not proof of an offence. Every case remains pending until a human reviews or dismisses it. This demo must not be used by itself to issue fines or make legal decisions. Validate speed measurements with a calibrated camera and an independent reference before any operational use.

## Start the application (Windows PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` for the dashboard, or `http://127.0.0.1:8000/docs` for the API. The dashboard supports both uploaded video and live browser-camera monitoring. On the first analysis the app downloads/loads the YOLO model and EasyOCR model, which can take a few minutes. CPU inference works; a compatible CUDA installation can accelerate OCR.

Upload MP4, AVI, MOV, MKV, or WEBM video through the dashboard, or select **Start live camera** and grant browser camera permission. Live mode sends sampled JPEG frames from the browser to FastAPI over a WebSocket and waits for each frame to be processed before sending another; it does not expose the server computer's camera to a remote browser. Enter the real distance between the two virtual lines and a camera location. The default speed limit is 50 km/h. You can change settings in `.env` and restart the server.

Camera access works on `localhost` or on a site served over HTTPS. The browser may ask for camera permission. Only one live/video inference runs at a time because the YOLO tracker is stateful; a live session may make an uploaded video wait until monitoring stops. Live estimates use server frame-receipt timestamps and sampled webcam frames, so browser/network delay and inference rate affect speed accuracy. Validate against an independent speed reference before interpreting measurements.

## System flow

1. **Detect speed:** YOLOv8 tracks cars, motorcycles, buses, and trucks. OpenCV timestamps the same track crossing both lines and computes `distance_m / elapsed_seconds * 3.6`.
2. **Capture the vehicle:** When an estimated speed is above the configured limit, a compact vehicle image is saved under `data/captures/`. The uploaded video itself is processed from a temporary file and removed afterwards.
3. **Read the plate:** EasyOCR scans vehicle crops, recognizes plausible plate text, and uses its text-region coordinates to save an optional plate-text crop. This is best-effort OCR, not plate validation; for better accuracy, train/configure a dedicated plate detector.
4. **Report for review:** The dashboard displays measurements and creates a pending-review case when the estimated speed is over the limit. A reviewer can mark a case reviewed or dismiss it. Only reviewed cases may trigger the explicit SMS action.

Cases include plate text (when recognized), estimated speed, configured limit, excess speed, camera location, processing timestamp, video timestamp, and capture identifiers. Image URLs are available through the API.

## Configuration

Copy `.env.example` to `.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `YOLO_MODEL_PATH` | `yolov8n.pt` | YOLO model path or Ultralytics model name |
| `MAX_UPLOAD_MB` | `250` | Maximum video upload size |
| `SPEED_LIMIT_KMH` | `50` | Speed above which a case is flagged for review |
| `CAMERA_LOCATION` | `Demo camera` | Location used when the upload has no location |
| `CAPTURE_DIRECTORY` | `data/captures` | Local folder for flagged vehicle and plate crops |
| `OCR_LANGUAGES` | `["en"]` | EasyOCR languages as a JSON list |
| `OCR_GPU` | `false` | Enable EasyOCR GPU inference |
| `FIREBASE_ENABLED` | `false` | Store events and cases in Firestore |
| `FIREBASE_PROJECT_ID` | unset | Firebase project ID |
| `FIREBASE_CREDENTIALS` | unset | Service-account file path; otherwise Application Default Credentials |
| `FIREBASE_COLLECTION` | `speed_events` | Firestore collection for measurements |
| `OFFENSE_COLLECTION` | `speed_offenses` | Firestore collection for review cases |
| `SMS_SIMULATION` | `true` | Record a simulated notice without sending a text |
| `DRIVER_DIRECTORY` | `{}` | JSON object mapping recognized plates to driver phone numbers |
| `AFRICASTALKING_USERNAME` | unset | Africa's Talking account username |
| `AFRICASTALKING_API_KEY` | unset | Africa's Talking API key |
| `AFRICASTALKING_SENDER_ID` | unset | Optional approved sender ID |

### Firebase

Enable Firestore and set `FIREBASE_ENABLED=true`, `FIREBASE_PROJECT_ID`, and either `FIREBASE_CREDENTIALS` or Application Default Credentials. The web dashboard reads cases from Firestore when enabled. When it is disabled, cases are held in memory and disappear when the server restarts.

Vehicle and plate images are stored locally in `CAPTURE_DIRECTORY`, not Firebase Storage. Back up and protect that folder separately if image retention is required.

### Driver lookup and SMS

By default, notifications are simulated and no SMS is sent. For a demo, add test plate-to-number mappings in `.env` as JSON, for example:

```dotenv
DRIVER_DIRECTORY={"TEST1234":"+233200000000"}
```

The notice action is available only after a case is marked reviewed, requires an explicit dashboard action, and cannot be repeated for the same case. To enable live Africa's Talking requests, configure valid account credentials and set `SMS_SIMULATION=false`. Live sends are billable and may reach real recipients; test with numbers you control and obtain any required consent first. The application has no authentication, so do not expose it to the public internet.

## API

- `GET /health` — liveness check
- `GET /api/v1/config` — non-secret threshold, camera, and integration settings
- `POST /api/v1/analyze` — upload a video and receive its measurements and review-case count
- `WS /api/v1/live` — stream browser camera frames for live measurements
- `GET /api/v1/offenses` — list cases for the dashboard
- `PATCH /api/v1/offenses/{case_id}` — mark a case `reviewed` or `dismissed`
- `POST /api/v1/offenses/{case_id}/notify` — simulate or explicitly send a notice after review
- `GET /api/v1/captures/{capture_id}.jpg` — view a local vehicle/plate capture

Example upload:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/analyze" `
  -F "video=@traffic.mp4" `
  -F "distance_m=10" `
  -F "line_a_ratio=0.35" `
  -F "line_b_ratio=0.65" `
  -F "location=Test road"
```

## Calibration, safety, and privacy

The two line ratios specify vertical image positions as fractions of frame height, not real-world distance. Measure the real distance between those lines along the direction vehicles travel. Perspective, frame-rate errors, tracking switches, line placement, camera angle, image quality, and vehicle acceleration affect the estimate. Validate and document calibration for each camera.

This prototype has no login, authorization, audit-grade evidence chain, cloud image storage, or automatic retention policy. Use only lawful test footage. Add access control, HTTPS, restricted storage, retention and deletion rules, operator audit logs, camera calibration records, and legal/privacy review before any deployment.

## Roadmap

1. **Speed detection:** calibrated two-line speed estimation.
2. **Plate recognition:** vehicle tracking, OCR, and plate text crop.
3. **Review cases:** configurable speed threshold and captured vehicle evidence.
4. **Dashboard and alerts:** case review plus simulated/manual Africa's Talking workflow.
5. **Deployment report:** calibration validation and operational/privacy controls remain necessary before real-world deployment.

## Tests

```powershell
python -m pytest
```
