# Vehicle Speed Detection and ANPR

A video-analysis API that tracks vehicles with YOLOv8, reads visible license plates with EasyOCR, calculates speed between two calibrated virtual lines, and can save speed events to Firebase Firestore.

> **Measurement note:** A camera needs a known distance between the two measurement lines. The reported speed is an estimate based on that distance and the video's frame rate; camera perspective, frame-rate accuracy, tracking quality, and line placement affect accuracy. Calibrate and validate the camera before using measurements operationally.

## Requirements

- Python 3.10 or newer
- A video file supported by your OpenCV installation
- Internet access on the first run if the YOLO weights file is not already present
- Optional: a Firebase project with Firestore enabled and server credentials

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation. `GET /health` checks that the API is running; it deliberately does not download or initialize AI models.

The first video request loads `yolov8n.pt` and EasyOCR's English recognition model. Model initialization and video inference can take time, and CPU-only inference is supported.

## Analyze a video

Send a multipart `POST /api/v1/analyze` request. `distance_m` is the real-world distance between the two horizontal lines; the line ratios are their vertical positions as a fraction of the video height:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/analyze" `
  -F "video=@traffic.mp4" `
  -F "distance_m=10" `
  -F "line_a_ratio=0.35" `
  -F "line_b_ratio=0.65"
```

The API returns measured vehicle events with track ID, best recognized plate (if readable), speed in km/h, direction, and the video time at which both lines were crossed. Vehicles are restricted to the car, motorcycle, bus, and truck classes in the standard COCO YOLO model. An event is only produced when a tracked vehicle crosses both lines.

### API settings

Copy `.env.example` to `.env` to configure:

| Variable | Default | Purpose |
| --- | --- | --- |
| `YOLO_MODEL_PATH` | `yolov8n.pt` | Ultralytics YOLO weights path or model name |
| `MAX_UPLOAD_MB` | `250` | Maximum accepted video upload size |
| `OCR_LANGUAGES` | `["en"]` | EasyOCR language codes, as a JSON array |
| `OCR_GPU` | `false` | Enable EasyOCR GPU inference when CUDA is available |
| `FIREBASE_ENABLED` | `false` | Enable event persistence to Firestore |
| `FIREBASE_PROJECT_ID` | unset | Firebase project ID |
| `FIREBASE_CREDENTIALS` | unset | Path to a service-account JSON file; otherwise Application Default Credentials are used |
| `FIREBASE_COLLECTION` | `speed_events` | Firestore collection for detections |

To enable Firebase, set `FIREBASE_ENABLED=true`, configure credentials (or Application Default Credentials), and enable Firestore in the Firebase project. Persistence errors are returned as server errors; events are not silently discarded.

## Design and limitations

- **Speed calibration:** Speed is calculated as `distance_m / elapsed_seconds * 3.6`. The two line ratios and distance must match the actual camera view. A single pixel-to-meter scale is not assumed.
- **Plate recognition:** OCR operates on vehicle detection crops and is best-effort. Occlusion, motion blur, resolution, viewing angle, lighting, and regional plate formats can prevent or degrade recognition. The API does not claim a plate is valid or verify vehicle identity.
- **Uploads:** Only `.mp4`, `.avi`, `.mov`, `.mkv`, and `.webm` file extensions are accepted. Files are processed from temporary storage and removed after analysis.
- **Concurrency:** Inference is serialized within each application process because the YOLO video tracker is stateful. For production, use a job queue and worker processes for long videos, plus authentication, HTTPS, retention controls, and appropriate access restrictions.
- **Privacy:** Video and recognized plate data can be sensitive. Secure the service, limit retention, and comply with applicable privacy and traffic-monitoring laws.

## Tests

```powershell
python -m pytest
```
