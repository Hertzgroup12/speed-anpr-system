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

## Sign up invitation code
1002d28df43434794126fc492482d154d6e7c884650b68d091704b74c93aff48

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
| `FIREBASE_WEB_API_KEY` | unset | Firebase web app API key for dashboard Analytics |
| `FIREBASE_AUTH_DOMAIN` | unset | Firebase web app auth domain |
| `FIREBASE_APP_ID` | unset | Firebase web app ID |
| `FIREBASE_MEASUREMENT_ID` | unset | Google Analytics measurement ID (`G-...`) |
| `FIREBASE_COLLECTION` | `speed_events` | Firestore collection for measurements |
| `OFFENSE_COLLECTION` | `speed_offenses` | Firestore collection for review cases |
| `SMS_SIMULATION` | `true` | Record a simulated notice without sending a text |
| `DRIVER_DIRECTORY` | `{}` | JSON object mapping recognized plates to driver phone numbers |
| `AFRICASTALKING_USERNAME` | unset | Africa's Talking account username |
| `AFRICASTALKING_API_KEY` | unset | Africa's Talking API key |
| `AFRICASTALKING_SENDER_ID` | unset | Optional approved sender ID |
| `APP_USERNAME` | unset | Required single operator login name |
| `APP_PASSWORD` | unset | Required operator password (at least 12 characters) |
| `APP_SECRET_KEY` | unset | Required random signing key (at least 32 characters) |
| `AUTH_COOKIE_SECURE` | `false` | Set `true` when accessed over HTTPS |
| `GEMINI_API_KEY` | unset | Optional Gemini API key for the AI assistant |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model used by the assistant |

### Login and AI assistant

The dashboard's API, review cases, captures, documentation, and live-camera WebSocket require a signed operator session. Configure `APP_USERNAME`, a unique `APP_PASSWORD` with at least 12 characters, and a cryptographically random `APP_SECRET_KEY` with at least 32 characters before starting the app. Without these settings the protected API fails closed. The session cookie is HTTP-only, SameSite strict, and expires after eight hours. Set `AUTH_COOKIE_SECURE=true` when using HTTPS (including a Cloudflare tunnel); leave it false only for local HTTP development.

Generate a signing key in PowerShell:

```powershell
$bytes = New-Object byte[] 48
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($bytes)
[Convert]::ToBase64String($bytes)
```

Copy the generated value into the ignored local `.env` as `APP_SECRET_KEY`, and configure the username and a strong password there too. Do not commit `.env` or share the password/signing key.

To enable the AI assistant, create a Gemini API key in [Google AI Studio](https://aistudio.google.com/app/apikey) and set `GEMINI_API_KEY` in the server environment. The key stays server-side. The assistant accepts up to 2,000 characters per request, allows 10 requests per minute per process, and does not receive dashboard cases or plate data automatically. Gemini calls use stateless requests (`store=false`); do not enter plate numbers or personal information in chat. Google API usage and quotas may apply.

The configured operator account is shared; invite-created accounts are individual logins but currently receive the same operator permissions. Use a unique strong password, keep the app private where possible, and rotate credentials if they may have been exposed.

The **Use device location** button requests browser geolocation only after the operator clicks it; browsers require HTTPS or localhost and location permission. Live-event locations include the coordinates when permission is granted, otherwise the manually configured camera location is used. Coordinates are sent to the server and included with live detections (and stored with event records when Firebase is enabled).

**Read embedded video location** checks an uploaded video for ISO 6709 GPS coordinates; the same metadata is also checked during analysis and included in the recorded location when present. Videos without embedded GPS continue to use the configured location. The authenticated `POST /api/v1/location/video` endpoint accepts multipart form data with a `video` file and returns `{"available": true, "location": {"latitude": ..., "longitude": ..., "altitude_m": ...}}` when coordinates exist, or `{"available": false, "location": null}` otherwise. Coordinates are processed by this app and are not reverse-geocoded; opening the map link sends them to OpenStreetMap.

The sign-in page offers **Create account** when `FIREBASE_ENABLED=true` and a 20-character-or-longer `ADMIN_INVITE_CODE` is configured. New accounts use an email address, a password of at least 12 characters, and the invitation code; account records are stored in Firestore. The configured operator account remains available for administration.

### Firebase

Enable Firestore and set `FIREBASE_ENABLED=true`, `FIREBASE_PROJECT_ID`, and either `FIREBASE_CREDENTIALS` or Application Default Credentials. The web dashboard reads cases from Firestore when enabled. When it is disabled, cases are held in memory and disappear when the server restarts.

To enable Firebase Analytics in the dashboard, register a **Web app** in Firebase, enable Google Analytics for the project, and copy its web configuration into `FIREBASE_WEB_API_KEY`, `FIREBASE_AUTH_DOMAIN`, `FIREBASE_APP_ID`, and `FIREBASE_MEASUREMENT_ID` in the server environment. The existing `FIREBASE_PROJECT_ID` is reused. Analytics initializes automatically after a user signs in and the browser supports it. These web configuration values are public client identifiers, not service-account secrets; continue to keep `FIREBASE_CREDENTIALS` private. The browser loads the Firebase 13.0.0 modular SDK from Google's `gstatic` CDN.

Vehicle and plate images are stored locally in `CAPTURE_DIRECTORY`, not Firebase Storage. Back up and protect that folder separately if image retention is required.

### Driver lookup and SMS

By default, notifications are simulated and no SMS is sent. For a demo, add test plate-to-number mappings in `.env` as JSON, for example:

```dotenv
DRIVER_DIRECTORY={"TEST1234":"+233200000000"}
```

The notice action is available only after a case is marked reviewed, requires an explicit dashboard action, and cannot be repeated for the same case. To enable live Africa's Talking requests, configure valid account credentials and set `SMS_SIMULATION=false`. Live sends are billable and may reach real recipients; test with numbers you control and obtain any required consent first. The application has no authentication, so do not expose it to the public internet.

## Deploy to Google Cloud Run

This app has no sign-in or authorization. Deploy it as a **private Cloud Run service**; do not allow unauthenticated/public access to footage, plate data, or case actions. The container uses Cloud Run's `PORT` setting and loads Firebase Admin credentials from the Cloud Run service identity (Application Default Credentials), so no service-account JSON key needs to be uploaded.

1. Push the project to GitHub, making sure `.env` and all service-account JSON files remain untracked.
2. In [Google Cloud Shell](https://console.cloud.google.com/?cloudshell=true), clone the repository and enter its directory:

   ```bash
   git clone https://github.com/Hertzgroup12/speed-anpr-system.git
   cd speed-anpr-system
   ```

   If deploying a newer commit, pull it in this directory before continuing.

3. Set the project and enable the Cloud Run build services:

   ```bash
   gcloud config set project speed-anpr-system
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```

4. Create a dedicated runtime identity and grant it Firestore access:

   ```bash
   gcloud iam service-accounts create jnrd-pro-runtime --display-name="JNRD PRO Cloud Run"
   gcloud projects add-iam-policy-binding speed-anpr-system \
     --member="serviceAccount:jnrd-pro-runtime@speed-anpr-system.iam.gserviceaccount.com" \
     --role="roles/datastore.user"
   ```

   If the service account already exists, skip its create command. The identity needs the Firestore/Datastore User role to read and write the configured Firestore database.

5. Add the operator password, session signing key, and Gemini API key to Secret Manager. Do not commit or upload these values to GitHub:

   ```bash
   read -r -s -p "Operator password (12+ characters): " APP_PASSWORD; echo
   printf '%s' "$APP_PASSWORD" | gcloud secrets create jnrd-pro-app-password --data-file=-
   unset APP_PASSWORD

   openssl rand -base64 48 | tr -d '\n' | gcloud secrets create jnrd-pro-app-secret --data-file=-

   read -r -s -p "Gemini API key: " GEMINI_API_KEY; echo
   printf '%s' "$GEMINI_API_KEY" | gcloud secrets create jnrd-pro-gemini-api-key --data-file=-
   unset GEMINI_API_KEY

   RUNTIME_SA="serviceAccount:jnrd-pro-runtime@speed-anpr-system.iam.gserviceaccount.com"
   for secret in jnrd-pro-app-password jnrd-pro-app-secret jnrd-pro-gemini-api-key; do
     gcloud secrets add-iam-policy-binding "$secret" \
       --member="$RUNTIME_SA" \
       --role="roles/secretmanager.secretAccessor"
   done
   ```

   If a secret already exists, add a new version with `gcloud secrets versions add SECRET_NAME --data-file=-` instead of creating it again. The Gemini key is available from [Google AI Studio](https://aistudio.google.com/app/apikey).

6. Deploy privately in the Netherlands region:

   ```bash
   gcloud run deploy jnrd-pro \
     --source . \
     --region europe-west4 \
     --service-account jnrd-pro-runtime@speed-anpr-system.iam.gserviceaccount.com \
     --memory 8Gi \
     --cpu 4 \
     --timeout 900 \
     --concurrency 1 \
     --max 2 \
     --no-allow-unauthenticated \
     --set-env-vars FIREBASE_ENABLED=true,FIREBASE_PROJECT_ID=speed-anpr-system,MAX_UPLOAD_MB=25,APP_USERNAME=operator,AUTH_COOKIE_SECURE=true \
     --set-secrets APP_PASSWORD=jnrd-pro-app-password:latest,APP_SECRET_KEY=jnrd-pro-app-secret:latest,GEMINI_API_KEY=jnrd-pro-gemini-api-key:latest
   ```

   Keep `FIREBASE_CREDENTIALS` unset on Cloud Run. The service identity provides Firebase credentials. The 25 MB upload limit stays below Cloud Run's request-size limit; the app's larger local upload default does not apply to Cloud Run. The first video analysis may take longer while YOLO and OCR models are downloaded.

6. Verify the service is healthy:

   ```bash
   gcloud run services describe jnrd-pro --region europe-west4 --format="value(status.url)"
   ```

   Access is private by default. To test from Cloud Shell, use the Cloud Run proxy:

   ```bash
   gcloud run services proxy jnrd-pro --region europe-west4 --port 8080
   ```

   In another Cloud Shell terminal, check `http://localhost:8080/health` and `http://localhost:8080/api/v1/config`. Stop the proxy with Ctrl+C. To let another operator use the service, grant that Google identity the `roles/run.invoker` role; do not grant `allUsers`.

Cloud Run's local filesystem is temporary: image captures may disappear when an instance stops, and instances do not share files. Firestore cases persist, but capture images are not backed up there. Use private Cloud Storage with access controls and retention rules before relying on captures in a real deployment. Cloud Run costs may include build, CPU/memory, and storage charges; review Google Cloud pricing and billing alerts first.

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
