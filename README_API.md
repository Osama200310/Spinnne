# Simple Python Web Server for Angular + Image Upload API

This project provides a minimal FastAPI server that can host a built Angular application and exposes a simple POST endpoint to receive images via multipart/form-data.

## Features
- Serves static files (your Angular build) from `server/static/` at the root path (`/`).
- SPA fallback: non-API routes return `index.html` so Angular routing works on refresh.
- `POST /api/upload` to receive image files and store them in `server/uploads/`.
- Optional static serving of uploaded files at `/uploads/<filename>`.
- CORS enabled for all origins by default (adjust for production).

## Requirements
- Python 3.9+
- Install dependencies (includes Pillow for image validation):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the server

```bash
cd server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Now open http://localhost:8000/ — you will see a placeholder page until you copy your Angular build.

Health check:
```bash
curl http://localhost:8000/api/health
```

## Place your Angular build
1. Build your Angular app (example):
   ```bash
   ng build --configuration production
   ```
2. Copy the build output (e.g., everything under `dist/<your-project>/`) into:
   ```
   server/static/
   ```

The server will then serve your `index.html`, assets, and any deep links will be handled by the SPA fallback.

## Upload images via POST
- Endpoint: `POST /api/upload`
- Content type: `multipart/form-data`
- Field name: `file`

Example with `curl` (explicit POST shown):
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:8000/api/upload
```

Successful response example:
```json
{
  "filename": "a1b2c3d4_image.jpg",
  "size": 123456,
  "content_type": "image/jpeg",
  "url": "/uploads/a1b2c3d4_image.jpg"
}
```

Notes:
- Basic validation checks `Content-Type` and verifies the file using Pillow (PIL). This works on Python 3.13+, where the stdlib `imghdr` module was removed.
- Files are saved into `server/uploads/` with a random prefix to avoid collisions.
- Uploaded files are served at `/uploads/<filename>` (public). Remove or protect this in production.

## CORS
CORS is currently set to allow all origins. In production, restrict it to your Angular app's origin by editing `allow_origins` in `server/main.py`.

## Production considerations
- Put this behind a reverse proxy (e.g. Nginx) for TLS, compression, and caching of static assets.
- Consider using `gunicorn` with `uvicorn.workers.UvicornWorker` for process management.
- Add authentication and size limits for uploads as needed.
- Replace basic `imghdr` validation with more robust checks (e.g., Pillow) if required.
