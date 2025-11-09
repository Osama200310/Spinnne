from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import URL
from pathlib import Path
import secrets
import sqlite3
from datetime import datetime, timezone
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
DB_FILE = BASE_DIR / "app.db"
API_PREFIX = "/api"

# Ensure directories exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Simple Angular Host + Image Upload API")


def init_db():
    """Create a small SQLite database with a simple uploads table if it doesn't exist."""
    conn = sqlite3.connect(str(DB_FILE))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                size INTEGER NOT NULL,
                content_type TEXT,
                url TEXT,
                score TEXT,
                objects TEXT,
                saved_at TEXT NOT NULL
            )
            """
        )
        # Best-effort migration for existing DBs created before `score` column existed
        try:
            cur.execute("ALTER TABLE uploads ADD COLUMN score TEXT")
        except Exception:
            # Column likely exists already; ignore
            pass
        # Best-effort migration for `objects` column
        try:
            cur.execute("ALTER TABLE uploads ADD COLUMN objects TEXT")
        except Exception:
            # Column likely exists already; ignore
            pass
        conn.commit()
    finally:
        conn.close()


def log_upload(filename: str, size: int, content_type: str | None, url: str, score: str | None, objects: str | None) -> None:
    """Insert a row into the uploads table. Failing to log should not break the request."""
    try:
        conn = sqlite3.connect(str(DB_FILE))
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO uploads (filename, size, content_type, url, score, objects, saved_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    filename,
                    int(size),
                    content_type,
                    url,
                    score,
                    objects,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        # Best-effort logging; in a real app consider proper logging instead of print
        print(f"[warn] failed to log upload to DB: {e}")


# Initialize DB on startup
@app.on_event("startup")
async def on_startup():
    init_db()

# CORS: allow localhost and all origins by default (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your Angular app origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class SPARouteFallbackMiddleware(BaseHTTPMiddleware):
    """If a non-API, non-static route is requested, serve index.html.

    This keeps Angular routing working when users refresh deep links.
    """

    async def dispatch(self, request: Request, call_next):
        # Let API and /uploads routes pass through
        if request.url.path.startswith(API_PREFIX) or request.url.path.startswith("/uploads"):
            return await call_next(request)

        # Let actual files under STATIC_DIR be served normally
        potential = STATIC_DIR / request.url.path.lstrip("/")
        if potential.exists():
            return await call_next(request)

        # For anything else under root, serve index.html if it exists
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)

        # If no index.html yet (e.g., before placing Angular build), show tip
        return JSONResponse(
            status_code=404,
            content={
                "error": "Not Found",
                "message": "Static app not found. Place your Angular build output (index.html, assets) into the 'server/static' directory.",
            },
        )


app.add_middleware(SPARouteFallbackMiddleware)


@app.get(f"{API_PREFIX}/health")
async def health():
    return {"status": "ok"}


@app.post(f"{API_PREFIX}/upload")
async def upload_image(file: UploadFile = File(...), score: str | None = Form(None), objects: str | None = Form(None)):
    # Basic content type validation
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Generate a random file name to avoid collisions
    random_token = secrets.token_hex(8)
    original_name = Path(file.filename or "upload").name
    dest_tmp = UPLOAD_DIR / f"{random_token}_{original_name}"

    # Save to disk
    try:
        with dest_tmp.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    finally:
        await file.close()

    # Verify the file is indeed an image using Pillow and detect format
    try:
        # Verify image integrity (does not decode entire image)
        with Image.open(dest_tmp) as img:
            img.verify()
        # Re-open to read format after verify()
        with Image.open(dest_tmp) as img2:
            detected_format = (img2.format or "").upper()
    except Exception:
        dest_tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File is not a valid image.")

    # Map PIL format to a conventional file extension
    format_to_ext = {
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "PNG": ".png",
        "GIF": ".gif",
        "WEBP": ".webp",
        "BMP": ".bmp",
        "TIFF": ".tiff",
        "ICO": ".ico",
    }
    ext = format_to_ext.get(detected_format, f".{detected_format.lower()}" if detected_format else "")

    final_path = dest_tmp
    if ext and not dest_tmp.name.lower().endswith(ext):
        final_path = dest_tmp.with_suffix(ext)
        dest_tmp.rename(final_path)

    # Prepare response fields
    resp = {
        "filename": final_path.name,
        "size": final_path.stat().st_size,
        "content_type": file.content_type,
        "url": f"/uploads/{final_path.name}",
        "score": score,
        "objects": objects,
    }

    # Log to SQLite (best-effort)
    log_upload(
        filename=resp["filename"],
        size=resp["size"],
        content_type=resp["content_type"],
        url=resp["url"],
        score=score,
        objects=objects,
    )

    return resp


# List uploads helper and endpoint

def list_uploads(limit: int = 100, offset: int = 0, order: str = "desc"):
    """Return a list of upload rows as dictionaries.

    Args:
        limit: max rows to return (1..1000)
        offset: rows to skip (>=0)
        order: 'asc' or 'desc' for ID ordering
    """
    # Sanitize inputs
    try:
        limit_i = max(1, min(1000, int(limit)))
    except Exception:
        limit_i = 100
    try:
        offset_i = max(0, int(offset))
    except Exception:
        offset_i = 0
    order_norm = "ASC" if str(order).lower() == "asc" else "DESC"

    conn = sqlite3.connect(str(DB_FILE))
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id, filename, size, content_type, url, score, objects, saved_at
            FROM uploads
            ORDER BY id {order_norm}
            LIMIT ? OFFSET ?
            """,
            (limit_i, offset_i),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    keys = ["id", "filename", "size", "content_type", "url", "score", "objects", "saved_at"]
    return [dict(zip(keys, row)) for row in rows]


@app.get(f"{API_PREFIX}/detections")
async def get_uploads(limit: int = 100, offset: int = 0, order: str = "desc"):
    """Return uploads as JSON. Defaults: latest first, limit 100."""
    items = list_uploads(limit=limit, offset=offset, order=order)
    return items


# Optionally serve uploaded files (use with caution; consider auth in production)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Mount static files at root AFTER API routes are registered to avoid intercepting /api/*
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

# Uvicorn entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )
