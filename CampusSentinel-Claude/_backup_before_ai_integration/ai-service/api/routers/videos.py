
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/videos", tags=["Video Upload"])

# Store uploaded videos inside the AI service.
UPLOAD_DIR = Path(__file__).resolve().parents[2] / "uploads"

# Maximum upload size: 500 MB.
MAX_FILE_SIZE = 500 * 1024 * 1024

# Only these recorded-video formats are accepted.
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Four simulated campus cameras.
CAMERAS = {
    "library": "Outside Library",
    "b_block": "B Block",
    "cos": "COS",
    "kravings": "Kravings",
}


@router.post("/upload")
async def upload_video(
    camera_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Validate and save a recorded video for a simulated camera."""

    # Validate the selected camera.
    if camera_id not in CAMERAS:
        raise HTTPException(
            status_code=400,
            detail="Invalid camera ID. Choose library, b_block, cos or kravings."
        )

    # Validate the video extension.
    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Use MP4, AVI, MOV or MKV."
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Generate a unique filename to avoid collisions.
    upload_id = str(uuid4())
    saved_name = f"{upload_id}{extension}"
    destination = UPLOAD_DIR / saved_name

    total_bytes = 0

    try:
        with destination.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 500 MB upload limit."
                    )

                output.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty."
            )

    except Exception:
        destination.unlink(missing_ok=True)
        raise

    finally:
        await file.close()

    return {
        "status": "uploaded",
        "upload_id": upload_id,
        "camera_id": camera_id,
        "camera_name": CAMERAS[camera_id],
        "original_filename": original_name,
        "saved_filename": saved_name,
        "size_bytes": total_bytes,
        "message": "Video uploaded successfully. AI processing is not connected yet."
    }
