
from fastapi import FastAPI

from api.routers.videos import router as video_router

app = FastAPI(
    title="CampusSentinel AI Service",
    description="AI video analysis service for campus safety monitoring",
    version="1.0.0",
)

# Register video upload endpoints.
app.include_router(video_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "campussentinel-ai",
        "message": "AI service is running"
    }
