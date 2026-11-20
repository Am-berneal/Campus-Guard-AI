from fastapi import FastAPI

from api.routers.videos import router as video_router
from api.routers import analysis

app = FastAPI(
    title="CampusSentinel AI Service",
    description="AI video analysis service for campus safety monitoring",
    version="1.0.0",
)

app.include_router(video_router)
app.include_router(analysis.router)

# Load available AI models in the background during startup.
analysis.preload_in_background()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "campussentinel-ai",
        "message": "AI service is running"
    }