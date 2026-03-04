from fastapi import APIRouter
from app.api.endpoints import camera

api_router = APIRouter()

# Include endpoints, e.g., /api/v1/camera/capture
api_router.include_router(camera.router, prefix="/camera", tags=["camera"])
