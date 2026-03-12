from fastapi import APIRouter
from app.api.endpoints import camera, hand_eye

api_router = APIRouter()

# Include endpoints, e.g., /api/v1/camera/capture
api_router.include_router(camera.router, prefix="/camera", tags=["camera"])
api_router.include_router(hand_eye.router, prefix="/hand-eye", tags=["Hand-Eye Calibration"])
