from fastapi import APIRouter
from app.api.endpoints import camera, hand_eye, object_detection

api_router = APIRouter()

# Include endpoints, e.g., /api/v1/camera/capture
api_router.include_router(camera.router, prefix="/camera", tags=["camera"])
api_router.include_router(hand_eye.router, prefix="/hand-eye", tags=["Hand-Eye Calibration"])
api_router.include_router(object_detection.router, prefix="/detection", tags=["Object Detection"])
