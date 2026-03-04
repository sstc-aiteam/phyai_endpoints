from fastapi import APIRouter, HTTPException
from app.services.realsense import realsense_service

router = APIRouter()

@router.get("/capture", summary="Capture image from RealSense camera")
def capture_image():
    """
    Captures a single frame from the connected RealSense camera and returns it as a Base64-encoded string.
    """
    try:
        b64_image = realsense_service.capture_image_base64()
        return {
            "status": "success",
            "image_format": "jpeg",
            "image_base64": b64_image
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to capture image: {str(e)}")
