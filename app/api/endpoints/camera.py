import base64
import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from app.services.realsense import realsense_service

router = APIRouter()

@router.get("/capture", summary="Capture image from RealSense camera")
def capture_image():
    """
    Captures a single frame from the connected RealSense camera and returns it as a Base64-encoded string.
    """
    try:
        color_image, _ = realsense_service.capture_images()

        # Encode the image to JPEG
        success, encoded_img = cv2.imencode('.jpg', color_image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image to JPEG")

        b64_image = base64.b64encode(encoded_img).decode('utf-8')
        return {
            "image_format": "jpeg",
            "image_base64": b64_image
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to capture image: {str(e)}")

@router.get("/capture_visual", summary="Capture and return a JPEG image directly")
def capture_image_jpeg():
    """
    Captures a single frame from the connected RealSense camera and returns it directly as a JPEG image.
    """
    try:
        color_image, _ = realsense_service.capture_images()

        # Encode the image to JPEG
        success, encoded_img = cv2.imencode('.jpg', color_image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image to JPEG")

        # Return the image as a response with the correct media type
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to capture image: {str(e)}")
