import base64
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from app.services.realsense import realsense_service

router = APIRouter()

@router.get("/capture", summary="Capture image from RealSense camera")
def capture():
    """
    Captures a single frame (RGB and Depth) from the connected RealSense camera and returns them as Base64-encoded strings.
    """
    try:
        color_image, depth_image = realsense_service.capture_images()

        # Encode the image to PNG
        success, encoded_img = cv2.imencode('.png', color_image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image to PNG")

        b64_image = base64.b64encode(encoded_img).decode('utf-8')

        # Encode the depth image to PNG (preserves 16-bit data)
        success_depth, encoded_depth = cv2.imencode('.png', depth_image)
        if not success_depth:
            raise HTTPException(status_code=500, detail="Failed to encode depth image to PNG")

        b64_depth = base64.b64encode(encoded_depth).decode('utf-8')

        return {
            "image_format": "png",
            "image_base64": b64_image,
            "depth_format": "png",
            "depth_base64": b64_depth
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to capture image: {str(e)}")

@router.get("/capture_visual", summary="Capture and return a combined RGB and Depth image")
def capture_visual():
    """
    Captures RGB and Depth frames, combines them side-by-side into a single visual image,
    and returns it directly as a PNG image.
    """
    try:
        color_image, depth_image = realsense_service.capture_images()

        # Convert depth image to a visual format (8-bit color)
        # The alpha value is a scaling factor to better visualize the depth.
        depth_colormap = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_GRAY2BGR)

        # Combine images horizontally
        combined_image = np.hstack((color_image, depth_colormap))

        # Encode the combined image to PNG
        success, encoded_img = cv2.imencode('.png', combined_image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image to PNG")

        # Return the image as a response with the correct media type
        return Response(content=encoded_img.tobytes(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to capture image: {str(e)}")
