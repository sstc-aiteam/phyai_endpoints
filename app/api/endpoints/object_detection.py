from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response
import base64
import cv2
import logging

from app.services.object_detection_service import object_detection_service, ObjectDetectionError

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class LocateResponse(BaseModel):
    message: str
    object_pose_in_base: list[float] | None
    object_pixel_coords: list[int] | None
    detection_image_base64: str | None = None

class GraspResponse(BaseModel):
    message: str
    executed_grasp_pose: list[float] | None

# --- API Endpoints ---
@router.post("/locate-bottle", response_model=LocateResponse, summary="Locate a bottle and return its pose")
def locate_bottle():
    """
    - Captures an image from the RealSense camera.
    - Uses YOLOv8 to detect a 'bottle' (COCO class ID 39).
    - Calculates the 3D position of the bottle in the robot's base frame using the stored hand-eye calibration.
    """
    try:
        BOTTLE_CLASS_ID = 39 # 'bottle' in COCO dataset
        bottle_coords, pixel_coords, detected_image = object_detection_service.locate_object_in_base(BOTTLE_CLASS_ID, "bottle")

        b64_image = None
        if detected_image is not None:
            success, encoded_img = cv2.imencode('.png', detected_image)
            if success:
                b64_image = base64.b64encode(encoded_img).decode('utf-8')

        if bottle_coords is not None:
            return {
                "message": "Bottle located successfully.",
                "object_pose_in_base": bottle_coords.tolist(),
                "object_pixel_coords": pixel_coords,
                "detection_image_base64": b64_image
            }
        else:
            return {
                "message": "Bottle not detected in the current view.",
                "object_pose_in_base": None,
                "object_pixel_coords": pixel_coords,
                "detection_image_base64": b64_image
            }
    except ObjectDetectionError as e:
        logger.error(f"Failed to locate bottle: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /locate-bottle: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
@router.post(
    "/locate-bottle-visual",
    summary="Locate a bottle and return a visual detection image",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Returns the camera image with the bottle detection (if any) drawn on it.",
        }
    },
)
def locate_bottle_visual():
    """
    - Captures an image from the RealSense camera.
    - Uses YOLOv8 to detect a 'bottle' (COCO class ID 39).
    - Returns the captured image with detection results (bounding box and center point) drawn on it.
    """
    try:
        BOTTLE_CLASS_ID = 39 # 'bottle' in COCO dataset
        _, _, detected_image = object_detection_service.locate_object_in_base(BOTTLE_CLASS_ID, "bottle")

        if detected_image is None:
            raise HTTPException(status_code=500, detail="Failed to get an image from the camera service.")

        success, encoded_img = cv2.imencode('.png', detected_image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image to PNG")

        return Response(content=encoded_img.tobytes(), media_type="image/png")
    except ObjectDetectionError as e:
        logger.error(f"Failed to locate bottle for visual: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /locate-bottle-visual: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post("/grasp-bottle", response_model=GraspResponse, summary="Detect a bottle and execute a grasp motion")
def grasp_bottle():
    """
    **WARNING: This endpoint will move the connected robot.**

    This endpoint performs a full sequence:
    1. Locates a bottle using the same logic as `/locate-bottle`.
    2. Calculates approach and grasp poses based on the bottle's location.
    3. Commands the robot to move to an approach position, descend to grasp, and then retract.
    """
    try:
        grasp_pose = object_detection_service.grasp_bottle()
        
        if grasp_pose:
            return {"message": "Grasp sequence executed successfully.", "executed_grasp_pose": grasp_pose}
        else:
            raise HTTPException(status_code=404, detail="Bottle not detected, cannot execute grasp.")

    except ObjectDetectionError as e:
        logger.error(f"Failed to execute grasp: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /grasp-bottle: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")