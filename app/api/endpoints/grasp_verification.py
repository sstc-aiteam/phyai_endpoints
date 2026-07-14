import base64
import cv2
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.services.grasp_verification_service import (
    GraspVerificationError,
    grasp_verification_service,
)
from app.services.realsense import RealSenseError, realsense_service

router = APIRouter()
logger = logging.getLogger(__name__)


def _capture_and_verify_grasp() -> tuple[dict, str, object]:
    color_image, _ = realsense_service.capture_images()

    success, encoded_img = cv2.imencode(".jpg", color_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image to JPEG")

    image_base64 = base64.b64encode(encoded_img).decode("utf-8")
    verification = grasp_verification_service.verify_grasp(
        image_base64=image_base64,
        image_mime_type="image/jpeg",
    )

    return verification, image_base64, color_image


def _verification_error_response(endpoint_name: str, e: Exception):
    if isinstance(e, RealSenseError):
        logger.error("Failed to capture RealSense image for grasp verification: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    if isinstance(e, GraspVerificationError):
        logger.error("Failed to verify grasp: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=str(e))
    if isinstance(e, HTTPException):
        raise e

    logger.error("Unexpected error in %s: %s", endpoint_name, e, exc_info=True)
    raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post(
    "/verify",
    response_model=bool,
    summary="Return true when the robot gripper successfully grasped an object",
)
def verify_grasp():
    """
    Captures a RealSense color image and returns True if OpenAI vision judges that the
    gripper is holding an object. Otherwise returns False.
    """
    try:
        verification, _, _ = _capture_and_verify_grasp()
        return verification["is_grasped"]
    except Exception as e:
        _verification_error_response("/grasp-verification/verify", e)


@router.post(
    "/verify-visual",
    summary="Verify grasp and return the captured image with result headers",
    response_class=Response,
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Returns the captured image. Result fields are included in response headers.",
        }
    },
)
def verify_grasp_visual():
    """
    Captures a RealSense color image, verifies grasp status with OpenAI vision, and returns
    the image directly so Swagger can display it.
    """
    try:
        verification, _, color_image = _capture_and_verify_grasp()

        visual = color_image.copy()
        status = "GRASPED" if verification["is_grasped"] else "NOT GRASPED"
        color = (0, 180, 0) if verification["is_grasped"] else (0, 0, 255)
        label = f"{status} | confidence={verification['confidence']:.2f}"
        reason = verification["reason"][:140]
        cv2.rectangle(visual, (0, 0), (visual.shape[1], 92), (0, 0, 0), -1)
        cv2.putText(
            visual,
            label,
            (16, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            visual,
            reason,
            (16, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        success, encoded_img = cv2.imencode(".jpg", visual, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image to JPEG")

        return Response(
            content=encoded_img.tobytes(),
            media_type="image/jpeg",
            headers={
                "X-Is-Grasped": str(verification["is_grasped"]).lower(),
                "X-Confidence": f"{verification['confidence']:.4f}",
                "X-Reason": verification["reason"],
            },
        )
    except Exception as e:
        _verification_error_response("/grasp-verification/verify-visual", e)
