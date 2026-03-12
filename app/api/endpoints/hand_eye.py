from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
import numpy as np
import logging

from app.services.hand_eye_calibration import hand_eye_calibration_service, HandEyeCalibrationError

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Pydantic Models for API ---

class CalibrationConfig(BaseModel):
    robot_ip: str = Field("192.168.50.75", description="IP address of the UR robot.")
    checkerboard_size: tuple[int, int] = Field((9, 6), description="Inner corners of the checkerboard (width, height).")
    square_size: float = Field(0.025, description="Size of a checkerboard square in meters.")

class StartResponse(BaseModel):
    message: str

class CaptureResponse(BaseModel):
    points_captured: int
    message: str

class CalculationResponse(BaseModel):
    message: str
    transform_matrix: list[list[float]]

class StatusResponse(BaseModel):
    points_captured: int
    robot_ip: str | None
    is_robot_connected: bool

# --- API Endpoints ---

@router.post("/start", response_model=StartResponse, summary="Start a new calibration session")
def start_session(config: CalibrationConfig):
    """
    Initializes or re-initializes a hand-eye calibration session.
    This clears any previously captured points and sets the robot IP for the session.
    """
    try:
        # Configure the service with the robot IP
        hand_eye_calibration_service.robot_ip = config.robot_ip
        # Invalidate old connection object if IP changes
        hand_eye_calibration_service.is_robot_connected = False
        hand_eye_calibration_service.rtde_r = None
        
        hand_eye_calibration_service.clear_points()
        
        return {"message": f"New calibration session started for robot at {config.robot_ip}. Previous points cleared."}
    except Exception as e:
        logger.error(f"Error starting calibration session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/capture", response_model=CaptureResponse, summary="Capture a calibration point")
def capture_point(config: CalibrationConfig = Body(
    ...,
    examples={
        "default": {
            "summary": "Default values",
            "value": {
                "robot_ip": "192.168.50.75",
                "checkerboard_size": [9, 6],
                "square_size": 0.025
            }
        }
    }
)):
    """
    Captures a single hand-eye calibration point.

    This involves:
    1. Finding the checkerboard in the camera's view.
    2. Calculating the checkerboard's pose relative to the camera.
    3. Getting the robot's current tool pose.
    4. Storing both poses for final calculation.

    The robot IP from the last `/start` call is used, but can be overridden here.
    """
    try:
        # If the service doesn't have an IP, or if a new one is provided, set it.
        if not hand_eye_calibration_service.robot_ip or hand_eye_calibration_service.robot_ip != config.robot_ip:
             hand_eye_calibration_service.robot_ip = config.robot_ip
             hand_eye_calibration_service.is_robot_connected = False
             hand_eye_calibration_service.rtde_r = None

        points_count = hand_eye_calibration_service.capture_calibration_point(
            checkerboard_size=config.checkerboard_size,
            square_size=config.square_size
        )
        return {
            "points_captured": points_count,
            "message": f"Successfully captured point {points_count}."
        }
    except HandEyeCalibrationError as e:
        logger.error(f"Calibration error during capture: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during capture: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/calculate", response_model=CalculationResponse, summary="Calculate the hand-eye transformation")
def calculate_calibration(save_to_file: bool = Body(True, embed=True)):
    """
    Performs the hand-eye calibration calculation using all captured points.
    Returns the resulting 4x4 transformation matrix from the camera to the robot's gripper/flange.
    If `save_to_file` is true, the result is also saved to `handeye_result.npy` on the server.
    """
    try:
        transform_matrix = hand_eye_calibration_service.calculate_hand_eye_calibration()
        
        if save_to_file:
            np.save("handeye_result.npy", transform_matrix)
            logger.info("Calibration result saved to handeye_result.npy")

        return {
            "message": f"Hand-eye calibration successful with {len(hand_eye_calibration_service.R_gripper2base)} points.",
            "transform_matrix": transform_matrix.tolist()
        }
    except HandEyeCalibrationError as e:
        logger.error(f"Calibration calculation error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during calculation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.get("/status", response_model=StatusResponse, summary="Get current calibration status")
def get_status():
    """
    Returns the number of points captured so far and the robot connection status.
    """
    count = len(hand_eye_calibration_service.R_gripper2base)
    # Check connection status without trying to connect
    is_connected = hand_eye_calibration_service.is_robot_connected and hand_eye_calibration_service.rtde_r is not None and hand_eye_calibration_service.rtde_r.isConnected()
    return {
        "points_captured": count,
        "robot_ip": hand_eye_calibration_service.robot_ip,
        "is_robot_connected": is_connected
    }

@router.delete("/points", summary="Clear all captured points")
def delete_points():
    """
    Clears all captured calibration points from the current session without changing robot IP.
    """
    hand_eye_calibration_service.clear_points()
    return {"message": "All captured points have been cleared."}