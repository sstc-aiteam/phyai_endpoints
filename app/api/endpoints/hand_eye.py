from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
import numpy as np
import logging
import os

from app.services.hand_eye_calibration import hand_eye_calibration_service, HandEyeCalibrationError
from app.services.realsense import realsense_service, RealSenseError
from app.core.config import settings


router = APIRouter()
logger = logging.getLogger(__name__)

# --- Pydantic Models for API ---

class CalibrationConfig(BaseModel):
    robot_ip: str = Field(settings.ROBOT_IP, description="IP address of the UR robot.")
    checkerboard_size: tuple[int, int] = Field((8, 5), description="Inner corners of the checkerboard (width, height).")
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

class VerifyPointRequest(BaseModel):
    u: int = Field(..., description="The horizontal pixel coordinate (x-axis) of the point to verify.")
    v: int = Field(..., description="The vertical pixel coordinate (y-axis) of the point to verify.")

class VerifyPointResponse(BaseModel):
    message: str
    target_robot_pose: list[float] = Field(..., description="The calculated target pose [x, y, z, rx, ry, rz] for the robot's TCP.")
    point_in_camera_frame: list[float] = Field(..., description="The 3D coordinates [x, y, z] of the selected pixel in the camera's frame.")


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
        hand_eye_calibration_service.rtde_c = None
        
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
                "robot_ip": settings.ROBOT_IP,
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
             hand_eye_calibration_service.rtde_c = None

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

@router.post("/verify_point", response_model=VerifyPointResponse, summary="Verify calibration by calculating a robot target pose")
def verify_point(req: VerifyPointRequest):
    """
    Verifies the hand-eye calibration by performing a forward calculation.

    Given a 2D pixel coordinate (u, v) from the camera's image, this endpoint:
    1. Reads the current depth value at that pixel.
    2. Converts the 2D pixel + depth into a 3D point in the camera's coordinate system.
    3. Loads the saved hand-eye transformation matrix (`handeye_result.npy`).
    4. Reads the robot's current pose.
    5. Calculates the world coordinates (in the robot's base frame) of the 3D point.

    The returned `target_robot_pose` can be used to command the robot. If the calibration
    is correct, the robot's tool tip will move to the physical point that corresponds
    to the selected pixel.
    """
    CALIBRATION_FILE = "handeye_result.npy"
    if not os.path.exists(CALIBRATION_FILE):
        raise HTTPException(status_code=404, detail=f"Calibration file '{CALIBRATION_FILE}' not found. Please run a calibration first.")

    try:
        # 1. Load the hand-eye transformation matrix
        T_cam2gripper = np.load(CALIBRATION_FILE)

        # 2. Capture image and get depth for the pixel
        _color_image, depth_image = realsense_service.capture_images()
        
        height, width = depth_image.shape
        if not (0 <= req.u < width and 0 <= req.v < height):
            raise HTTPException(status_code=400, detail=f"Pixel coordinates ({req.u}, {req.v}) are out of image bounds ({width}x{height}).")

        # Depth from realsense is in mm, convert to meters for deprojection
        depth_in_meters = depth_image[req.v, req.u] * 0.001
        
        if depth_in_meters == 0:
            raise HTTPException(status_code=400, detail=f"Depth at pixel ({req.u}, {req.v}) is zero. Cannot calculate 3D point. Please select another point.")

        # 3. Deproject pixel to a 3D point in the camera's frame
        point_in_cam_frame = realsense_service.deproject_pixel_to_point([req.u, req.v], depth_in_meters)
        P_cam_homogeneous = np.array(point_in_cam_frame + [1.0])

        # 4. Get current robot pose
        R_gripper2base, t_gripper2base_vec = hand_eye_calibration_service.get_robot_pose()
        pose_vector = hand_eye_calibration_service.rtde_r.getActualTCPPose()
        current_orientation_rv = pose_vector[3:]

        T_gripper2base = np.eye(4)
        T_gripper2base[:3, :3] = R_gripper2base
        T_gripper2base[:3, 3] = t_gripper2base_vec

        # 5. Calculate the target point in the robot's base frame: P_base = T_gripper2base @ T_cam2gripper @ P_cam
        P_base_homogeneous = T_gripper2base @ T_cam2gripper @ P_cam_homogeneous
        target_xyz = P_base_homogeneous[:3]

        # Combine the calculated XYZ with the robot's current orientation for a stable move
        target_pose = target_xyz.tolist() + current_orientation_rv

        return {
            "message": "Successfully calculated target robot pose for verification.",
            "point_in_camera_frame": point_in_cam_frame,
            "target_robot_pose": target_pose
        }
    except (HandEyeCalibrationError, RealSenseError) as e:
        logger.error(f"Verification failed due to calibration/camera error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred during verification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")