import numpy as np
import cv2
import logging
import rtde_receive

from app.services.realsense import realsense_service, RealSenseError

logger = logging.getLogger(__name__)

# --- Custom Exceptions ---
class HandEyeCalibrationError(Exception):
    """Base exception for calibration errors."""
    pass

class RobotConnectionError(HandEyeCalibrationError):
    """Exception for robot connection issues."""
    pass

class CalibrationPointError(HandEyeCalibrationError):
    """Exception for errors during point capture."""
    pass

class HandEyeCalibrationService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(HandEyeCalibrationService, cls).__new__(cls)
            cls._instance.robot_ip = None
            cls._instance.rtde_r = None
            cls._instance.is_robot_connected = False
            cls._instance.clear_points()  # Initialize storage
        return cls._instance

    def _connect_robot(self):
        """Connects to the robot if not already connected."""
        if self.is_robot_connected and self.rtde_r and self.rtde_r.isConnected():
            return
        if not self.robot_ip:
            raise RobotConnectionError("Robot IP is not configured. Please call /start first.")
        try:
            logger.info(f"Connecting to robot at {self.robot_ip}...")
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            if not self.rtde_r.isConnected():
                raise RobotConnectionError("Failed to establish connection with the robot controller.")
            self.is_robot_connected = True
            logger.info("✅ Successfully connected to robot.")
        except Exception as e:
            self.is_robot_connected = False
            raise RobotConnectionError(f"Failed to connect to robot: {e}") from e

    def get_robot_pose(self):
        """Gets the current 4x4 transform of the UR5 flange."""
        self._connect_robot()  # Ensure connection
        try:
            pose = self.rtde_r.getActualTCPPose()
            logger.info(f"Current robot pose: {pose}")
            if not self.rtde_r.isConnected():
                self.is_robot_connected = False
                raise RobotConnectionError("Robot disconnected during pose retrieval.")
            
            t = np.array(pose[:3])
            rv = np.array(pose[3:])
            R, _ = cv2.Rodrigues(rv)
            return R, t
        except Exception as e:
            self.is_robot_connected = False
            raise RobotConnectionError(f"Failed to get robot pose: {e}") from e

    def clear_points(self):
        """Clears all captured calibration points."""
        self.R_gripper2base = []
        self.t_gripper2base = []
        self.R_target2cam = []
        self.t_target2cam = []
        logger.info("Cleared all hand-eye calibration points.")

    def capture_calibration_point(self, checkerboard_size: tuple[int, int], square_size: float):
        """Captures a single calibration point."""
        try:
            # 1. Get image and intrinsics from RealSense service
            if not realsense_service.is_initialized or realsense_service.color_intrinsics is None:
                realsense_service._initialize() # Attempt to initialize if not ready
                if not realsense_service.is_initialized or realsense_service.color_intrinsics is None:
                    raise RealSenseError("RealSense service not ready or intrinsics not available.")
            
            color_image, _ = realsense_service.capture_images()
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # 2. Find Checkerboard
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            if not ret:
                raise CalibrationPointError("Checkerboard not found in the image.")

            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Define object points
            objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
            objp *= square_size

            # 3. Get Camera Intrinsics
            intr = realsense_service.color_intrinsics
            mtx = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
            dist = np.array(intr.coeffs)

            # 4. Solve PnP to get target pose relative to camera
            _, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
            R_cam, _ = cv2.Rodrigues(rvec)

            # 5. Get Robot Pose
            R_robot, t_robot = self.get_robot_pose()

            # 6. Store data (correcting a typo from the original script t_vec -> tvec)
            self.R_gripper2base.append(R_robot)
            self.t_gripper2base.append(t_robot)
            self.R_target2cam.append(R_cam)
            self.t_target2cam.append(tvec)

            return len(self.R_gripper2base)

        except (RealSenseError, RobotConnectionError) as e:
            raise e  # Re-raise specific, informative errors
        except Exception as e:
            logger.error(f"An unexpected error occurred during point capture: {e}", exc_info=True)
            raise CalibrationPointError(f"Failed to capture calibration point: {e}") from e

    def calculate_hand_eye_calibration(self, method=cv2.CALIB_HAND_EYE_TSAI):
        """Performs the hand-eye calibration calculation."""
        num_points = len(self.R_gripper2base)
        if num_points < 3:
            raise HandEyeCalibrationError(f"Not enough points for calibration. Need at least 3, but have {num_points}.")

        logger.info(f"Performing hand-eye calibration with {num_points} points.")
        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                self.R_gripper2base, self.t_gripper2base,
                self.R_target2cam, self.t_target2cam,
                method=method
            )

            # Construct 4x4 Matrix
            T_cam2gripper = np.eye(4)
            T_cam2gripper[:3, :3] = R_cam2gripper
            T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

            return T_cam2gripper
        except cv2.error as e:
            raise HandEyeCalibrationError(f"OpenCV error during calibration: {e}") from e

# Instantiate the service as a singleton
hand_eye_calibration_service = HandEyeCalibrationService()