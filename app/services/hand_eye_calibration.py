import numpy as np
import cv2
import cv2.aruco as aruco
import logging
import rtde_control
import rtde_receive

from app.services.realsense import realsense_service, RealSenseError
from app.core.config import settings

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
            cls._instance.robot_ip = settings.ROBOT_IP
            cls._instance.rtde_c = None
            cls._instance.rtde_r = None
            cls._instance.is_robot_connected = False
            cls._instance.clear_points()  # Initialize storage
        return cls._instance

    def _connect_receive(self):
        """Connects to the robot's receive interface if not already connected."""
        if self.is_robot_connected and self.rtde_r and self.rtde_r.isConnected():
            return
        if not self.robot_ip:
            raise RobotConnectionError("Robot IP is not configured. Please call /start first.")

        try:
            logger.info(f"Connecting to robot receive interface at {self.robot_ip}...")
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            if not self.rtde_r.isConnected():
                raise RobotConnectionError("Failed to establish connection with the robot's receive interface.")
            self.is_robot_connected = True
            logger.info("✅ Successfully connected to robot receive interface.")
        except Exception as e:
            self.rtde_r = None
            self.is_robot_connected = False
            raise RobotConnectionError(f"Failed to connect to robot receive interface: {e}") from e

    def _connect_control(self):
        """Connects to the robot's control interface if not already connected."""
        if self.rtde_c:
            return
        if not self.robot_ip:
            raise RobotConnectionError("Robot IP is not configured. Please call /start first.")

        try:
            logger.info(f"Connecting to robot control interface at {self.robot_ip}...")
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            logger.info("✅ Successfully connected to robot control interface.")
        except Exception as e:
            self.rtde_c = None
            raise RobotConnectionError(f"Failed to connect to robot control interface: {e}") from e

    def _connect_robot(self):
        """Connects to both the control and receive interfaces of the robot."""
        self._connect_receive()
        self._connect_control()

    def get_robot_pose(self):
        """Gets the current 4x4 transform of the UR5 flange."""
        self._connect_receive()  # Ensure connection for receiving data
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
            # 1. Get image from RealSense service. This will also initialize the camera if needed.
            color_image, _ = realsense_service.capture_images()
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # 2. Find Checkerboard
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            if not ret:
                raise CalibrationPointError("Checkerboard not found in the image.")

            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 3. Define object points for PnP
            objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
            objp *= square_size

            # 4. Get Camera Intrinsics
            intr = realsense_service.color_intrinsics
            if intr is None:
                # This case should ideally not be reached if capture_images() is successful
                raise RealSenseError("RealSense intrinsics not available even after image capture.")
            mtx = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
            dist = np.array(intr.coeffs)

            # 5. Solve PnP to get target pose relative to camera
            success, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
            if not success:
                raise CalibrationPointError("solvePnP failed to calculate the checkerboard pose.")
            R_cam, _ = cv2.Rodrigues(rvec)

            # 6. Get Robot Pose
            R_robot, t_robot = self.get_robot_pose()

            # 7. Store poses for calibration
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

    def capture_charuco_calibration_point(self, squares_x: int, squares_y: int, square_length: float, marker_length: float, dictionary_name: str):
        """Captures a single calibration point using a ChArUco board."""
        try:
            # 1. Setup ChArUco board
            try:
                # e.g., "DICT_4X4_50"
                aruco_dict_id = getattr(aruco, dictionary_name)
                dictionary = aruco.getPredefinedDictionary(aruco_dict_id)
            except AttributeError:
                raise CalibrationPointError(f"Invalid ArUco dictionary name: {dictionary_name}")

            # For modern OpenCV versions. If using an older version, this might need to be aruco.CharucoBoard_create(...)
            board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
            # For modern OpenCV versions. If using an older version, this might need to be aruco.DetectorParameters_create()
            params = aruco.DetectorParameters()

            # Create CharucoDetector
            charuco_detector = aruco.CharucoDetector(board, detectorParams=params)

            # 2. Get image from RealSense service
            color_image, _ = realsense_service.capture_images()
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # 3. Detect markers and interpolate Charuco corners in one go
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

            # Check if enough ChArUco corners were found
            if charuco_corners is None or len(charuco_corners) < 4:
                raise CalibrationPointError(f"Not enough ChArUco corners found for pose estimation. Found {len(charuco_corners) if charuco_corners is not None else 0}.")

            # 4. Get Camera Intrinsics
            intr = realsense_service.color_intrinsics
            if intr is None:
                raise RealSenseError("RealSense intrinsics not available.")
            mtx = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
            dist = np.array(intr.coeffs)

            # 5. Estimate pose of the ChArUco board
            # The function returns a boolean success value, and the rvec and tvec
            # Install package "opencv-contrib-python" if aruco.estimatePoseCharucoBoard is not available
            rvec = None
            tvec = None
            success, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, mtx, dist, rvec, tvec)
            if not success:
                raise CalibrationPointError("Failed to estimate pose of the ChArUco board.")
            
            R_cam, _ = cv2.Rodrigues(rvec)

            # 6. Get Robot Pose
            R_robot, t_robot = self.get_robot_pose()

            # 7. Store poses for calibration
            self.R_gripper2base.append(R_robot)
            self.t_gripper2base.append(t_robot)
            self.R_target2cam.append(R_cam)
            self.t_target2cam.append(tvec)

            return len(self.R_gripper2base)

        except (RealSenseError, RobotConnectionError) as e:
            raise e  # Re-raise specific, informative errors
        except Exception as e:
            logger.error(f"An unexpected error occurred during ChArUco point capture: {e}", exc_info=True)
            raise CalibrationPointError(f"Failed to capture ChArUco calibration point: {e}") from e

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