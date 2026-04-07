import numpy as np
import os
import cv2
import logging

from app.core.config import settings
from app.services.realsense import realsense_service, RealSenseError
from app.services.hand_eye_calibration import hand_eye_calibration_service, HandEyeCalibrationError
from app.services.yolo_service import yolo_service
from app.services.gripper.robotiq_gripper_control import RobotiqGripper

logger = logging.getLogger(__name__)

class ObjectDetectionError(Exception):
    pass

class ObjectDetectionService:
    def _get_3d_point_from_pixel(self, depth_image, u, v):
        """
        Calculates the depth and 3D point in the camera frame for a given pixel.
        If depth is 0, it attempts to find a valid depth in a 5x5 neighborhood.

        Returns:
            A tuple of (depth_in_meters, p_cam).
            p_cam will be None if no valid depth is found.
        """
        depth_in_meters = depth_image[v, u] * realsense_service.depth_scale
        logger.info(f"Initial depth at pixel ({u}, {v}): {depth_in_meters:.3f}m")

        if depth_in_meters == 0:
            logger.info("Initial depth is 0, checking neighborhood...")
            dist_subset = []
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if 0 <= v + j < depth_image.shape[0] and 0 <= u + i < depth_image.shape[1]:
                        d_units = depth_image[v + j, u + i]
                        if d_units > 0:
                            dist_subset.append(d_units)
            if dist_subset:
                depth_in_meters = np.median(dist_subset) * realsense_service.depth_scale
                logger.info(f"Found valid median depth in neighborhood: {depth_in_meters:.3f}m")

        p_cam = None
        if depth_in_meters > 0:
            # Deproject: Pixel -> Camera 3D
            p_cam = realsense_service.deproject_pixel_to_point([u, v], depth_in_meters)

        return depth_in_meters, p_cam

    def locate_object_in_base(self, object_class_id: int, object_name: str):
        """
        Locates a specified object using YOLO and calculates its pose in the robot's base frame.
        """
        CALIBRATION_FILE = "handeye_result.npy"
        if not os.path.exists(CALIBRATION_FILE):
            raise ObjectDetectionError(f"Calibration file '{CALIBRATION_FILE}' not found. Please run a hand-eye calibration first.")

        try:
            # 1. Get services ready
            model = yolo_service.get_model()
            T_cam_wrist = np.load(CALIBRATION_FILE)
            
            # Get robot pose (this will also connect to the robot's receive interface if needed)
            R_gripper2base, t_gripper2base_vec = hand_eye_calibration_service.get_robot_pose()
            arm_joint_info = hand_eye_calibration_service.get_arm_joint_info()
            
            # Ensure camera is ready
            if not realsense_service.is_initialized:
                realsense_service._initialize()

            # 2. Capture Frames
            color_image, depth_image = realsense_service.capture_images()
            
            # 3. Run Inference
            results = model(color_image, verbose=False)[0]

            object_coords = None
            pixel_coords = None
            depth_in_meters = None
            depth_gripper2object = None
            
            for box in results.boxes:
                if int(box.cls) == object_class_id:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    pixel_coords = [u, v]

                    # 4. Get Depth and 3D point in camera frame
                    depth_in_meters, p_cam = self._get_3d_point_from_pixel(depth_image, u, v)

                    if p_cam:
                        # 5. Transform: Camera -> Wrist -> Base
                        p_cam_homog = np.array(p_cam + [1.0])

                        T_wrist_base = np.eye(4)
                        T_wrist_base[:3, :3] = R_gripper2base
                        T_wrist_base[:3, 3] = t_gripper2base_vec
                        
                        p_base = T_wrist_base @ T_cam_wrist @ p_cam_homog
                        object_coords = p_base[:3]
                        
                        # Draw on the image for the successful detection
                        cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)

                        logger.info(f"Found {object_name} at pixel ({u}, {v}) with depth to gripper {depth_in_meters:.3f}m.")
                        logger.info(f"Calculated base coordinates: {object_coords.tolist()}")
                        break 
            
            return t_gripper2base_vec, arm_joint_info, object_coords, pixel_coords, depth_in_meters, color_image

        except (HandEyeCalibrationError, RealSenseError) as e:
            raise ObjectDetectionError(f"Error during object detection: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in locate_object_in_base: {e}", exc_info=True)
            raise ObjectDetectionError(f"An unexpected error occurred: {e}") from e

    def grasp_bottle(self):
        """Finds a bottle and executes a grasp motion sequence with the robot."""
        BOTTLE_CLASS_ID = 39 # 'bottle' in COCO dataset
        _, _, bottle_xyz, _, _, _ = self.locate_object_in_base(BOTTLE_CLASS_ID, "bottle")

        if bottle_xyz is not None:
            # Ensure the control interface is connected before sending move commands
            hand_eye_calibration_service._connect_robot()

            rtde_r = hand_eye_calibration_service.rtde_r
            rtde_c = hand_eye_calibration_service.rtde_c
            if not rtde_r or not rtde_c:
                raise ObjectDetectionError("Robot interfaces are not available.")

            # Get the robot's current orientation as the first option.
            # This is often a good starting point if the object is already in view.
            current_pose = rtde_r.getActualTCPPose()
            current_orientation = current_pose[3:]

            # Define several "top-down" orientations to try, as some may be unreachable
            # depending on the target's position (e.g., too close to the base).
            # The values 2.22 are ~pi/sqrt(2), creating a 180-degree rotation around a non-cardinal axis
            # which helps avoid wrist singularity.
            grasp_orientations = [
                current_orientation,
                [np.pi, 0, 0],  # Primary: 180-deg rotation around tool X
                [2.22, 2.22, 0],  # Secondary: Non-singular alternative
                [-2.22, 2.22, 0],  # Tertiary: Another non-singular alternative
            ]

            reachable_approach_pose = None
            ik_solution_approach = None
            reachable_grasp_pose = None
            for orientation in grasp_orientations:
                # Convert orientation to rotation matrix to find the tool's Z-axis in the base frame
                R_wrist2base, _ = cv2.Rodrigues(np.array(orientation))
                tool_z_axis_in_base = R_wrist2base[:, 2]

                # The TCP must be offset back from the object by the length of the gripper (e.g., 10cm)
                # so the gripper tip lands on the object.
                gripper_length = settings.GRIPPER_LEN_OFFSET_IN_METERS
                grasp_xyz = bottle_xyz - (gripper_length * tool_z_axis_in_base)

                # Define approach and grasp poses. The approach is 10cm above the grasp pose in the world Z-axis,
                # preserving the vertical approach motion.
                approach_offset = settings.APPROACH_OFFSET_IN_METERS
                approach_pose = (grasp_xyz + np.array([0, 0, approach_offset])).tolist() + orientation
                grasp_pose = grasp_xyz.tolist() + orientation
                try:
                    # Check if the approach pose is reachable by asking for an IK solution,
                    # guiding it towards our "unwound" q_near preference.
                    ik_solution_approach = rtde_c.getInverseKinematics(approach_pose)

                    # Also check if the grasp pose is reachable from that configuration.
                    # We don't need to store this IK, just confirm it can be found.
                    rtde_c.getInverseKinematics(grasp_pose, ik_solution_approach)

                    reachable_approach_pose = approach_pose
                    reachable_grasp_pose = grasp_pose
                    logger.info(f"Found reachable approach and grasp poses with orientation: {orientation}")
                    logger.info(f"IK solution for approach: {np.round(ik_solution_approach, 2).tolist()}")
                    break  # Found a good pose, exit the loop
                except RuntimeError as e:
                    logger.warning(f"Approach or grasp pose with orientation {orientation} is not reachable: {e}")
                    continue

            if not reachable_approach_pose:
                raise ObjectDetectionError(
                    "Could not find a reachable top-down approach pose for the detected object. "
                    "The object may be too close to the robot base or outside its workspace."
                )

            logger.info(f"Executing grasp sequence. Approach: {reachable_approach_pose}, Grasp: {reachable_grasp_pose}")

            gripper = RobotiqGripper(rtde_c)
            gripper.open()   # Ensure it's open before closing
            logger.info("Gripper action placeholder: Opening gripper...")

            # Use moveJ with the specific IK solution to move to the approach pose.
            # This ensures the robot takes the 'unwound' configuration we selected,
            # giving it more flexibility and avoiding joint limits.
            rtde_c.moveJ(ik_solution_approach, 0.3, 0.7)

            # Use moveL for the final, precise descent and retraction.
            rtde_c.moveL(reachable_grasp_pose, 0.1, 0.5)
            
            gripper.close()  # Close to grasp the bottle
            logger.info("Gripper action placeholder: Closing gripper...")
            
            rtde_c.moveL(reachable_approach_pose, 0.1, 0.5)  # Retract

            return reachable_grasp_pose
        else:
            return None

object_detection_service = ObjectDetectionService()