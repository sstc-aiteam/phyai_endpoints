import numpy as np
import os
import cv2
import logging

from app.services.realsense import realsense_service, RealSenseError
from app.services.hand_eye_calibration import hand_eye_calibration_service, HandEyeCalibrationError
from app.services.yolo_service import yolo_service

logger = logging.getLogger(__name__)

class ObjectDetectionError(Exception):
    pass

class ObjectDetectionService:
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
            
            # Ensure robot is connected (needed for pose)
            hand_eye_calibration_service._connect_robot()
            
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
            
            for box in results.boxes:
                if int(box.cls) == object_class_id:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    pixel_coords = [u, v]

                    # 4. Get Depth at Center (depth is in mm from service)
                    depth_in_meters = depth_image[v, u] * 0.001
                    logger.info(f"Depth at pixel ({u}, {v}): {depth_in_meters:.3f}m")

                    if depth_in_meters == 0:
                        dist_subset = []
                        for i in range(-2, 3):
                            for j in range(-2, 3):
                                if 0 <= v+j < depth_image.shape[0] and 0 <= u+i < depth_image.shape[1]:
                                    d_mm = depth_image[v+j, u+i]
                                    if d_mm > 0: dist_subset.append(d_mm)
                        if dist_subset:
                            depth_in_meters = np.median(dist_subset) * 0.001
                    
                    if depth_in_meters > 0:
                        # 5. Deproject: Pixel -> Camera 3D
                        p_cam = realsense_service.deproject_pixel_to_point([u, v], depth_in_meters)
                        p_cam_homog = np.array(p_cam + [1.0])

                        # 6. Transform: Camera -> Wrist -> Base
                        R_gripper2base, t_gripper2base_vec = hand_eye_calibration_service.get_robot_pose()
                        T_wrist_base = np.eye(4)
                        T_wrist_base[:3, :3] = R_gripper2base
                        T_wrist_base[:3, 3] = t_gripper2base_vec
                        
                        p_base = T_wrist_base @ T_cam_wrist @ p_cam_homog
                        object_coords = p_base[:3]
                        
                        # Draw on the image for the successful detection
                        cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)

                        logger.info(f"Found {object_name} at pixel ({u}, {v}) with depth {depth_in_meters:.3f}m.")
                        logger.info(f"Calculated base coordinates: {object_coords.tolist()}")
                        break 
            
            return object_coords, pixel_coords, depth_in_meters, color_image

        except (HandEyeCalibrationError, RealSenseError) as e:
            raise ObjectDetectionError(f"Error during object detection: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in locate_object_in_base: {e}", exc_info=True)
            raise ObjectDetectionError(f"An unexpected error occurred: {e}") from e

    def grasp_bottle(self):
        """Finds a bottle and executes a grasp motion sequence with the robot."""
        BOTTLE_CLASS_ID = 39 # 'bottle' in COCO dataset
        bottle_xyz, _, _, _ = self.locate_object_in_base(BOTTLE_CLASS_ID, "bottle")

        if bottle_xyz is not None:
            rtde_r = hand_eye_calibration_service.rtde_r
            rtde_c = hand_eye_calibration_service.rtde_c
            if not rtde_r or not rtde_c:
                raise ObjectDetectionError("Robot interfaces are not available.")

            current_ori = rtde_r.getActualTCPPose()[3:]
            approach_pose = [bottle_xyz[0], bottle_xyz[1], bottle_xyz[2] + 0.1] + current_ori
            grasp_pose = [bottle_xyz[0], bottle_xyz[1], bottle_xyz[2]] + current_ori
            
            logger.info(f"Executing grasp sequence. Approach: {approach_pose}, Grasp: {grasp_pose}")
            # rtde_c.moveL(approach_pose, 0.5, 0.2)
            # rtde_c.moveL(grasp_pose, 0.1, 0.05)
            # logger.info("Gripper action placeholder: Closing gripper...")
            # rtde_c.moveL(approach_pose, 0.5, 0.2) # Retract
            rtde_c.moveL(approach_pose, 0.1, 0.2)
            rtde_c.moveL(grasp_pose, 0.1, 0.05)
            logger.info("Gripper action placeholder: Closing gripper...")
            rtde_c.moveL(approach_pose, 0.1, 0.2) # Retract

            return grasp_pose
        else:
            return None

object_detection_service = ObjectDetectionService()