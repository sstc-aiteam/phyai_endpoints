import numpy as np
import os
import cv2
import logging
import time

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

    ## 
    # Main method to locate object and calculate its pose in the robot's base frame
    # param object_class_id: The class ID of the object to detect (e.g., 39 for 'bottle' in COCO), 
    #     see reference: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml#L18
    # param object_name: A human-readable name for the object (used for logging and error messages)
    ## 
    def locate_object_in_base(self, object_class_id: int, object_name: str):
        """
        Locates a specified object using YOLO and calculates its pose in the robot's base frame.
        """
        if not os.path.exists(settings.CALIBRATION_FILE):
            raise ObjectDetectionError(f"Calibration file '{settings.CALIBRATION_FILE}' not found. Please run a hand-eye calibration first.")

        try:
            # 1. Get services ready
            model = yolo_service.get_model()
            T_cam_wrist = np.load(settings.CALIBRATION_FILE)
            
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

    # ------------------------------------------------------------------ #
    # Helper methods                                                     #
    # ------------------------------------------------------------------ #
    def _get_vertical_down_orientation(self):
        """
        Returns the Euler rotation vector (rx, ry, rz) for the TCP
        to point straight down (Z-axis of TCP aligned with -Z of base frame).
        For UR robots: [pi, 0, 0] means the tool Z points downward.
        """
        return [np.pi, 0.0, 0.0]

    def _get_best_box(self, results, object_class_id: int):
        """Return the first detected box matching class id, or None."""
        for box in results.boxes:
            if int(box.cls) == object_class_id:
                return box
        return None

    def _get_box_center(self, box) -> tuple[int, int]:
        """Return (u, v) pixel center of a YOLO bounding box."""
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def center_on_object(self, object_class_id: int, object_name: str, 
                            max_iterations=12, tolerance_pixels=10):
        
        GAIN_Z = 0.05       # Z 軸移動增益 (公尺/像素比)
        GAIN_W1 = 0.4       # Wrist 1 補償增益 (弧度)
        GAIN_W2 = 0.15      # Wrist 2 水平增益

        logger.info(f"開始執行 '{object_name}' 的精密笛卡兒對準...")

        model = yolo_service.get_model()
        if not realsense_service.is_initialized:
            realsense_service._initialize()
        
        hand_eye_calibration_service._connect_robot()
        rtde_c = hand_eye_calibration_service.rtde_c
        rtde_r = hand_eye_calibration_service.rtde_r
        
        center_u = settings.RS_STREAM_WIDTH // 2
        center_v = settings.RS_STREAM_HEIGHT // 2

        # ------------------------------------------------------------------ #
        # Phase 1: 垂直對齊 (固定 X,Y, 調整 Z + Wrist 1)
        # ------------------------------------------------------------------ #
        for i in range(max_iterations):
            color_image, _ = realsense_service.capture_images()
            results = model(color_image, verbose=False)[0]
            best_box = self._get_best_box(results, object_class_id)

            if best_box is None:
                logger.warning("目標丟失，停止 Phase 1")
                break

            u, v = self._get_box_center(best_box)
            error_v = v - center_v
            
            if abs(error_v) < tolerance_pixels:
                logger.info("垂直對準完成。")
                break

            # 獲取當前位姿 (Cartesian) 與 關節 (Joints)
            current_pose = rtde_r.getActualTCPPose() # [x, y, z, rx, ry, rz]
            current_joints = list(rtde_r.getActualQ()) # [q0, q1, q2, q3, q4, q5]
            logger.info(f"Phase 1 Iteration {i+1} current pose: {current_pose}")

            # 計算標準化誤差
            v_norm = error_v / center_v

            # 1. 調整 Z 軸 (Base 座標系)
            # error_v < 0 代表物體在影像上方 -> 手臂應向下移動 (Z 減少)
            # error_v > 0 代表物體在影像下方 -> 手臂應向上移動 (Z 增加)
            z_step = v_norm * GAIN_Z 
            current_pose[2] += z_step

            # 2. 計算 Wrist 1 補償 (Joint 3)
            # 手臂下降時，Wrist 1 通常需要向下壓 (增加或減少視安裝方向而定)
            w1_step = v_norm * GAIN_W1
            
            # 3. 執行複合移動
            # 我們先用 IK 解算出新的關節空間，再手動修改 Wrist 1
            # 提供 current_joints 作為 q_near 參數，可以讓 IK 找到最接近當前姿態的解，
            # 這能穩定 "wrist1 的朝向"，避免過程中發生非預期的手腕翻轉。
            try:
                target_joints = rtde_c.getInverseKinematics(current_pose, current_joints)
                target_joints[3] -= w1_step # 直接補償 Wrist 1
                rtde_c.moveJ(target_joints, speed=0.1, acceleration=0.2)
                logger.info(f"Phase 1 Iteration {i+1} moving to pose: {current_pose} with Z step: {z_step:.4f}m")
            except RuntimeError as e:
                logger.warning(f"垂直對準步驟找不到 IK 解：{e}")
                # 如果找不到解，可能是目標姿態無法到達，終止對準
                break
            
            time.sleep(1)  # 等待機械臂穩定，並給相機時間更新畫面

        # ------------------------------------------------------------------ #
        # Phase 2: 水平對齊 (Wrist 2 + Wrist 3)
        # ------------------------------------------------------------------ #
        logger.info("[Phase 2] 調整水平偏角...")
        for i in range(max_iterations):
            color_image, _ = realsense_service.capture_images()
            results = model(color_image, verbose=False)[0]
            best_box = self._get_best_box(results, object_class_id)

            if best_box is None: break

            u, v = self._get_box_center(best_box)
            error_u = u - center_u
            
            if abs(error_u) < tolerance_pixels:
                break

            current_joints = list(rtde_r.getActualQ())
            u_norm = error_u / center_u
            u_delta = u_norm * GAIN_W2

            # 調整 Wrist 2 轉向物體
            current_joints[4] += u_delta
            # Wrist 3 同步補償，保持末端工具（夾爪）相對於物體的旋轉角度不變
            current_joints[5] -= u_delta 

            rtde_c.moveJ(current_joints, speed=0.2, acceleration=0.4)
            time.sleep(0.8)    

    def grasp_bottle(self):
        """Finds a bottle, centers the gripper over it, and executes a grasp motion sequence."""        
        if not os.path.exists(settings.CALIBRATION_FILE):
            raise ObjectDetectionError(f"Calibration file '{settings.CALIBRATION_FILE}' not found. Please run a hand-eye calibration first.")

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
            return None # Centering failed or object not found

object_detection_service = ObjectDetectionService()