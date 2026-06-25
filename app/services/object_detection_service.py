import numpy as np
import os
import cv2
import logging
import time
import math

from app.core.config import settings
from app.services.realsense import realsense_service, RealSenseError
from app.services.hand_eye_calibration import hand_eye_calibration_service, HandEyeCalibrationError
from app.services.yolo_service import bottle_yolo_service, ward_item_yolo_service
from app.services.gripper.robotiq_gripper_control import RobotiqGripper

from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

ROI_X_MIN_RATIO = 0.10
ROI_X_MAX_RATIO = 0.90
ROI_Y_MIN_RATIO = 0.10
ROI_Y_MAX_RATIO = 0.70

class ObjectDetectionError(Exception):
    pass

class ObjectDetectionService:
    def build_detection_roi(self, image):
        height, width = image.shape[:2]
        x1 = int(width * ROI_X_MIN_RATIO)
        x2 = int(width * ROI_X_MAX_RATIO)
        y1 = int(height * ROI_Y_MIN_RATIO)
        y2 = int(height * ROI_Y_MAX_RATIO)
        return x1, y1, x2, y2

    def draw_detection_roi(self, image):
        if image is None:
            return

        roi_x1, roi_y1, roi_x2, roi_y2 = self.build_detection_roi(image)
        cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 1)
        cv2.putText(
            image,
            "ROI",
            (roi_x1 + 6, max(20, roi_y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def draw_view_grid(self, image):
        if image is None:
            return

        height, width = image.shape[:2]
        overlay = image.copy()
        grid_color = (255, 255, 0)
        center_color = (0, 255, 255)

        for x in (width // 3, 2 * width // 3):
            cv2.line(overlay, (x, 0), (x, height), grid_color, 1)
        for y in (height // 3, 2 * height // 3):
            cv2.line(overlay, (0, y), (width, y), grid_color, 1)

        cx, cy = width // 2, height // 2
        cv2.line(overlay, (cx, 0), (cx, height), center_color, 1)
        cv2.line(overlay, (0, cy), (width, cy), center_color, 1)
        cv2.circle(overlay, (cx, cy), 5, center_color, -1)

        cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)

    def calc_yaw_from_bbox_pca(self, image, bbox):
        if image is None or bbox is None:
            return None, None

        height, width = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None, None

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None

        crop_h, crop_w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        border = max(3, int(min(crop_w, crop_h) * 0.04))
        edges[:border, :] = 0
        edges[-border:, :] = 0
        edges[:, :border] = 0
        edges[:, -border:] = 0

        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [
            c for c in contours
            if cv2.contourArea(c) > 80 or cv2.arcLength(c, closed=False) > 40
        ]

        if not contours:
            return None, None

        pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
        pts = pts[
            (pts[:, 0] >= border) &
            (pts[:, 0] < crop_w - border) &
            (pts[:, 1] >= border) &
            (pts[:, 1] < crop_h - border)
        ]

        if len(pts) < 20:
            return None, None

        _mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
        if eigenvalues[1][0] == 0:
            return None, None

        axis_ratio = eigenvalues[0][0] / eigenvalues[1][0]
        if axis_ratio < 1.4:
            logger.info(f"Yaw PCA rejected: weak principal axis ratio={axis_ratio:.2f}")
            return None, None

        vx, vy = eigenvectors[0]

        # Image coordinates: x points right, y points down.
        # Define vertical up/down as 0 deg, right-leaning as positive, left-leaning as negative.
        yaw_deg = math.degrees(math.atan2(vx, vy))

        if yaw_deg > 90:
            yaw_deg -= 180
        elif yaw_deg < -90:
            yaw_deg += 180

        return yaw_deg, math.radians(yaw_deg)

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

    def get_detection_transform_context(self):
        if not os.path.exists(settings.CALIBRATION_FILE):
            raise ObjectDetectionError(f"Calibration file '{settings.CALIBRATION_FILE}' not found. Please run a hand-eye calibration first.")

        try:
            T_cam_wrist = np.load(settings.CALIBRATION_FILE)
            R_gripper2base, t_gripper2base_vec = hand_eye_calibration_service.get_robot_pose()
            arm_joint_info = hand_eye_calibration_service.get_arm_joint_info()
            return T_cam_wrist, R_gripper2base, t_gripper2base_vec, arm_joint_info
        except HandEyeCalibrationError as e:
            raise ObjectDetectionError(f"Error during object detection: {e}") from e

    def locate_box_in_base(
        self,
        box,
        color_image,
        depth_image,
        T_cam_wrist,
        R_gripper2base,
        t_gripper2base_vec,
    ):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        u, v = self._get_box_center(box)
        depth_in_meters, p_cam = self._get_3d_point_from_pixel(depth_image, u, v)
        object_yaw_deg, object_yaw_rad = self.calc_yaw_from_bbox_pca(color_image, bbox)

        object_coords = None
        if p_cam is not None:
            p_cam_homog = np.array(list(p_cam) + [1.0])

            T_wrist_base = np.eye(4)
            T_wrist_base[:3, :3] = R_gripper2base
            T_wrist_base[:3, 3] = t_gripper2base_vec

            p_base = T_wrist_base @ T_cam_wrist @ p_cam_homog
            object_coords = p_base[:3]

        return object_coords, [u, v], bbox, object_yaw_deg, object_yaw_rad, depth_in_meters

    ## 
    # Main method to locate object and calculate its pose in the robot's base frame
    # param object_class_id: The class ID of the object to detect (e.g., 0 for a single-class bottle model),
    #     see reference: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml#L18
    # param object_name: A human-readable name for the object (used for logging and error messages)
    ## 
    def locate_object_in_base(self, object_class_id: int, object_name: str, model=None, color_image=None, depth_image=None):
        """
        Locates a specified object using YOLO and calculates its pose in the robot's base frame.
        """
        if not os.path.exists(settings.CALIBRATION_FILE):
            raise ObjectDetectionError(f"Calibration file '{settings.CALIBRATION_FILE}' not found. Please run a hand-eye calibration first.")

        try:
            # 1. Get services ready
            if model is None:
                model = bottle_yolo_service.get_model()
            T_cam_wrist, R_gripper2base, t_gripper2base_vec, arm_joint_info = self.get_detection_transform_context()
            
            # Ensure camera is ready
            if not realsense_service.is_initialized:
                realsense_service._initialize()

            # 2. Capture frames unless the caller already provided an aligned pair.
            if color_image is None or depth_image is None:
                color_image, depth_image = realsense_service.capture_images()

            detection_image = color_image.copy()
            
            # 3. Run Inference
            results = model(color_image, verbose=False)[0]

            object_coords = None
            pixel_coords = None
            bbox = None
            object_yaw_deg = None
            object_yaw_rad = None
            depth_in_meters = None
            self.draw_view_grid(detection_image)
            self.draw_detection_roi(detection_image)

            for box in self._get_candidate_boxes(results, object_class_id, color_image):
                object_coords, candidate_pixel_coords, candidate_bbox, candidate_yaw_deg, candidate_yaw_rad, candidate_depth = self.locate_box_in_base(
                    box,
                    color_image,
                    depth_image,
                    T_cam_wrist,
                    R_gripper2base,
                    t_gripper2base_vec,
                )
                if object_coords is None:
                    continue

                bbox = candidate_bbox
                pixel_coords = candidate_pixel_coords
                depth_in_meters = candidate_depth
                object_yaw_deg = candidate_yaw_deg
                object_yaw_rad = candidate_yaw_rad

                # Draw on the image for the successful detection
                x1, y1, x2, y2 = bbox
                u, v = pixel_coords
                cv2.rectangle(detection_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(detection_image, (u, v), 5, (0, 0, 255), -1)
                image_h, image_w = detection_image.shape[:2]
                image_center = (image_w // 2, image_h // 2)
                dx_pixels = u - image_center[0]
                dy_pixels = v - image_center[1]
                cv2.line(detection_image, image_center, (u, v), (0, 255, 255), 2)
                cv2.putText(detection_image, f"dx:{dx_pixels} dy:{dy_pixels}", (int(x1), min(image_h - 10, int(y2) + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                if object_yaw_deg is not None:
                    axis_len = int(max(x2 - x1, y2 - y1) * 0.45)
                    dx = math.sin(object_yaw_rad) * axis_len
                    dy = math.cos(object_yaw_rad) * axis_len
                    cv2.line(detection_image, (int(u - dx), int(v - dy)), (int(u + dx), int(v + dy)), (255, 0, 255), 2)
                    cv2.putText(detection_image, f"yaw: {object_yaw_deg:.1f} deg", (int(x1), max(20, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                logger.info(f"Found {object_name} at pixel ({u}, {v}) with depth to gripper {depth_in_meters:.3f}m.")
                logger.info(f"Estimated {object_name} yaw: {object_yaw_deg} degrees.")
                logger.info(f"Calculated base coordinates: {object_coords.tolist()}")
                break

            return t_gripper2base_vec, arm_joint_info, object_coords, pixel_coords, bbox, object_yaw_deg, object_yaw_rad, depth_in_meters, detection_image

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

    def _get_candidate_boxes(self, results, object_class_id: int, image):
        """Return matching boxes inside the ROI, ordered by confidence."""
        roi_x1, roi_y1, roi_x2, roi_y2 = self.build_detection_roi(image)
        candidates = []

        for box in results.boxes:
            if int(box.cls) != object_class_id:
                continue

            u, v = self._get_box_center(box)
            if roi_x1 <= u <= roi_x2 and roi_y1 <= v <= roi_y2:
                confidence = float(box.conf) if box.conf is not None else 1.0
                candidates.append((confidence, box))

        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        return [box for _, box in candidates]

    def _get_best_box(self, results, object_class_id: int, image):
        """Return the highest-confidence matching box inside the ROI, or None."""
        candidates = self._get_candidate_boxes(results, object_class_id, image)
        return candidates[0] if candidates else None

    def _get_box_center(self, box) -> tuple[int, int]:
        """Return (u, v) pixel center of a YOLO bounding box."""
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def adjust_gripper_parallel(self, max_iterations=10):
        """
        Adjusts the robot's wrist 1 joint to make the gripper parallel to the horizontal plane.
        It uses a differential kinematic approach to automatically determine the adjustment direction.
        """
        TOLERANCE_RAD = 0.01  # ~0.6 degrees
        PROBE_DELTA = 0.02    # small wrist1 nudge (rad) used to measure d(ry)/d(wrist1)
        MAX_STEP = 0.15       # clamp per-iteration correction (rad)

        hand_eye_calibration_service._connect_robot()
        rtde_r = hand_eye_calibration_service.rtde_r
        rtde_c = hand_eye_calibration_service.rtde_c

        def get_ry_euler(tcp_pose):
            return R.from_rotvec(tcp_pose[3:6]).as_euler('xyz')[1]

        current_joints = list(rtde_r.getActualQ())
        ry = get_ry_euler(rtde_r.getActualTCPPose())

        if abs(ry) < TOLERANCE_RAD:
            logger.info(f"adjust_gripper_parallel: already parallel (ry={np.degrees(ry):.2f}°)")
            return

        logger.info(f"adjust_gripper_parallel: initial ry={np.degrees(ry):.2f}°, probing wrist1 direction...")

        # Probe: nudge wrist1 by +PROBE_DELTA to measure the local Jacobian d(ry)/d(wrist1)
        probe_joints = current_joints.copy()
        probe_joints[3] += PROBE_DELTA
        rtde_c.moveJ(probe_joints, speed=0.2, acceleration=0.4)
        time.sleep(0.3)
        ry_after_probe = get_ry_euler(rtde_r.getActualTCPPose())
        d_ry_dw1 = (ry_after_probe - ry) / PROBE_DELTA

        # Return to the pre-probe position before starting the correction loop
        rtde_c.moveJ(current_joints, speed=0.2, acceleration=0.4)
        time.sleep(0.3)

        if abs(d_ry_dw1) < 1e-6:
            logger.warning("adjust_gripper_parallel: wrist1 has negligible effect on ry, cannot adjust")
            return

        logger.info(f"adjust_gripper_parallel: d(ry)/d(wrist1)={d_ry_dw1:.4f}, starting correction loop")

        for i in range(max_iterations):
            current_joints = list(rtde_r.getActualQ())
            ry = get_ry_euler(rtde_r.getActualTCPPose())
            logger.info(f"adjust_gripper_parallel iter {i + 1}: ry={np.degrees(ry):.2f}°")

            if abs(ry) < TOLERANCE_RAD:
                logger.info(f"adjust_gripper_parallel: converged (ry={np.degrees(ry):.2f}°) after {i + 1} iters")
                return

            # Newton step: delta_w1 = -ry / d(ry)/d(wrist1), clamped to MAX_STEP
            step = float(np.clip(-ry / d_ry_dw1, -MAX_STEP, MAX_STEP))
            target_joints = current_joints.copy()
            target_joints[3] += step
            rtde_c.moveJ(target_joints, speed=0.2, acceleration=0.4)
            time.sleep(0.3)

        logger.warning(f"adjust_gripper_parallel: did not converge within {max_iterations} iterations")

    def adjust_gripper_vertical_alignment(self, object_class_id: int, max_iterations=10, tolerance_pixels=10):
        """
        Adjusts the robot's Z position to vertically center the specified object in the camera's view.
        It uses a proportional control approach based on the pixel error in the vertical direction.
        """
        GAIN_Z = 0.1        # Z correction gain: how aggressively to move in Z based on vertical pixel error. .
        MAX_Z_STEP = 0.06   # cap per-iteration Z movement to 2cm to stay within reachable workspace

        # -------------------------------------------------------------------------- #
        # Phase 1: Vertical Alignment (Fix X,Y and orientation vectors, only adjust Z)
        # -------------------------------------------------------------------------- #
        # Vertical offset setting: 
        # Because the camera is mounted above the gripper, 
        # when the gripper aligns with the object, the object will appear higher in the image (lower v value). 
        # Setting a negative value can move the target center point up. 
        VERTICAL_OFFSET_PIXELS = -80

        center_v = settings.RS_STREAM_HEIGHT // 2        
        center_v = center_v + VERTICAL_OFFSET_PIXELS

        model = bottle_yolo_service.get_model()
        if not realsense_service.is_initialized:
            realsense_service._initialize()

        hand_eye_calibration_service._connect_robot()
        rtde_r = hand_eye_calibration_service.rtde_r
        rtde_c = hand_eye_calibration_service.rtde_c

        logger.info("[Phase 1] Vertical alignment: Adjusting Z to center object vertically in the image...")
        for i in range(max_iterations):
            color_image, _ = realsense_service.capture_images()
            results = model(color_image, verbose=False)[0]
            best_box = self._get_best_box(results, object_class_id, color_image)

            if best_box is None:
                logger.warning("lost target during vertical alignment, stopping Phase 1.")
                break

            u, v = self._get_box_center(best_box)
            error_v = v - center_v
            
            if abs(error_v) < tolerance_pixels:
                logger.info("Vertical alignment complete within tolerance.")
                break

            # get current pose and joints for IK reference
            current_pose = rtde_r.getActualTCPPose() # [x, y, z, rx, ry, rz]
            current_joints = list(rtde_r.getActualQ()) # [q0, q1, q2, q3, q4, q5]
            logger.info(f"Phase 1 Iteration {i+1} current pose: {current_pose}")

            # calculate the normalized vertical error (v_norm)
            v_norm = error_v / center_v

            # ajust Z based on vertical error: we want to move the Camera up or down to center the object in the image.
            # error_v < 0 代表物體在影像上方 -> 手臂應向下移動 (Z 減少)
            # error_v > 0 代表物體在影像下方 -> 手臂應向上移動 (Z 增加)
            z_step = float(np.clip(-v_norm * GAIN_Z, -MAX_Z_STEP, MAX_Z_STEP))
            logger.info(f"Phase 1 Iteration {i+1} vertical error: {error_v} pixels, object is {'Above' if error_v < 0 else 'Below'} center, applying Z step: {z_step:.4f}m")

            # bisect z_step to find the largest reachable step when exact IK fails
            try:
                MIN_Z_STEP = 0.001  # 1 mm
                step = z_step
                target_joints = None
                approx_pose = None

                for attempt in range(5):
                    candidate_pose = current_pose.copy()
                    candidate_pose[2] += step
                    try:
                        target_joints = rtde_c.getInverseKinematics(candidate_pose, current_joints)
                        approx_pose = candidate_pose
                        break
                    except RuntimeError:
                        step *= 0.5
                        if abs(step) < MIN_Z_STEP:
                            break
                        logger.warning(f"Phase 1 Iteration {i+1} IK failed, reducing Z step to {step:.4f}m (attempt {attempt+1})")

                if approx_pose is None:
                    logger.warning(f"Phase 1 Iteration {i+1} unable to find any reachable pose, stopping vertical alignment")
                    break

                rtde_c.moveJ(target_joints, speed=0.5, acceleration=1.0)
                logger.info(f"Phase 1 Iteration {i+1} moving to pose: {approx_pose} with Z step: {step:.4f}m")
                # wait to ensure the robot has settled and the camera has updated the image before the next iteration
                time.sleep(0.2)  
            except Exception as e:
                logger.error(f"vertical alignment failed with unexpected error: {e}", exc_info=True)
                break

    def adjust_gripper_horizontal_alignment(self, object_class_id: int, max_iterations=10, tolerance_pixels=10):
        """
        Adjusts the robot's wrist 2 joint to horizontally center the specified object in the camera's view.
        It uses a probing method to estimate the Jacobian d(u)/d(wrist2) and then applies a proportional control approach based on the horizontal pixel error.
        """
        PROBE_DELTA_W2 = 0.02  # small wrist2 nudge (rad) to measure d(u)/d(wrist2)
        MAX_W2_STEP = 0.15     # clamp per-iteration wrist2 correction (rad)

        center_u = settings.RS_STREAM_WIDTH // 2

        hand_eye_calibration_service._connect_robot()
        rtde_r = hand_eye_calibration_service.rtde_r
        rtde_c = hand_eye_calibration_service.rtde_c

        model = bottle_yolo_service.get_model()
        if not realsense_service.is_initialized:
            realsense_service._initialize()

        color_image, _ = realsense_service.capture_images()
        results = model(color_image, verbose=False)[0]
        best_box = self._get_best_box(results, object_class_id, color_image)

        if best_box is None:
            logger.warning("[Phase 2] Object not found before probe, skipping horizontal alignment.")
        else:
            u_before, _ = self._get_box_center(best_box)
            pre_probe_joints = list(rtde_r.getActualQ())

            probe_joints = pre_probe_joints.copy()
            probe_joints[4] += PROBE_DELTA_W2
            rtde_c.moveJ(probe_joints, speed=0.2, acceleration=0.4)
            time.sleep(0.3)

            color_image, _ = realsense_service.capture_images()
            results = model(color_image, verbose=False)[0]
            best_box_probe = self._get_best_box(results, object_class_id, color_image)

            rtde_c.moveJ(pre_probe_joints, speed=0.2, acceleration=0.4)
            time.sleep(0.3)

            if best_box_probe is None:
                logger.warning("[Phase 2] Object lost during probe, skipping horizontal alignment.")
            else:
                u_after, _ = self._get_box_center(best_box_probe)
                du_dw2 = (u_after - u_before) / PROBE_DELTA_W2
                logger.info(f"[Phase 2] d(u)/d(wrist2) = {du_dw2:.2f} px/rad")

                if abs(du_dw2) < 1e-3:
                    logger.warning("[Phase 2] Wrist 2 has negligible effect on horizontal pixel position, skipping.")
                else:
                    for i in range(max_iterations):
                        color_image, _ = realsense_service.capture_images()
                        results = model(color_image, verbose=False)[0]
                        best_box = self._get_best_box(results, object_class_id, color_image)

                        if best_box is None:
                            logger.warning("[Phase 2] Lost target during horizontal alignment, stopping.")
                            break

                        u, _ = self._get_box_center(best_box)
                        error_u = u - center_u
                        logger.info(f"[Phase 2] Iter {i+1}: u={u}, error_u={error_u} px ({'Left' if error_u < 0 else 'Right'} of center)")

                        if abs(error_u) < tolerance_pixels:
                            logger.info(f"[Phase 2] Horizontal alignment converged (error={error_u} px) after {i+1} iters.")
                            break

                        current_joints = list(rtde_r.getActualQ())
                        w2_delta = float(np.clip(-error_u / du_dw2, -MAX_W2_STEP, MAX_W2_STEP))
                        logger.info(f"[Phase 2] Iter {i+1}: applying wrist2 delta={np.degrees(w2_delta):.2f}°")

                        current_joints[4] += w2_delta

                        rtde_c.moveJ(current_joints, speed=0.2, acceleration=0.4)
                        time.sleep(0.2)
                    else:
                        logger.warning(f"[Phase 2] Horizontal alignment did not converge within {max_iterations} iterations.")

    def center_on_object(self, object_class_id: int, object_name: str, max_iterations=10, tolerance_pixels=10):
        logger.info(f"Starting centering on {object_name} (class ID {object_class_id}) ...")

        model = bottle_yolo_service.get_model()
        if not realsense_service.is_initialized:
            realsense_service._initialize()

        # ------------------------------------------------------------------ #
        # Phase 0: Initial Horizontal Orientation vectors.
        # ------------------------------------------------------------------ #
        logger.info("Phase 0: Adjusting the Gripper (Camera) orientation to be parallel with the horizontal plane...")
        self.adjust_gripper_parallel(max_iterations)

        # -------------------------------------------------------------------------- #
        # Phase 1: Vertical Alignment (Fix X,Y and orientation vectors, only adjust Z)
        # -------------------------------------------------------------------------- #
        logger.info("Phase 1: Adjusting the Gripper (Camera) Vertical position to center the object in the image...")
        self.adjust_gripper_vertical_alignment(object_class_id, max_iterations, tolerance_pixels)

        # ------------------------------------------------------------------ #
        # Phase 2: Horizontal Alignment (Wrist 2)
        # ------------------------------------------------------------------ #
        logger.info("Phase 2: Adjusting the Gripper (Camera) Horizontal position to center the object in the image...")
        self.adjust_gripper_horizontal_alignment(object_class_id, max_iterations, tolerance_pixels)

    def grasp_bottle(self):
        """Finds a bottle, centers the gripper over it, and executes a grasp motion sequence."""        
        if not os.path.exists(settings.CALIBRATION_FILE):
            raise ObjectDetectionError(f"Calibration file '{settings.CALIBRATION_FILE}' not found. Please run a hand-eye calibration first.")

        BOTTLE_CLASS_ID = settings.BOTTLE_CLASS_ID
        _, _, bottle_xyz, _, _, _, _, _, _ = self.locate_object_in_base(BOTTLE_CLASS_ID, "bottle")

        if bottle_xyz is not None:
            logger.info(f"Attempting to grasp bottle at base coordinates: {bottle_xyz.tolist()}")

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
