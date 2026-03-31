import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import cv2
import rtde_control
import rtde_receive

# --- 1. INITIALIZE YOLO ---
# 'yolov8n.pt' is the Nano model (fastest). It has 'bottle' as class 39.
model = YOLO('yolov8n.pt') 

def get_bottle_pose_in_base(pipeline, intr, T_cam_wrist, rtde_r, rtde_c):
    # 1. Capture and Align Frames
    frames = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        return None

    # Convert to numpy for YOLO
    img = np.asanyarray(color_frame.get_data())

    # 2. Run Inference
    results = model(img, verbose=False)[0]
    
    bottle_coords = None
    for box in results.boxes:
        # Class 39 is 'bottle' in COCO dataset
        if int(box.cls) == 39:
            # Get center of the bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # 3. Get Depth at Center (Median of a small window is safer)
            depth = depth_frame.get_distance(u, v)
            
            # If depth is 0 (hole), try a small 5x5 average
            if depth == 0:
                dist_subset = []
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        d = depth_frame.get_distance(u+i, v+j)
                        if d > 0: dist_subset.append(d)
                depth = np.median(dist_subset) if dist_subset else 0

            if depth > 0:
                # 4. Deproject: Pixel -> Camera 3D
                p_cam = rs.rs2_deproject_pixel_to_point(intr, [u, v], depth)
                p_cam_homog = np.array([p_cam[0], p_cam[1], p_cam[2], 1])

                # 5. Transform: Camera -> Wrist -> Base
                tcp_pose = rtde_r.getActualTCPPose()
                T_wrist_base = rtde_c.poseToMatrix(tcp_pose) 
                
                p_base = T_wrist_base @ T_cam_wrist @ p_cam_homog
                bottle_coords = p_base[:3]
                
                # Visual Feedback (Optional)
                cv2.circle(img, (u, v), 5, (0, 255, 0), -1)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                break # Take the first bottle found

    cv2.imshow("D405 YOLO View", img)
    cv2.waitKey(1)
    return bottle_coords


# --- 1. CONFIGURATION & CALIBRATION ---
ROBOT_IP = "192.168.50.75"  # Update to your UR5 IP
# This matrix (T_cam_wrist) comes from your Hand-Eye Calibration
# It represents the 4x4 transform from the Wrist (Flange) to the Camera Lens
# T_cam_wrist = np.array([
#     [1, 0, 0, 0.05],  # Example: 50mm offset in X
#     [0, 1, 0, -0.02], # Example: -20mm offset in Y
#     [0, 0, 1, 0.10],  # Example: 100mm offset in Z
#     [0, 0, 0, 1]
# ])
CALIB_FILE = "handeye_result.npy"


# --- 2. INITIALIZE HARDWARE ---
# UR5 Interfaces
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

# RealSense D405 Pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Get intrinsics for deprojection (pixels to 3D)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()


# --- 3. EXECUTION SEQUENCE ---
try:
    T_cam_wrist = np.load(CALIB_FILE)

    print("Locating bottle...")
    bottle_xyz = get_bottle_pose_in_base(pipeline, intr, T_cam_wrist, rtde_r, rtde_c)
    
    if bottle_xyz is not None:
        print(f"Bottle found at: {bottle_xyz}")
        
        # Define approach and grasp poses
        # UR poses are usually [x, y, z, rx, ry, rz]
        # We keep the current orientation (rx, ry, rz) but update xyz
        current_ori = rtde_r.getActualTCPPose()[3:]
        
        approach = [bottle_xyz[0], bottle_xyz[1], bottle_xyz[2] + 0.1] + current_ori
        grasp = [bottle_xyz[0], bottle_xyz[1], bottle_xyz[2]] + current_ori
        
        # Move Sequence
        rtde_c.moveL(approach, 0.5, 0.2)  # Approach (10cm above)
        rtde_c.moveL(grasp, 0.1, 0.05)    # Descend slowly
        
        print("Closing gripper...")
        # Add your specific gripper command here (e.g., Robotiq or standard Digital Out)
        
        rtde_c.moveL(approach, 0.5, 0.2) # Retract
    else:
        print("Bottle not detected.")

finally:
    pipeline.stop()
    rtde_c.stopScript()