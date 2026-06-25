from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.responses import Response
import base64
import cv2
import logging
import numpy as np
from datetime import datetime

from app.core.config import settings
from app.services.object_detection_service import object_detection_service, ObjectDetectionError
from app.services.pointcloud import encode_binary_ply
from app.services.yolo_service import ward_item_yolo_service
from app.services.realsense import realsense_service, RealSenseError


router = APIRouter()
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class LocateResponse(BaseModel):
    message: str
    gripper_translation_vector: list[float] | None = None
    arm_joint_info: list[float] | None = None
    object_pose_in_base: list[float] | None
    object_pixel_coords: list[int] | None
    bbox: list[int] | None = None
    roi_xyxy: list[int] | None = None
    object_yaw_deg: float | None = None
    object_yaw_rad: float | None = None
    depth_in_meters: float | None = None
    detection_image_base64: str | None = None

class GraspResponse(BaseModel):
    message: str
    executed_grasp_pose: list[float] | None

BEST_CLASS_NAMES = [
    'ac_remotecontrol', 'bottle_alcohol_spray', 'cotton_swab', 'cotton_swabs_pp',
    'disposable_mask', 'gauze_pp', 'saline', 'syringe_nipro', 'waterproof_bandages_ppb',
]

class LocateWardItemRequest(BaseModel):
    object_name: str = Field(..., description=f"Name of ward item to detect. Valid values: {BEST_CLASS_NAMES}")

class CenterOnObjectRequest(BaseModel):
    object_class_id: int = Field(settings.BOTTLE_CLASS_ID, description="The class ID of the object to detect.")
    object_name: str = Field("bottle", description="A human-readable name for the object.")
    max_iterations: int = Field(5, description="Maximum number of centering iterations.")
    tolerance_pixels: int = Field(10, description="Pixel tolerance for successful centering.")

class CenterOnObjectResponse(BaseModel):
    message: str
    object_pose_in_base: list[float] | None = None

class DetectedWardItem(BaseModel):
    class_name: str
    bbox: list[int]
    confidence: float
    object_pose_in_base: list[float] | None = None
    object_pixel_coords: list[int] | None = None
    roi_xyxy: list[int] | None = None
    object_yaw_deg: float | None = None
    object_yaw_rad: float | None = None
    depth_in_meters: float | None = None

class DetectAllWardItemsResponse(BaseModel):
    message: str
    detected_items: list[DetectedWardItem]
    detection_image_base64: str | None = None



# --- API Endpoints ---
@router.post("/locate-bottle", response_model=LocateResponse, summary="Locate a bottle and return its pose")
def locate_bottle():
    """
    - Captures an image from the RealSense camera.
    - Uses YOLOv8 to detect a 'bottle' (COCO class ID 39).
    - Calculates the 3D position of the bottle in the robot's base frame using the stored hand-eye calibration.
    """
    try:
        BOTTLE_CLASS_ID = settings.BOTTLE_CLASS_ID
        gripper_vec, arm_joint_info, bottle_coords, pixel_coords, bbox, object_yaw_deg, object_yaw_rad, depth_in_meters, detected_image = object_detection_service.locate_object_in_base(BOTTLE_CLASS_ID, "bottle")

        b64_image = None
        if detected_image is not None:
            success, encoded_img = cv2.imencode('.png', detected_image)
            if success:
                b64_image = base64.b64encode(encoded_img).decode('utf-8')

        gripper_vec_list = gripper_vec.tolist() if gripper_vec is not None else None
        roi_xyxy = (
            list(object_detection_service.build_detection_roi(detected_image))
            if detected_image is not None
            else None
        )

        if bottle_coords is not None:
            return {
                "message": "Bottle located successfully.",
                "gripper_translation_vector": gripper_vec_list,
                "arm_joint_info": arm_joint_info,
                "object_pose_in_base": bottle_coords.tolist(),
                "object_pixel_coords": pixel_coords,
                "bbox": bbox,
                "roi_xyxy": roi_xyxy,
                "object_yaw_deg": object_yaw_deg,
                "object_yaw_rad": object_yaw_rad,
                "depth_in_meters": depth_in_meters,
                "detection_image_base64": b64_image,
            }
        else:
            return {
                "message": "Bottle not detected in the current view.",
                "gripper_translation_vector": gripper_vec_list,
                "arm_joint_info": arm_joint_info,
                "object_pose_in_base": None,
                "object_pixel_coords": pixel_coords,
                "bbox": bbox,
                "roi_xyxy": roi_xyxy,
                "object_yaw_deg": object_yaw_deg,
                "object_yaw_rad": object_yaw_rad,
                "depth_in_meters": depth_in_meters,
                "detection_image_base64": b64_image,
            }
    except ObjectDetectionError as e:
        logger.error(f"Failed to locate bottle: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /locate-bottle: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post(
    "/locate-bottle-visual",
    summary="Locate a bottle and return a visual detection image",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Returns the camera image with the bottle detection (if any) drawn on it.",
        }
    },
)
def locate_bottle_visual():
    """
    - Captures an image from the RealSense camera.
    - Uses YOLOv8 to detect a 'bottle' (COCO class ID 39).
    - Returns the captured image with detection results (bounding box and center point) drawn on it.
    """
    try:
        BOTTLE_CLASS_ID = settings.BOTTLE_CLASS_ID
        _, _, _, _, _, _, _, _, detected_image = object_detection_service.locate_object_in_base(BOTTLE_CLASS_ID, "bottle")

        if detected_image is None:
            raise HTTPException(status_code=500, detail="Failed to get an image from the camera service.")

        success, encoded_img = cv2.imencode('.png', detected_image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image to PNG")

        return Response(content=encoded_img.tobytes(), media_type="image/png")
    except ObjectDetectionError as e:
        logger.error(f"Failed to locate bottle for visual: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /locate-bottle-visual: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post(
    "/locate-bottle-pointcloud",
    summary="Locate a bottle and return its bbox point cloud",
    response_class=Response,
    responses={
        200: {
            "content": {"application/octet-stream": {}},
            "description": "Returns the detected bottle bbox as a colored binary little-endian PLY point cloud.",
        }
    },
)
def locate_bottle_pointcloud(
    depth_margin_m: float | None = Query(
        0.08,
        description="Keep only points within +/- this many meters from the detected center depth. Set to 0 to keep the full bbox.",
    ),
):
    """
    Detects the bottle bbox, then returns the same frame's RealSense point cloud cropped to that
    bbox and filtered by the bottle depth.
    """
    try:
        BOTTLE_CLASS_ID = settings.BOTTLE_CLASS_ID
        color_frame, depth_frame = realsense_service.capture_aligned_frames()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        _, _, bottle_coords, _, bbox, _, _, depth_in_meters, _ = object_detection_service.locate_object_in_base(
            BOTTLE_CLASS_ID,
            "bottle",
            color_image=color_image,
            depth_image=depth_image,
        )
        if bottle_coords is None or bbox is None:
            raise HTTPException(status_code=404, detail="Bottle not detected in the current view.")

        vertices, colors = realsense_service.point_cloud_from_frames(
            color_frame,
            depth_frame,
            bbox=bbox,
            depth_center_m=depth_in_meters,
            depth_margin_m=None if depth_margin_m == 0 else depth_margin_m,
        )
        if len(vertices) == 0:
            raise HTTPException(status_code=500, detail=f"Detected bbox has no valid depth points: {bbox}")

        ply_bytes = encode_binary_ply(vertices, colors)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"bottle_pointcloud_{timestamp}.ply"

        return Response(
            content=ply_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-PointCloud-BBox": ",".join(map(str, bbox)),
                "X-PointCloud-Depth-Meters": f"{depth_in_meters:.6f}",
                "X-PointCloud-Depth-Margin-Meters": "" if depth_margin_m is None else f"{depth_margin_m:.6f}",
            },
        )
    except HTTPException:
        raise
    except ObjectDetectionError as e:
        logger.error(f"Failed to locate bottle point cloud: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /locate-bottle-pointcloud: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post(
    "/locate-bottle-pointcloud-visual",
    summary="Preview the dynamic bottle bbox/depth mask used for point cloud",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Returns a PNG preview of the detected bbox and depth mask used for point cloud generation.",
        }
    },
)
def locate_bottle_pointcloud_visual(
    depth_margin_m: float = Query(
        0.08,
        description="Show points within +/- this many meters from the detected center depth.",
    ),
):
    """
    Shows the dynamic detection bbox and depth-filtered pixels that would become the bottle point cloud.
    """
    try:
        BOTTLE_CLASS_ID = settings.BOTTLE_CLASS_ID
        color_frame, depth_frame = realsense_service.capture_aligned_frames()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        _, _, bottle_coords, _, bbox, _, _, depth_in_meters, _ = object_detection_service.locate_object_in_base(
            BOTTLE_CLASS_ID,
            "bottle",
            color_image=color_image,
            depth_image=depth_image,
        )
        if bottle_coords is None or bbox is None:
            raise HTTPException(status_code=404, detail="Bottle not detected in the current view.")

        height, width = depth_image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        depth_m = depth_image.astype(np.float32) * realsense_service.depth_scale
        mask = np.zeros((height, width), dtype=bool)
        mask[y1:y2, x1:x2] = (
            (depth_m[y1:y2, x1:x2] > 0)
            & (depth_m[y1:y2, x1:x2] >= max(0.0, depth_in_meters - depth_margin_m))
            & (depth_m[y1:y2, x1:x2] <= depth_in_meters + depth_margin_m)
        )

        preview = color_image.copy()
        overlay = preview.copy()
        overlay[mask] = (0, 255, 0)
        cv2.addWeighted(overlay, 0.45, preview, 0.55, 0, preview)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            preview,
            f"depth {depth_in_meters:.3f}m +/- {depth_margin_m:.3f}m",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        success, encoded_img = cv2.imencode(".png", preview)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode mask preview to PNG")

        return Response(content=encoded_img.tobytes(), media_type="image/png")
    except HTTPException:
        raise
    except ObjectDetectionError as e:
        logger.error(f"Failed to locate bottle point cloud visual: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /locate-bottle-pointcloud-visual: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post("/center-on-object", response_model=CenterOnObjectResponse, summary="Center the robot gripper over a detected object")
def center_on_object(req: CenterOnObjectRequest):
    """
    **WARNING: This endpoint will move the connected robot.**

    This endpoint performs a visual servoing sequence to center the robot's TCP over a detected object.
    1. It iteratively captures images and moves the robot in the XY plane until the object is centered in the camera's view.
    2. Once centered, it descends vertically to a specified approach height above the object.
    3. The TCP orientation is kept pointing vertically downwards throughout the process.
    """
    try:
        object_pose = object_detection_service.center_on_object(
            object_class_id=req.object_class_id,
            object_name=req.object_name,
            max_iterations=req.max_iterations,
            tolerance_pixels=req.tolerance_pixels,
        )
        
        if object_pose is not None:
            return {
                "message": f"Successfully centered on '{req.object_name}'.",
                "object_pose_in_base": object_pose.tolist()
            }
        else:
            # Using 404 is reasonable if the process completes but fails to achieve the goal.
            raise HTTPException(status_code=404, detail=f"Failed to center on '{req.object_name}'. Object may not be detectable or centering failed to converge.")

    except ObjectDetectionError as e:
        logger.error(f"Failed to execute centering: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /center-on-object: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")



@router.post("/locate-ward-item", response_model=LocateResponse, summary="Locate a specific ward item using best.pt and return its pose")
def locate_ward_item(req: LocateWardItemRequest):
    """
    - Accepts an `object_name` from: `['ac_remotecontrol', 'bottle_alcohol_spray', 'cotton_swab', 'cotton_swabs_pp', 'disposable_mask', 'gauze_pp', 'saline', 'syringe_nipro', 'waterproof_bandages_ppb']`
    - Captures an image from the RealSense camera.
    - Uses the `ward_item.pt` YOLOv26n model to detect the requested ward item.
    - Calculates the 3D position in the robot's base frame using the stored hand-eye calibration.
    """
    if req.object_name not in BEST_CLASS_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid object_name '{req.object_name}'. Valid values: {BEST_CLASS_NAMES}",
        )

    class_id = BEST_CLASS_NAMES.index(req.object_name)
    model = ward_item_yolo_service.get_model()

    try:
        gripper_vec, arm_joint_info, obj_coords, pixel_coords, bbox, object_yaw_deg, object_yaw_rad, depth_in_meters, detected_image = \
            object_detection_service.locate_object_in_base(class_id, req.object_name, model=model)

        b64_image = None
        if detected_image is not None:
            success, encoded_img = cv2.imencode('.png', detected_image)
            if success:
                b64_image = base64.b64encode(encoded_img).decode('utf-8')

        gripper_vec_list = gripper_vec.tolist() if gripper_vec is not None else None
        roi_xyxy = (
            list(object_detection_service.build_detection_roi(detected_image))
            if detected_image is not None
            else None
        )

        detected = obj_coords is not None
        return {
            "message": f"'{req.object_name}' located successfully." if detected else f"'{req.object_name}' not detected in the current view.",
            "gripper_translation_vector": gripper_vec_list,
            "arm_joint_info": arm_joint_info,
            "object_pose_in_base": obj_coords.tolist() if detected else None,
            "object_pixel_coords": pixel_coords,
            "bbox": bbox,
            "roi_xyxy": roi_xyxy,
            "object_yaw_deg": object_yaw_deg,
            "object_yaw_rad": object_yaw_rad,
            "depth_in_meters": depth_in_meters,
            "detection_image_base64": b64_image,
        }
    except ObjectDetectionError as e:
        logger.error(f"Failed to locate ward item: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /locate-ward-item: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post("/detect-all-ward-items", response_model=DetectAllWardItemsResponse, summary="Detect all ward items in the current camera view")
def detect_all_ward_items():
    """
    - Captures an image from the RealSense camera.
    - Uses the `ward_item.pt` YOLOv26n model to detect all ward item classes.
    - Returns all detected classes, bounding boxes, and an annotated image (base64-encoded PNG).
    """
    try:
        if not realsense_service.is_initialized:
            realsense_service._initialize()

        color_image, depth_image = realsense_service.capture_images()
        model = ward_item_yolo_service.get_model()
        results = model(color_image, verbose=False)[0]
        T_cam_wrist, R_gripper2base, t_gripper2base_vec, _ = object_detection_service.get_detection_transform_context()
        roi_xyxy = list(object_detection_service.build_detection_roi(color_image))

        detection_image = color_image.copy()
        detected_items: list[DetectedWardItem] = []

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id < 0 or cls_id >= len(BEST_CLASS_NAMES):
                continue
            conf = float(box.conf[0].item())
            class_name = BEST_CLASS_NAMES[cls_id]
            obj_coords, pixel_coords, bbox, object_yaw_deg, object_yaw_rad, depth_in_meters = object_detection_service.locate_box_in_base(
                box,
                color_image,
                depth_image,
                T_cam_wrist,
                R_gripper2base,
                t_gripper2base_vec,
            )

            detected_items.append(
                DetectedWardItem(
                    class_name=class_name,
                    bbox=bbox,
                    confidence=round(conf, 4),
                    object_pose_in_base=obj_coords.tolist() if obj_coords is not None else None,
                    object_pixel_coords=pixel_coords,
                    roi_xyxy=roi_xyxy,
                    object_yaw_deg=object_yaw_deg,
                    object_yaw_rad=object_yaw_rad,
                    depth_in_meters=depth_in_meters,
                )
            )

            cv2.rectangle(detection_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.circle(detection_image, (pixel_coords[0], pixel_coords[1]), 5, (0, 0, 255), -1)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(detection_image, label, (bbox[0], max(bbox[1] - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if object_yaw_deg is not None and object_yaw_rad is not None:
                axis_len = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.45)
                dx = np.sin(object_yaw_rad) * axis_len
                dy = np.cos(object_yaw_rad) * axis_len
                cv2.line(
                    detection_image,
                    (int(pixel_coords[0] - dx), int(pixel_coords[1] - dy)),
                    (int(pixel_coords[0] + dx), int(pixel_coords[1] + dy)),
                    (255, 0, 255),
                    2,
                )

        b64_image = None
        success, encoded_img = cv2.imencode('.png', detection_image)
        if success:
            b64_image = base64.b64encode(encoded_img).decode('utf-8')

        msg = f"Detected {len(detected_items)} ward item(s)." if detected_items else "No ward items detected in the current view."
        return DetectAllWardItemsResponse(message=msg, detected_items=detected_items, detection_image_base64=b64_image)

    except RealSenseError as e:
        logger.error(f"Camera error in /detect-all-ward-items: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Camera error: {str(e)}")
    except ObjectDetectionError as e:
        logger.error(f"Failed to locate detected ward items: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /detect-all-ward-items: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post(
    "/detect-all-ward-items-visual",
    summary="Detect all ward items and return a visual detection image",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Returns the camera image with all detected ward items drawn on it.",
        }
    },
)
def detect_all_ward_items_visual():
    """
    - Captures an image from the RealSense camera.
    - Uses the `ward_item.pt` YOLOv26n model to detect all ward item classes.
    - Returns the captured image with bounding boxes and labels drawn on it.
    """
    try:
        if not realsense_service.is_initialized:
            realsense_service._initialize()

        color_image, _ = realsense_service.capture_images()
        model = ward_item_yolo_service.get_model()
        results = model(color_image, verbose=False)[0]

        detection_image = color_image.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id < 0 or cls_id >= len(BEST_CLASS_NAMES):
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            conf = float(box.conf[0].item())
            class_name = BEST_CLASS_NAMES[cls_id]

            cv2.rectangle(detection_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(detection_image, label, (bbox[0], max(bbox[1] - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        success, encoded_img = cv2.imencode('.png', detection_image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image to PNG")

        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except RealSenseError as e:
        logger.error(f"Camera error in /detect-all-ward-items-visual: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Camera error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in /detect-all-ward-items-visual: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post("/grasp-bottle", response_model=GraspResponse, summary="Detect a bottle and execute a grasp motion")
def grasp_bottle():
    """
    **WARNING: This endpoint will move the connected robot.**

    This endpoint performs a full sequence:
    1. Locates a bottle using the same logic as `/locate-bottle`.
    2. Calculates approach and grasp poses based on the bottle's location.
    3. Commands the robot to move to an approach position, descend to grasp, and then retract.
    """
    try:
        grasp_pose = object_detection_service.grasp_bottle()
        
        if grasp_pose:
            return {"message": "Grasp sequence executed successfully.", "executed_grasp_pose": grasp_pose}
        else:
            raise HTTPException(status_code=404, detail="Bottle not detected, cannot execute grasp.")

    except ObjectDetectionError as e:
        logger.error(f"Failed to execute grasp: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /grasp-bottle: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
