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
from app.util.annotation import draw_detection_annotation, draw_yaw_annotation, draw_seg_mask_annotation, palette_color
from app.util.pointcloud import encode_binary_ply, transform_camera_points_to_base
from app.services.yolo_service import ward_item_yolo_service, ward_item_seg_yolo_service, bottle_yolo_service
from app.services.realsense import realsense_service, RealSenseError
from app.services.ward_object_pipeline_service import ward_object_pipeline_service
from app.services.ward_object_pipeline.api_output_formatter import compressed_rle_to_binary_mask


router = APIRouter()
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class DetectItem(BaseModel):
    bbox: list[int] | None = None
    object_pose_in_base: list[float] | None = None
    object_pixel_coords: list[int] | None = None
    object_yaw_deg: float | None = None
    object_yaw_rad: float | None = None
    depth_in_meters: float | None = None

class DetectResponse(DetectItem):
    message: str
    gripper_translation_vector: list[float] | None = None
    arm_joint_info: list[float] | None = None
    detection_image_base64: str | None = None

class SegResponse(DetectResponse):
    mask_contour: list[list[int]] | None = None

class WardItem(DetectItem):
    class_name: str
    bbox: list[int]
    confidence: float

class SegWardItem(WardItem):
    mask_contour: list[list[int]] | None = None

class GraspResponse(BaseModel):
    message: str
    executed_grasp_pose: list[float] | None

class LocateWardItemRequest(BaseModel):
    object_name: str = Field(..., description=f"Name of ward item to detect. Valid values: {settings.WARD_ITEM_CLASS_NAMES}")
    depth_offset_m: float | None = Field(
        None, description="Constant offset (meters) added to the measured depth."
    )

class CenterOnObjectRequest(BaseModel):
    object_class_id: int = Field(settings.BOTTLE_CLASS_ID, description="The class ID of the object to detect.")
    object_name: str = Field("bottle", description="A human-readable name for the object.")
    max_iterations: int = Field(5, description="Maximum number of centering iterations.")
    tolerance_pixels: int = Field(10, description="Pixel tolerance for successful centering.")

class CenterOnObjectResponse(BaseModel):
    message: str
    object_pose_in_base: list[float] | None = None

class DetectAllWardItemsResponse(BaseModel):
    message: str
    detected_items: list[WardItem]
    detection_image_base64: str | None = None

class SegAllWardItemsResponse(BaseModel):
    message: str
    detected_items: list[SegWardItem]
    detection_image_base64: str | None = None


POINTCLOUD_OUTPUT_FRAME = "base_link"
POINTCLOUD_SOURCE_FRAME = "camera"


def _pointcloud_depth_margin(depth_margin_m: float | None) -> float | None:
    return None if depth_margin_m == 0 else depth_margin_m


def _to_base_link_pointcloud(vertices: np.ndarray, transform_context) -> np.ndarray:
    T_cam_wrist, R_gripper2base, t_gripper2base_vec, _ = transform_context
    vertices_base = transform_camera_points_to_base(
        vertices,
        T_cam_wrist,
        R_gripper2base,
        t_gripper2base_vec,
    )
    return vertices_base.astype(np.float32, copy=False)


def _pointcloud_ply_response(
    vertices: np.ndarray,
    colors: np.ndarray,
    filename_prefix: str,
    bbox: list[int],
    depth_in_meters: float,
    depth_margin_m: float | None,
    extra_headers: dict[str, str] | None = None,
) -> Response:
    ply_bytes = encode_binary_ply(vertices, colors)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{filename_prefix}_pointcloud_{POINTCLOUD_OUTPUT_FRAME}_{timestamp}.ply"

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-PointCloud-BBox": ",".join(map(str, bbox)),
        "X-PointCloud-Depth-Meters": f"{depth_in_meters:.6f}",
        "X-PointCloud-Depth-Margin-Meters": "" if depth_margin_m is None else f"{depth_margin_m:.6f}",
        "X-PointCloud-Frame": POINTCLOUD_OUTPUT_FRAME,
        "X-PointCloud-Source-Frame": POINTCLOUD_SOURCE_FRAME,
    }
    if extra_headers:
        headers.update(extra_headers)

    return Response(
        content=ply_bytes,
        media_type="application/octet-stream",
        headers=headers,
    )


# --- API Endpoints ---
@router.post("/locate-bottle", response_model=DetectResponse, summary="Locate a bottle and return its pose")
def locate_bottle(
    depth_offset_m: float | None = Query(
        None, description="Constant offset (meters) added to the measured depth."
    ),
):
    """
    - Captures an image from the RealSense camera.
    - Uses YOLOv8 to detect a 'bottle' (COCO class ID 39).
    - Calculates the 3D position of the bottle in the robot's base frame using the stored hand-eye calibration.
    """
    try:
        BOTTLE_CLASS_ID = settings.BOTTLE_CLASS_ID
        bottle_model = bottle_yolo_service.get_model()
        gripper_vec, arm_joint_info, bottle_coords, pixel_coords, bbox, object_yaw_deg, object_yaw_rad, depth_in_meters, detected_image, _ = \
            object_detection_service.locate_object_in_base(BOTTLE_CLASS_ID, "bottle", model=bottle_model, depth_offset_m=depth_offset_m)

        b64_image = None
        if detected_image is not None:
            success, encoded_img = cv2.imencode('.png', detected_image)
            if success:
                b64_image = base64.b64encode(encoded_img).decode('utf-8')

        gripper_vec_list = gripper_vec.tolist() if gripper_vec is not None else None

        if bottle_coords is not None:
            return {
                "message": "Bottle located successfully.",
                "gripper_translation_vector": gripper_vec_list,
                "arm_joint_info": arm_joint_info,
                "object_pose_in_base": bottle_coords.tolist(),
                "object_pixel_coords": pixel_coords,
                "bbox": bbox,
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
def locate_bottle_visual(
    depth_offset_m: float | None = Query(
        None, description="Constant offset (meters) added to the measured depth."
    ),
):
    """
    - Delegates to `locate_bottle()` for full detection logic.
    - Returns the annotated detection image as a PNG.
    """
    result = locate_bottle(depth_offset_m=depth_offset_m)
    if not result.get("detection_image_base64"):
        raise HTTPException(status_code=500, detail="Detection produced no image.")
    image_bytes = base64.b64decode(result["detection_image_base64"])
    return Response(content=image_bytes, media_type="image/png")

@router.post(
    "/locate-bottle-pointcloud",
    summary="Locate a bottle and return its bbox point cloud in base_link",
    response_class=Response,
    responses={
        200: {
            "content": {"application/octet-stream": {}},
            "description": "Returns the detected bottle bbox as a colored binary little-endian PLY point cloud in base_link coordinates.",
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
    bbox, filtered by the bottle depth, and transformed into the robot base frame.
    Note that the point cloud does not consinder depth_offset_m, as the point cloud is generated from the raw depth image.
    """
    try:
        BOTTLE_CLASS_ID = settings.BOTTLE_CLASS_ID
        bottle_model = bottle_yolo_service.get_model()
        color_frame, depth_frame = realsense_service.capture_aligned_frames()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        transform_context = object_detection_service.get_detection_transform_context()

        _, _, bottle_coords, _, bbox, _, _, depth_in_meters, _, _ = object_detection_service.locate_object_in_base(
            BOTTLE_CLASS_ID,
            "bottle",
            bottle_model,
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
            depth_margin_m=_pointcloud_depth_margin(depth_margin_m),
        )
        if len(vertices) == 0:
            raise HTTPException(status_code=500, detail=f"Detected bbox has no valid depth points: {bbox}")

        vertices = _to_base_link_pointcloud(vertices, transform_context)
        return _pointcloud_ply_response(
            vertices,
            colors,
            filename_prefix="bottle",
            bbox=bbox,
            depth_in_meters=depth_in_meters,
            depth_margin_m=depth_margin_m,
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
    summary="Preview the bottle pixels used for the base_link point cloud",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Returns a PNG preview of the detected bbox and depth mask before the PLY point cloud is transformed to base_link.",
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
    Shows the dynamic detection bbox and depth-filtered pixels used before the bottle
    point cloud is transformed into the robot base frame.
    Note that the point cloud does not consinder depth_offset_m, as the point cloud is generated from the raw depth image.
    """
    try:
        BOTTLE_CLASS_ID = settings.BOTTLE_CLASS_ID
        bottle_model = bottle_yolo_service.get_model()
        color_frame, depth_frame = realsense_service.capture_aligned_frames()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        _, _, bottle_coords, _, bbox, _, _, depth_in_meters, _, _ = object_detection_service.locate_object_in_base(
            BOTTLE_CLASS_ID,
            "bottle",
            bottle_model,
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
        if depth_margin_m == 0:
            mask[y1:y2, x1:x2] = depth_m[y1:y2, x1:x2] > 0
        else:
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



@router.post("/detect-ward-item", response_model=DetectResponse, summary="Locate a specific ward item using best.pt and return its pose")
def detect_ward_item(req: LocateWardItemRequest):
    """
    - Accepts an `object_name` from: `['ac_remotecontrol', 'bottle_alcohol_spray', 'cotton_swab', 'cotton_swabs_pp', 'disposable_mask', 'gauze_pp', 'saline', 'syringe_nipro', 'waterproof_bandages_ppb']`
    - Captures an image from the RealSense camera.
    - Uses the `ward_item.pt` YOLOv26n model to detect the requested ward item.
    - Calculates the 3D position in the robot's base frame using the stored hand-eye calibration.
    """
    if req.object_name not in settings.WARD_ITEM_CLASS_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid object_name '{req.object_name}'. Valid values: {settings.WARD_ITEM_CLASS_NAMES}",
        )

    class_id = settings.WARD_ITEM_CLASS_NAMES.index(req.object_name)
    model = ward_item_yolo_service.get_model()

    try:
        gripper_vec, arm_joint_info, obj_coords, pixel_coords, bbox, object_yaw_deg, object_yaw_rad, depth_in_meters, detected_image, _ = \
            object_detection_service.locate_object_in_base(class_id, req.object_name, model=model, depth_offset_m=req.depth_offset_m)

        b64_image = None
        if detected_image is not None:
            success, encoded_img = cv2.imencode('.png', detected_image)
            if success:
                b64_image = base64.b64encode(encoded_img).decode('utf-8')

        gripper_vec_list = gripper_vec.tolist() if gripper_vec is not None else None

        detected = obj_coords is not None
        return {
            "message": f"'{req.object_name}' located successfully." if detected else f"'{req.object_name}' not detected in the current view.",
            "gripper_translation_vector": gripper_vec_list,
            "arm_joint_info": arm_joint_info,
            "object_pose_in_base": obj_coords.tolist() if detected else None,
            "object_pixel_coords": pixel_coords,
            "bbox": bbox,
            "object_yaw_deg": object_yaw_deg,
            "object_yaw_rad": object_yaw_rad,
            "depth_in_meters": depth_in_meters,
            "detection_image_base64": b64_image,
        }
    except ObjectDetectionError as e:
        logger.error(f"Failed to locate ward item: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /detect-ward-item: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post("/detect-all-ward-items", response_model=DetectAllWardItemsResponse, summary="Detect all ward items in the current camera view")
def detect_all_ward_items(
    depth_offset_m: float | None = Query(
        None, description="Constant offset (meters) added to the measured depth."
    ),
):
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

        detection_image = color_image.copy()
        detected_items: list[WardItem] = []

        for box_idx, box in enumerate(results.boxes):
            cls_id = int(box.cls[0].item())
            if cls_id < 0 or cls_id >= len(settings.WARD_ITEM_CLASS_NAMES):
                continue
            conf = float(box.conf[0].item())
            class_name = settings.WARD_ITEM_CLASS_NAMES[cls_id]
            obj_coords, pixel_coords, bbox, object_yaw_deg, object_yaw_rad, depth_in_meters, _ = object_detection_service.locate_box_in_base(
                box,
                color_image,
                depth_image,
                T_cam_wrist,
                R_gripper2base,
                t_gripper2base_vec,
                depth_offset_m=depth_offset_m,
            )

            detected_items.append(
                WardItem(
                    class_name=class_name,
                    bbox=bbox,
                    confidence=round(conf, 4),
                    object_pose_in_base=obj_coords.tolist() if obj_coords is not None else None,
                    object_pixel_coords=pixel_coords,
                    object_yaw_deg=object_yaw_deg,
                    object_yaw_rad=object_yaw_rad,
                    depth_in_meters=depth_in_meters,
                )
            )

            color = palette_color(box_idx)
            label = f"{class_name} {conf:.2f}"
            draw_detection_annotation(detection_image, bbox, pixel_coords, label, color=color, class_name=class_name, skip_classes=settings.ANNOTATION_SKIP_CLASSES)
            draw_yaw_annotation(
                detection_image,
                bbox,
                pixel_coords,
                object_yaw_deg,
                object_yaw_rad,
                show_label=True,
                color=color,
                class_name=class_name,
                skip_classes=settings.ANNOTATION_SKIP_CLASSES,
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
def detect_all_ward_items_visual(
    depth_offset_m: float | None = Query(
        None, description="Constant offset (meters) added to the measured depth."
    ),
):
    """
    - Delegates to `detect_all_ward_items()` for full detection logic.
    - Returns the annotated detection image as a PNG.
    """
    result = detect_all_ward_items(depth_offset_m=depth_offset_m)
    if not result.detection_image_base64:
        raise HTTPException(status_code=500, detail="Detection produced no image.")
    image_bytes = base64.b64decode(result.detection_image_base64)
    return Response(content=image_bytes, media_type="image/png")


@router.post("/segment-ward-item", response_model=SegResponse, summary="Locate a specific ward item using ward_item_seg.pt and return its pose with mask")
def segment_ward_item(req: LocateWardItemRequest):
    """
    - Accepts an `object_name` from the same set as `/detect-ward-item`.
    - Captures an image from the RealSense camera.
    - Uses the `ward_item_seg.pt` segmentation model to detect the requested ward item.
    - Pixel coordinates are derived from the mask centroid for higher accuracy.
    - Returns the 3D pose in the robot's base frame plus the mask contour polygon.
    """
    if req.object_name not in settings.WARD_ITEM_CLASS_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid object_name '{req.object_name}'. Valid values: {settings.WARD_ITEM_CLASS_NAMES}",
        )

    class_id = settings.WARD_ITEM_CLASS_NAMES.index(req.object_name)
    model = ward_item_seg_yolo_service.get_model()

    try:
        gripper_vec, arm_joint_info, obj_coords, pixel_coords, bbox, object_yaw_deg, object_yaw_rad, depth_in_meters, detected_image, mask_contour = \
            object_detection_service.locate_object_in_base(class_id, req.object_name, model=model, depth_offset_m=req.depth_offset_m)

        b64_image = None
        if detected_image is not None:
            success, encoded_img = cv2.imencode('.png', detected_image)
            if success:
                b64_image = base64.b64encode(encoded_img).decode('utf-8')

        gripper_vec_list = gripper_vec.tolist() if gripper_vec is not None else None
        detected = obj_coords is not None
        return SegResponse(
            message=f"'{req.object_name}' located successfully." if detected else f"'{req.object_name}' not detected in the current view.",
            gripper_translation_vector=gripper_vec_list,
            arm_joint_info=arm_joint_info,
            object_pose_in_base=obj_coords.tolist() if detected else None,
            object_pixel_coords=pixel_coords,
            bbox=bbox,
            object_yaw_deg=object_yaw_deg,
            object_yaw_rad=object_yaw_rad,
            depth_in_meters=depth_in_meters,
            detection_image_base64=b64_image,
            mask_contour=mask_contour,
        )
    except ObjectDetectionError as e:
        logger.error(f"Failed to locate ward item (seg): {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /segment-ward-item: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post("/segment-all-ward-items", response_model=SegAllWardItemsResponse, summary="Detect all ward items using ward_item_seg.pt with segmentation masks")
def segment_all_ward_items(
    depth_offset_m: float | None = Query(
        None, description="Constant offset (meters) added to the measured depth."
    ),
):
    """
    - Captures an image from the RealSense camera.
    - Uses the `ward_item_seg.pt` segmentation model to detect all ward item classes.
    - Returns bounding boxes, mask contour polygons, poses, and an annotated image (base64 PNG).
    """
    try:
        if not realsense_service.is_initialized:
            realsense_service._initialize()

        color_image, depth_image = realsense_service.capture_images()
        model = ward_item_seg_yolo_service.get_model()
        results = model(color_image, verbose=False)[0]
        masks = results.masks
        T_cam_wrist, R_gripper2base, t_gripper2base_vec, _ = object_detection_service.get_detection_transform_context()

        detection_image = color_image.copy()
        detected_items: list[SegWardItem] = []

        for box_idx, box in enumerate(results.boxes):
            cls_id = int(box.cls[0].item())
            if cls_id < 0 or cls_id >= len(settings.WARD_ITEM_CLASS_NAMES):
                continue
            conf = float(box.conf[0].item())
            class_name = settings.WARD_ITEM_CLASS_NAMES[cls_id]

            obj_coords, pixel_coords, bbox, object_yaw_deg, object_yaw_rad, depth_in_meters, mask_contour = \
                object_detection_service.locate_box_in_base(
                    box, color_image, depth_image,
                    T_cam_wrist, R_gripper2base, t_gripper2base_vec,
                    masks, box_idx,
                    depth_offset_m=depth_offset_m,
                )

            detected_items.append(
                SegWardItem(
                    class_name=class_name,
                    bbox=bbox,
                    confidence=round(conf, 4),
                    mask_contour=mask_contour,
                    object_pose_in_base=obj_coords.tolist() if obj_coords is not None else None,
                    object_pixel_coords=pixel_coords,
                    object_yaw_deg=object_yaw_deg,
                    object_yaw_rad=object_yaw_rad,
                    depth_in_meters=depth_in_meters,
                )
            )

            color = palette_color(box_idx)
            label = f"{class_name} {conf:.2f}"
            if mask_contour:
                draw_seg_mask_annotation(detection_image, mask_contour, color=color, class_name=class_name, skip_classes=settings.ANNOTATION_SKIP_CLASSES)
            draw_detection_annotation(detection_image, bbox, pixel_coords, label, color=color, class_name=class_name, skip_classes=settings.ANNOTATION_SKIP_CLASSES)
            draw_yaw_annotation(
                detection_image, bbox, pixel_coords, object_yaw_deg, object_yaw_rad, show_label=True, color=color, class_name=class_name, skip_classes=settings.ANNOTATION_SKIP_CLASSES,
            )

        b64_image = None
        success, encoded_img = cv2.imencode('.png', detection_image)
        if success:
            b64_image = base64.b64encode(encoded_img).decode('utf-8')

        msg = f"Detected {len(detected_items)} ward item(s)." if detected_items else "No ward items detected in the current view."
        return SegAllWardItemsResponse(message=msg, detected_items=detected_items, detection_image_base64=b64_image)

    except RealSenseError as e:
        logger.error(f"Camera error in /segment-all-ward-items: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Camera error: {str(e)}")
    except ObjectDetectionError as e:
        logger.error(f"Failed in /segment-all-ward-items: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /segment-all-ward-items: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post(
    "/segment-all-ward-items-visual",
    summary="Detect all ward items (seg) and return annotated image with masks",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Returns the camera image with all detected ward items and segmentation masks drawn on it.",
        }
    },
)
def segment_all_ward_items_visual(
    depth_offset_m: float | None = Query(
        None, description="Constant offset (meters) added to the measured depth."
    ),
):
    """
    - Delegates to `segment_all_ward_items()` for full detection logic.
    - Returns the annotated detection image (with mask overlays) as a PNG.
    """
    result = segment_all_ward_items(depth_offset_m=depth_offset_m)
    if not result.detection_image_base64:
        raise HTTPException(status_code=500, detail="Detection produced no image.")
    image_bytes = base64.b64decode(result.detection_image_base64)
    return Response(content=image_bytes, media_type="image/png")


@router.post(
    "/segment-ward-item-pointcloud",
    summary="Segment a ward item and return its mask point cloud in base_link",
    response_class=Response,
    responses={
        200: {
            "content": {"application/octet-stream": {}},
            "description": "Returns the detected item segmentation mask as a colored binary little-endian PLY point cloud in base_link coordinates.",
        }
    },
)
def segment_ward_item_pointcloud(
    req: LocateWardItemRequest,
    depth_margin_m: float | None = Query(
        0.08,
        description="For segmented point clouds, remove only points farther than center depth + this many meters. Set to 0 to keep the full mask.",
    ),
):
    """
    Segments the ward item using the seg model, then returns the same frame's RealSense
    point cloud cropped to the item's mask contour, with points too far behind the object
    removed, and transformed into the robot base frame.
    Note that the point cloud does not consinder depth_offset_m, as the point cloud is generated from the raw depth image.
    """
    if req.object_name not in settings.WARD_ITEM_CLASS_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid object_name '{req.object_name}'. Valid values: {settings.WARD_ITEM_CLASS_NAMES}",
        )

    class_id = settings.WARD_ITEM_CLASS_NAMES.index(req.object_name)
    model = ward_item_seg_yolo_service.get_model()

    try:
        color_frame, depth_frame = realsense_service.capture_aligned_frames()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        transform_context = object_detection_service.get_detection_transform_context()

        _, _, obj_coords, _, bbox, _, _, depth_in_meters, _, mask_contour = \
            object_detection_service.locate_object_in_base(
                class_id,
                req.object_name,
                model,
                color_image=color_image,
                depth_image=depth_image,
                depth_offset_m=req.depth_offset_m,
            )

        if obj_coords is None or bbox is None:
            raise HTTPException(status_code=404, detail=f"'{req.object_name}' not detected in the current view.")
        if not mask_contour:
            raise HTTPException(status_code=404, detail=f"'{req.object_name}' segmentation mask not detected in the current view.")

        vertices, colors = realsense_service.point_cloud_from_frames(
            color_frame,
            depth_frame,
            depth_center_m=depth_in_meters,
            depth_margin_m=_pointcloud_depth_margin(depth_margin_m),
            mask_contour=mask_contour,
            depth_filter_mode="far_only",
        )
        if len(vertices) == 0:
            raise HTTPException(status_code=500, detail=f"Detected mask has no valid depth points: {bbox}")

        vertices = _to_base_link_pointcloud(vertices, transform_context)
        return _pointcloud_ply_response(
            vertices,
            colors,
            filename_prefix=req.object_name,
            bbox=bbox,
            depth_in_meters=depth_in_meters,
            depth_margin_m=depth_margin_m,
            extra_headers={"X-PointCloud-Mask-Used": "true" if mask_contour else "false"},
        )
    except HTTPException:
        raise
    except ObjectDetectionError as e:
        logger.error(f"Failed to segment item point cloud: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /segment-ward-item-pointcloud: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post(
    "/segment-ward-item-pointcloud-visual",
    summary="Preview the item pixels used for the base_link point cloud",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Returns a PNG preview of the segmentation mask and depth-filtered pixels before the PLY point cloud is transformed to base_link.",
        }
    },
)
def segment_ward_item_pointcloud_visual(
    req: LocateWardItemRequest,
    depth_margin_m: float = Query(
        0.08,
        description="Show mask points after removing only points farther than center depth + this many meters.",
    ),
):
    """
    Shows the seg mask polygon after removing only points too far behind the detected item.
    """
    if req.object_name not in settings.WARD_ITEM_CLASS_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid object_name '{req.object_name}'. Valid values: {settings.WARD_ITEM_CLASS_NAMES}",
        )

    class_id = settings.WARD_ITEM_CLASS_NAMES.index(req.object_name)
    model = ward_item_seg_yolo_service.get_model()

    try:
        color_frame, depth_frame = realsense_service.capture_aligned_frames()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        _, _, obj_coords, pixel_coords, bbox, _, _, depth_in_meters, _, mask_contour = \
            object_detection_service.locate_object_in_base(
                class_id,
                req.object_name,
                model,
                color_image=color_image,
                depth_image=depth_image,
            )

        if obj_coords is None or bbox is None:
            raise HTTPException(status_code=404, detail=f"'{req.object_name}' not detected in the current view.")

        height, width = depth_image.shape[:2]
        depth_m = depth_image.astype(np.float32) * realsense_service.depth_scale

        if mask_contour:
            poly = np.array(mask_contour, dtype=np.int32)
            fill = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(fill, [poly], 1)
            spatial_mask = fill.astype(bool)
        else:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))
            spatial_mask = np.zeros((height, width), dtype=bool)
            spatial_mask[y1:y2, x1:x2] = True

        if depth_margin_m == 0:
            depth_filter = depth_m > 0
        else:
            depth_filter = (
                (depth_m > 0)
                & (depth_m <= depth_in_meters + depth_margin_m)
            )
        active_mask = spatial_mask & depth_filter

        preview = color_image.copy()
        overlay = preview.copy()
        overlay[active_mask] = (0, 255, 0)
        cv2.addWeighted(overlay, 0.45, preview, 0.55, 0, preview)

        if mask_contour:
            cv2.polylines(preview, [np.array(mask_contour, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            label_x, label_y = int(mask_contour[0][0]), int(mask_contour[0][1])
        else:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_x, label_y = x1, y1

        depth_label = (
            f"{req.object_name} full mask"
            if depth_margin_m == 0
            else f"{req.object_name} max depth {depth_in_meters + depth_margin_m:.3f}m"
        )
        cv2.putText(
            preview,
            depth_label,
            (label_x, max(20, label_y - 10)),
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
        logger.error(f"Failed to render segment item point cloud visual: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /segment-ward-item-pointcloud-visual: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post(
    "/segment-unknown-ward-item",
    response_model=SegResponse,
    summary="Locate a specific ward item, including unrecognized ones, using the RF-DETR + SAM2 + DINOv2 pipeline",
)
def segment_unknown_ward_item(req: LocateWardItemRequest):
    """
    - Accepts an `object_name` from the same set as `/detect-ward-item`.
    - Captures an image from the RealSense camera.
    - Runs the ward_object_pipeline (RF-DETR + SAM2 + DINOv2), same as
      `/segment-unknown-all-ward-items`, then returns the highest-confidence
      detection matching `object_name`.
    - Returns the 3D pose in the robot's base frame plus the mask contour polygon.
    """
    if req.object_name not in settings.WARD_ITEM_CLASS_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid object_name '{req.object_name}'. Valid values: {settings.WARD_ITEM_CLASS_NAMES}",
        )

    try:
        if not realsense_service.is_initialized:
            realsense_service._initialize()

        color_image, depth_image = realsense_service.capture_images()
        T_cam_wrist, R_gripper2base, t_gripper2base_vec, arm_joint_info = object_detection_service.get_detection_transform_context()

        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        pipeline = ward_object_pipeline_service.get_pipeline()
        result = pipeline.predict(color_image_rgb, verbose=False)

        detection_image = color_image.copy()
        obj_coords = pixel_coords = bbox = object_yaw_deg = object_yaw_rad = depth_in_meters = mask_contour = None
        best_confidence = -1.0

        for idx, obj in enumerate(result.get("objects", [])):
            logger.info(f"Detected object {idx}: {obj}")
            if obj["class_name"] != req.object_name:
                continue
            confidence = float(result["final_results"][idx].get("rfdetr_confidence", 0.0))
            if confidence <= best_confidence:
                continue

            candidate_bbox = [int(round(v)) for v in obj["bbox"]]
            mask = compressed_rle_to_binary_mask(obj["rle_mask"]).astype(bool)

            candidate_coords, candidate_pixel_coords, candidate_bbox, candidate_yaw_deg, candidate_yaw_rad, candidate_depth, candidate_mask_contour = \
                object_detection_service.locate_mask_in_base(
                    mask, candidate_bbox, color_image, depth_image,
                    T_cam_wrist, R_gripper2base, t_gripper2base_vec,
                    depth_offset_m=req.depth_offset_m,
                )
            if candidate_coords is None:
                continue

            best_confidence = confidence
            obj_coords, pixel_coords, bbox = candidate_coords, candidate_pixel_coords, candidate_bbox
            object_yaw_deg, object_yaw_rad, depth_in_meters, mask_contour = candidate_yaw_deg, candidate_yaw_rad, candidate_depth, candidate_mask_contour

        detected = obj_coords is not None
        if detected:
            color = palette_color(0)
            label = f"{req.object_name} {best_confidence:.2f}"
            if mask_contour:
                draw_seg_mask_annotation(detection_image, mask_contour, color=color, class_name=req.object_name, skip_classes=settings.ANNOTATION_SKIP_CLASSES)
            draw_detection_annotation(detection_image, bbox, pixel_coords, label, color=color, class_name=req.object_name, skip_classes=settings.ANNOTATION_SKIP_CLASSES)
            draw_yaw_annotation(
                detection_image, bbox, pixel_coords, object_yaw_deg, object_yaw_rad, show_label=True, color=color, class_name=req.object_name, skip_classes=settings.ANNOTATION_SKIP_CLASSES,
            )

        b64_image = None
        success, encoded_img = cv2.imencode('.png', detection_image)
        if success:
            b64_image = base64.b64encode(encoded_img).decode('utf-8')

        if not result.get("success", True):
            msg = result.get("reason", "Pipeline failed.")
        elif detected:
            msg = f"'{req.object_name}' located successfully."
        else:
            msg = f"'{req.object_name}' not detected in the current view."

        return SegResponse(
            message=msg,
            gripper_translation_vector=t_gripper2base_vec.tolist() if t_gripper2base_vec is not None else None,
            arm_joint_info=arm_joint_info,
            object_pose_in_base=obj_coords.tolist() if detected else None,
            object_pixel_coords=pixel_coords,
            bbox=bbox,
            object_yaw_deg=object_yaw_deg,
            object_yaw_rad=object_yaw_rad,
            depth_in_meters=depth_in_meters,
            detection_image_base64=b64_image,
            mask_contour=mask_contour,
        )
    except RealSenseError as e:
        logger.error(f"Camera error in /segment-unknown-ward-item: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Camera error: {str(e)}")
    except ObjectDetectionError as e:
        logger.error(f"Failed to locate ward item (unknown pipeline): {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /segment-unknown-ward-item: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post(
    "/segment-unknown-ward-item-pointcloud",
    summary="Segment a ward item (including unrecognized ones) and return its mask point cloud in base_link",
    response_class=Response,
    responses={
        200: {
            "content": {"application/octet-stream": {}},
            "description": "Returns the detected item segmentation mask as a colored binary little-endian PLY point cloud in base_link coordinates.",
        }
    },
)
def segment_unknown_ward_item_pointcloud(
    req: LocateWardItemRequest,
    depth_margin_m: float | None = Query(
        0.08,
        description="For segmented point clouds, remove only points farther than center depth + this many meters. Set to 0 to keep the full mask.",
    ),
):
    """
    Segments the ward item using the ward_object_pipeline (RF-DETR + SAM2 + DINOv2), same as
    `/segment-unknown-ward-item`, then returns the same frame's RealSense point cloud cropped
    to the item's mask contour, with points too far behind the object removed, and transformed
    into the robot base frame.
    Note that the point cloud does not consinder depth_offset_m, as the point cloud is generated from the raw depth image.
    """
    if req.object_name not in settings.WARD_ITEM_CLASS_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid object_name '{req.object_name}'. Valid values: {settings.WARD_ITEM_CLASS_NAMES}",
        )

    try:
        color_frame, depth_frame = realsense_service.capture_aligned_frames()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        transform_context = object_detection_service.get_detection_transform_context()
        T_cam_wrist, R_gripper2base, t_gripper2base_vec, _ = transform_context

        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        pipeline = ward_object_pipeline_service.get_pipeline()
        result = pipeline.predict(color_image_rgb, verbose=False)

        obj_coords = bbox = depth_in_meters = mask_contour = None
        best_confidence = -1.0

        for idx, obj in enumerate(result.get("objects", [])):
            if obj["class_name"] != req.object_name:
                continue
            confidence = float(result["final_results"][idx].get("rfdetr_confidence", 0.0))
            if confidence <= best_confidence:
                continue

            candidate_bbox = [int(round(v)) for v in obj["bbox"]]
            mask = compressed_rle_to_binary_mask(obj["rle_mask"]).astype(bool)

            candidate_coords, _, candidate_bbox, _, _, candidate_depth, candidate_mask_contour = \
                object_detection_service.locate_mask_in_base(
                    mask, candidate_bbox, color_image, depth_image,
                    T_cam_wrist, R_gripper2base, t_gripper2base_vec,
                )
            if candidate_coords is None:
                continue

            best_confidence = confidence
            obj_coords, bbox = candidate_coords, candidate_bbox
            depth_in_meters, mask_contour = candidate_depth, candidate_mask_contour

        if obj_coords is None or bbox is None:
            raise HTTPException(status_code=404, detail=f"'{req.object_name}' not detected in the current view.")
        if not mask_contour:
            raise HTTPException(status_code=404, detail=f"'{req.object_name}' segmentation mask not detected in the current view.")

        vertices, colors = realsense_service.point_cloud_from_frames(
            color_frame,
            depth_frame,
            depth_center_m=depth_in_meters,
            depth_margin_m=_pointcloud_depth_margin(depth_margin_m),
            mask_contour=mask_contour,
            depth_filter_mode="far_only",
        )
        if len(vertices) == 0:
            raise HTTPException(status_code=500, detail=f"Detected mask has no valid depth points: {bbox}")

        vertices = _to_base_link_pointcloud(vertices, transform_context)
        return _pointcloud_ply_response(
            vertices,
            colors,
            filename_prefix=req.object_name,
            bbox=bbox,
            depth_in_meters=depth_in_meters,
            depth_margin_m=depth_margin_m,
            extra_headers={"X-PointCloud-Mask-Used": "true" if mask_contour else "false"},
        )
    except HTTPException:
        raise
    except RealSenseError as e:
        logger.error(f"Camera error in /segment-unknown-ward-item-pointcloud: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Camera error: {str(e)}")
    except ObjectDetectionError as e:
        logger.error(f"Failed to segment unknown item point cloud: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /segment-unknown-ward-item-pointcloud: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post(
    "/segment-unknown-ward-item-pointcloud-visual",
    summary="Preview the item pixels used for the base_link point cloud (unknown pipeline)",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Returns a PNG preview of the segmentation mask and depth-filtered pixels before the PLY point cloud is transformed to base_link.",
        }
    },
)
def segment_unknown_ward_item_pointcloud_visual(
    req: LocateWardItemRequest,
    depth_margin_m: float = Query(
        0.08,
        description="Show mask points after removing only points farther than center depth + this many meters.",
    ),
):
    """
    Shows the ward_object_pipeline (RF-DETR + SAM2 + DINOv2) mask polygon after removing
    only points too far behind the detected item.
    Note that the point cloud does not consinder depth_offset_m, as the point cloud is generated from the raw depth image.
    """
    if req.object_name not in settings.WARD_ITEM_CLASS_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid object_name '{req.object_name}'. Valid values: {settings.WARD_ITEM_CLASS_NAMES}",
        )

    try:
        color_frame, depth_frame = realsense_service.capture_aligned_frames()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        T_cam_wrist, R_gripper2base, t_gripper2base_vec, _ = object_detection_service.get_detection_transform_context()

        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        pipeline = ward_object_pipeline_service.get_pipeline()
        result = pipeline.predict(color_image_rgb, verbose=False)

        obj_coords = bbox = depth_in_meters = mask_contour = None
        best_confidence = -1.0

        for idx, obj in enumerate(result.get("objects", [])):
            if obj["class_name"] != req.object_name:
                continue
            confidence = float(result["final_results"][idx].get("rfdetr_confidence", 0.0))
            if confidence <= best_confidence:
                continue

            candidate_bbox = [int(round(v)) for v in obj["bbox"]]
            mask = compressed_rle_to_binary_mask(obj["rle_mask"]).astype(bool)

            candidate_coords, _, candidate_bbox, _, _, candidate_depth, candidate_mask_contour = \
                object_detection_service.locate_mask_in_base(
                    mask, candidate_bbox, color_image, depth_image,
                    T_cam_wrist, R_gripper2base, t_gripper2base_vec,
                )
            if candidate_coords is None:
                continue

            best_confidence = confidence
            obj_coords, bbox = candidate_coords, candidate_bbox
            depth_in_meters, mask_contour = candidate_depth, candidate_mask_contour

        if obj_coords is None or bbox is None:
            raise HTTPException(status_code=404, detail=f"'{req.object_name}' not detected in the current view.")

        height, width = depth_image.shape[:2]
        depth_m = depth_image.astype(np.float32) * realsense_service.depth_scale

        if mask_contour:
            poly = np.array(mask_contour, dtype=np.int32)
            fill = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(fill, [poly], 1)
            spatial_mask = fill.astype(bool)
        else:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))
            spatial_mask = np.zeros((height, width), dtype=bool)
            spatial_mask[y1:y2, x1:x2] = True

        if depth_margin_m == 0:
            depth_filter = depth_m > 0
        else:
            depth_filter = (
                (depth_m > 0)
                & (depth_m <= depth_in_meters + depth_margin_m)
            )
        active_mask = spatial_mask & depth_filter

        preview = color_image.copy()
        overlay = preview.copy()
        overlay[active_mask] = (0, 255, 0)
        cv2.addWeighted(overlay, 0.45, preview, 0.55, 0, preview)

        if mask_contour:
            cv2.polylines(preview, [np.array(mask_contour, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            label_x, label_y = int(mask_contour[0][0]), int(mask_contour[0][1])
        else:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_x, label_y = x1, y1

        depth_label = (
            f"{req.object_name} full mask"
            if depth_margin_m == 0
            else f"{req.object_name} max depth {depth_in_meters + depth_margin_m:.3f}m"
        )
        cv2.putText(
            preview,
            depth_label,
            (label_x, max(20, label_y - 10)),
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
    except RealSenseError as e:
        logger.error(f"Camera error in /segment-unknown-ward-item-pointcloud-visual: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Camera error: {str(e)}")
    except ObjectDetectionError as e:
        logger.error(f"Failed to render segment unknown item point cloud visual: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /segment-unknown-ward-item-pointcloud-visual: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post(
    "/segment-unknown-all-ward-items",
    response_model=SegAllWardItemsResponse, 
    summary="Detect all ward items, including unrecognized ones, using the RF-DETR + SAM2 + DINOv2 pipeline"
)
def segment_unknown_all_ward_items(
    depth_offset_m: float | None = Query(
        None, description="Constant offset (meters) added to the measured depth."
    ),
):
    """
    - Captures an image from the RealSense camera.
    - Runs the ward_object_pipeline (https://github.com/sstc-aiteam/ward_object_pipeline):
      an RF-DETR detector finds the chair/platform ROI and candidate objects, SAM2 segments
      everything sitting inside that ROI, and DINOv2 verifies the detector's class guess.
    - Items where the detector and DINOv2 disagree (or either is inconclusive) are reported
      with class_name "unknown", so this endpoint can surface ward items that
      `segment_all_ward_items()` cannot recognize.
    - Returns bounding boxes, mask contour polygons, poses, and an annotated image (base64 PNG).
    """
    try:
        if not realsense_service.is_initialized:
            realsense_service._initialize()

        color_image, depth_image = realsense_service.capture_images()
        T_cam_wrist, R_gripper2base, t_gripper2base_vec, _ = object_detection_service.get_detection_transform_context()

        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        pipeline = ward_object_pipeline_service.get_pipeline()
        result = pipeline.predict(color_image_rgb, verbose=False)

        detection_image = color_image.copy()
        detected_items: list[SegWardItem] = []

        for idx, obj in enumerate(result.get("objects", [])):
            class_name = obj["class_name"]
            bbox = [int(round(v)) for v in obj["bbox"]]
            mask = compressed_rle_to_binary_mask(obj["rle_mask"]).astype(bool)
            confidence = float(result["final_results"][idx].get("rfdetr_confidence", 0.0))

            obj_coords, pixel_coords, bbox, object_yaw_deg, object_yaw_rad, depth_in_meters, mask_contour = \
                object_detection_service.locate_mask_in_base(
                    mask, bbox, color_image, depth_image,
                    T_cam_wrist, R_gripper2base, t_gripper2base_vec,
                    depth_offset_m=depth_offset_m,
                )

            detected_items.append(
                SegWardItem(
                    class_name=class_name,
                    bbox=bbox,
                    confidence=round(confidence, 4),
                    mask_contour=mask_contour,
                    object_pose_in_base=obj_coords.tolist() if obj_coords is not None else None,
                    object_pixel_coords=pixel_coords,
                    object_yaw_deg=object_yaw_deg,
                    object_yaw_rad=object_yaw_rad,
                    depth_in_meters=depth_in_meters,
                )
            )

            color = palette_color(idx)
            label = f"{class_name} {confidence:.2f}"
            if mask_contour:
                draw_seg_mask_annotation(detection_image, mask_contour, color=color, class_name=class_name, skip_classes=settings.ANNOTATION_SKIP_CLASSES)
            draw_detection_annotation(detection_image, bbox, pixel_coords, label, color=color, class_name=class_name, skip_classes=settings.ANNOTATION_SKIP_CLASSES)
            draw_yaw_annotation(
                detection_image, bbox, pixel_coords, object_yaw_deg, object_yaw_rad, show_label=True, color=color, class_name=class_name, skip_classes=settings.ANNOTATION_SKIP_CLASSES,
            )

        b64_image = None
        success, encoded_img = cv2.imencode('.png', detection_image)
        if success:
            b64_image = base64.b64encode(encoded_img).decode('utf-8')

        if not result.get("success", True):
            msg = result.get("reason", "Pipeline failed.")
        elif detected_items:
            msg = f"Detected {len(detected_items)} ward item(s)."
        else:
            msg = "No ward items detected in the current view."
        return SegAllWardItemsResponse(message=msg, detected_items=detected_items, detection_image_base64=b64_image)

    except RealSenseError as e:
        logger.error(f"Camera error in /segment-unknown-all-ward-items: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Camera error: {str(e)}")
    except ObjectDetectionError as e:
        logger.error(f"Failed in /segment-unknown-all-ward-items: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /segment-unknown-all-ward-items: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post(
    "/segment-unknown-all-ward-items-visual",
    summary="Detect all ward items, including unrecognized ones, and return annotated image with masks",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Returns the camera image with all detected ward items (including 'unknown') and segmentation masks drawn on it.",
        }
    },
)
def segment_unknown_all_ward_items_visual(
    depth_offset_m: float | None = Query(
        None, description="Constant offset (meters) added to the measured depth."
    ),
):
    """
    - Delegates to `segment_unknown_all_ward_items()` for full detection logic.
    - Returns the annotated detection image (with mask overlays) as a PNG.
    """
    result = segment_unknown_all_ward_items(depth_offset_m=depth_offset_m)
    if not result.detection_image_base64:
        raise HTTPException(status_code=500, detail="Detection produced no image.")
    image_bytes = base64.b64decode(result.detection_image_base64)
    return Response(content=image_bytes, media_type="image/png")


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
