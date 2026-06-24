import base64
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from app.services.realsense import realsense_service
from app.services.pointcloud import encode_binary_ply
from app.api.endpoints.schema import CameraCaptureResponse
from app.core.config import settings

router = APIRouter()

@router.get(
        "/capture", 
        summary="Capture image from RealSense camera", 
        response_model=CameraCaptureResponse
)
def capture():
    """
    Captures a single frame (RGB and Depth) from the connected RealSense camera and returns them as Base64-encoded strings.
    """
    try:
        color_image, depth_image = realsense_service.capture_images()

        # Encode the image to PNG
        success, encoded_img = cv2.imencode('.png', color_image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image to PNG")

        b64_image = base64.b64encode(encoded_img).decode('utf-8')

        # Encode the depth image to PNG (preserves 16-bit data)
        success_depth, encoded_depth = cv2.imencode('.png', depth_image)
        if not success_depth:
            raise HTTPException(status_code=500, detail="Failed to encode depth image to PNG")

        b64_depth = base64.b64encode(encoded_depth).decode('utf-8')

        return {
            "image_format": "png",
            "image_base64": b64_image,
            "depth_format": "png",
            "depth_base64": b64_depth
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to capture image: {str(e)}")

@router.get(
    "/capture_visual", 
    summary="Capture and return a combined RGB and Depth image",
    response_class=Response,
    # This part updates the Swagger UI schema
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Return the combined RGB and Depth image as a PNG.",
        }
    }
)
def capture_visual():
    """
    Captures RGB and Depth frames, combines them side-by-side into a single visual image,
    and returns it directly as a PNG image.
    """
    try:
        color_image, depth_image = realsense_service.capture_images()

        # Normalize the depth image to 0-255 range 
        # This ensures the gradient scales based on what the camera actually sees
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # This turns the grayscale into a "heat map" which is much easier to read, i.e., 
        # blue usually represents "Close to camera" and red represents "Far from camera".
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # Save the RGB image to disk
        capture_dir = settings.CAPTURE_DIR
        capture_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = capture_dir / f"rgb_{timestamp}.png"
        cv2.imwrite(str(save_path), color_image)

        # Combine images horizontally
        combined_image = np.hstack((color_image, depth_colormap))

        # Encode the combined image to PNG
        success, encoded_img = cv2.imencode('.png', combined_image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image to PNG")

        # Return the image as a response with the correct media type
        return Response(content=encoded_img.tobytes(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to capture image: {str(e)}")

@router.get(
    "/capture_pointcloud",
    summary="Capture and return a colored point cloud from RealSense depth",
    response_class=Response,
    responses={
        200: {
            "content": {"application/octet-stream": {}},
            "description": "Return a colored point cloud as a binary little-endian PLY file.",
        }
    },
)
def capture_pointcloud(
    use_hole_filling: bool = Query(False, description="Apply RealSense hole-filling before point cloud generation."),
    x1: int | None = Query(None, description="Optional bbox left pixel."),
    y1: int | None = Query(None, description="Optional bbox top pixel."),
    x2: int | None = Query(None, description="Optional bbox right pixel."),
    y2: int | None = Query(None, description="Optional bbox bottom pixel."),
    depth_center_m: float | None = Query(None, description="Optional center depth in meters for filtering bbox points."),
    depth_margin_m: float | None = Query(None, description="Optional +/- meter range around depth_center_m."),
):
    """
    Captures aligned RGB and Depth frames, converts depth to a colored point cloud using
    pyrealsense2 pointcloud, and returns it as a binary PLY file.
    """
    try:
        bbox_values = [x1, y1, x2, y2]
        if any(value is not None for value in bbox_values) and not all(value is not None for value in bbox_values):
            raise HTTPException(status_code=400, detail="Provide all bbox values: x1, y1, x2, y2.")

        bbox = [x1, y1, x2, y2] if all(value is not None for value in bbox_values) else None
        vertices, colors = realsense_service.capture_point_cloud(
            use_hole_filling=use_hole_filling,
            bbox=bbox,
            depth_center_m=depth_center_m,
            depth_margin_m=depth_margin_m,
        )
        if len(vertices) == 0:
            raise HTTPException(status_code=500, detail="Captured point cloud has no valid depth points.")

        ply_bytes = encode_binary_ply(vertices, colors)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"pointcloud_{timestamp}.ply"

        return Response(
            content=ply_bytes,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to capture point cloud: {str(e)}")
