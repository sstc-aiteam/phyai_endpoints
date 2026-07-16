import base64
import cv2
import numpy as np
import pyrealsense2 as rs
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


# --- Custom Exceptions 
class RealSenseError(Exception):
    """Base exception for RealSense camera errors."""
    pass

class NoDeviceError(RealSenseError):
    """Exception raised when no RealSense device is found."""
    pass

class FrameCaptureError(RealSenseError):
    """Exception raised when frames cannot be captured."""
    pass


class RealSenseService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RealSenseService, cls).__new__(cls)

            cls._instance.is_initialized = False
            cls._instance.align = None # Initialize to None
            cls._instance.depth_intrinsics = None
            cls._instance.color_intrinsics = None
            cls._instance.depth_scale = None
            cls._instance.pointcloud = None
        return cls._instance

    def _initialize(self):
        """Initializes the RealSense pipeline and configuration, only if not already initialized."""
        
        if self.is_initialized:
            logger.info("RealSense pipeline is already initialized.")
            return

        try:
            logger.info("Attempting to initialize RealSense pipeline...")
            self.pipeline = rs.pipeline()
            config = rs.config()
            # Create hole filling filter with mode 2 (nearest from around)
            self.hole_filling = rs.hole_filling_filter(2)
            self.pointcloud = rs.pointcloud()
            
            # 1. Check for device connection
            context = rs.context()
            if len(context.devices) == 0:
                logger.warning("No RealSense device connected.")
                raise NoDeviceError("No RealSense device connected.")
        
            # 2. Configure streams
            width, height, fps = settings.RS_STREAM_WIDTH, settings.RS_STREAM_HEIGHT, settings.RS_STREAM_FPS
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

            # 3. Start pipeline
            self.profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)

            # Log the depth scale to confirm units
            # D405 has a depth scale of 0.0001 (0.1mm), while D415/D435 typically have 0.001 (1mm)
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            logger.info(f"RealSense initialized with depth scale: {self.depth_scale} meters/unit")

            # 4. Get and store intrinsics
            depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self.depth_intrinsics = depth_profile.get_intrinsics()
            color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.color_intrinsics = color_profile.get_intrinsics()
            
            self.is_initialized = True
            logging.info("✅ RealSense pipeline started successfully.")
            
        except RuntimeError as e:
            # Catch all expected (and unexpected) errors during initialization
            self.is_initialized = False
            # Clean up the pipeline if it was partially started
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass # Ignore if stop fails (likely because it wasn't started)
            
            raise RealSenseError(f"Failed to start RealSense pipeline: {e}") from e

    def _capture_aligned_frames(self, use_hole_filling: bool = False):
        """
        Captures and aligns one pair of color and depth frames, returning RealSense frame objects.

        Args:
            use_hole_filling (bool): If True, applies a hole-filling filter to the depth frame. Defaults to False.
        Returns: 
            A tuple containing (color_frame, depth_frame).
        """
        try:
            # 1. Check/Re-initialize device if not ready       
            if not self.is_initialized:
                self._initialize() # Retry connection
                if not self.is_initialized:
                    # If retry failed, return empty images as requested
                    raise RealSenseError("RealSense device is not initialized after retry.")

            # 2. Capture frames (only runs if self.is_initialized is True)
            # Skip initial frames for auto-exposure/gain to settle
            for _ in range(5):
                # Using 100ms timeout for quick discard if device is flaky
                self.pipeline.wait_for_frames(timeout_ms=5000) 

            # Wait for the actual frames to process
            frames = self.pipeline.wait_for_frames(timeout_ms=3000)
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                raise FrameCaptureError("Could not capture valid frames from RealSense camera.")

            # Conditionally apply the hole filling filter to the depth frame
            if use_hole_filling:
                depth_frame = self.hole_filling.process(depth_frame)

            return color_frame, depth_frame

        except RuntimeError as e:
            # This handles errors during frame capture (e.g., device unplugged mid-run)
            logging.error(f"RealSense runtime error during capture: {e}. Device is now considered uninitialized.", exc_info=True)
            raise RealSenseError(f"Error with RealSense camera: {e}") from e

    def capture_aligned_frames(self, use_hole_filling: bool = False):
        """
        Captures and aligns one pair of color and depth frames, returning RealSense frame objects.
        """
        return self._capture_aligned_frames(use_hole_filling=use_hole_filling)

    def capture_images(self, use_hole_filling: bool = False):
        """
        Captures and aligns one pair of color and depth frames.
        Will attempt to re-initialize the device if not connected.

        Args:
            use_hole_filling (bool): If True, applies a hole-filling filter to the depth frame. Defaults to False.
        Returns:
            A tuple containing (color_image, depth_image). Depth values are raw depth units.
        """
        color_frame, depth_frame = self._capture_aligned_frames(use_hole_filling=use_hole_filling)
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def point_cloud_from_frames(
        self,
        color_frame,
        depth_frame,
        bbox: list[int] | None = None,
        depth_center_m: float | None = None,
        depth_margin_m: float | None = None,
        mask_contour: list[list[int]] | None = None,
        depth_filter_mode: str = "symmetric",
    ):
        """
        Converts aligned RealSense color/depth frames to a colored point cloud.

        Args:
            bbox (list[int] | None): Optional [x1, y1, x2, y2] pixel crop. Ignored when
                mask_contour is provided.
            depth_center_m (float | None): Optional object center depth in meters.
            depth_margin_m (float | None): Optional +/- meter range around depth_center_m.
            mask_contour (list[list[int]] | None): Optional polygon [[x,y],...] from a
                segmentation mask. When provided, takes precedence over bbox for spatial
                filtering.
            depth_filter_mode (str): "symmetric" keeps points within
                depth_center_m +/- depth_margin_m. "far_only" only removes points farther
                than depth_center_m + depth_margin_m.

        Returns:
            A tuple containing (vertices, colors), where vertices are Nx3 float32 meters in
            camera coordinates and colors are Nx3 uint8 RGB values.
        """
        self.pointcloud.map_to(color_frame)
        points = self.pointcloud.calculate(depth_frame)

        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        color_image = np.asanyarray(color_frame.get_data())
        colors = color_image.reshape(-1, 3)[:, ::-1]

        valid = np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0)
        height, width = color_image.shape[:2]

        if mask_contour is not None:
            poly = np.array(mask_contour, dtype=np.int32)
            spatial_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(spatial_mask, [poly], 1)
            valid &= spatial_mask.reshape(-1).astype(bool)
        elif bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                raise RealSenseError(f"Invalid point cloud bbox after clipping: {[x1, y1, x2, y2]}")

            bbox_mask_2d = np.zeros((height, width), dtype=bool)
            bbox_mask_2d[y1:y2, x1:x2] = True
            valid &= bbox_mask_2d.reshape(-1)

        if depth_center_m is not None and depth_margin_m is not None and depth_margin_m > 0:
            depth_max = depth_center_m + depth_margin_m
            if depth_filter_mode == "far_only":
                valid &= vertices[:, 2] <= depth_max
            else:
                depth_min = max(0.0, depth_center_m - depth_margin_m)
                valid &= (vertices[:, 2] >= depth_min) & (vertices[:, 2] <= depth_max)

        return vertices[valid], colors[valid]

    def capture_point_cloud(
        self,
        use_hole_filling: bool = False,
        bbox: list[int] | None = None,
        depth_center_m: float | None = None,
        depth_margin_m: float | None = None,
    ):
        """
        Captures aligned frames and converts the RealSense depth frame to a colored point cloud.

        Args:
            use_hole_filling (bool): If True, applies a hole-filling filter to the depth frame.
            bbox (list[int] | None): Optional [x1, y1, x2, y2] pixel crop in the aligned
                color/depth image. x2 and y2 are treated as exclusive bounds.
            depth_center_m (float | None): Optional object center depth in meters.
            depth_margin_m (float | None): Optional +/- meter range around depth_center_m.

        Returns:
            A tuple containing (vertices, colors), where vertices are Nx3 float32 meters in
            camera coordinates and colors are Nx3 uint8 RGB values.
        """
        color_frame, depth_frame = self._capture_aligned_frames(use_hole_filling=use_hole_filling)
        return self.point_cloud_from_frames(
            color_frame,
            depth_frame,
            bbox=bbox,
            depth_center_m=depth_center_m,
            depth_margin_m=depth_margin_m,
        )

    def deproject_pixel_to_point(
        self, 
        pixel: list[int], 
        depth: float, 
        depth_offset_m: float | None = None,
    ) -> tuple[float, list[float]]:
        """
        Deprojects a 2D pixel with a given depth into a 3D point in the camera's coordinate space.

        Args:
            pixel (list[int]): The [u, v] pixel coordinates.
            depth (float): The depth at that pixel, in meters.
            depth_offset_m (float | None): Constant offset added to `depth` before deprojecting.
                Defaults to settings.DEPTH_OFFSET_IN_METERS if None.

        Returns:
            tuple[float, list[float]]: The offset-adjusted depth in meters, and the [x, y, z]
            coordinates of the 3D point computed from that adjusted depth.

        Raises:
            RealSenseError: If the camera intrinsics are not available.
        """
        if not self.color_intrinsics:
            raise RealSenseError("Cannot deproject point: color intrinsics not available. Ensure the camera is initialized.")
        
        d = self.adjust_depth(depth, depth_offset_m)  # Adjust depth with offset if needed

        # pyrealsense2.rs2_deproject_pixel_to_point takes intrinsics, pixel, and depth
        return d, rs.rs2_deproject_pixel_to_point(self.color_intrinsics, pixel, d)

    def adjust_depth(self, depth_in_meters, offset_m=None):
        """
        Adjusts a depth reading by adding a constant offset.
        """
        d = depth_in_meters if offset_m is None else depth_in_meters + offset_m
        logging.info(f"Adjusting depth: adjusted={d:.4f}m = original={depth_in_meters:.4f}m + offset={offset_m:.4f}m, ")
        
        return d

    def shutdown(self):
        """Stops the RealSense pipeline, checking if it was ever initialized."""
        if self.is_initialized:
            try:
                self.pipeline.stop()
                logger.info("RealSense pipeline stopped.")
            except Exception as e:
                logger.error(f"Error while stopping pipeline: {e}")
            finally:
                self.is_initialized = False
                self.align = None
                self.depth_intrinsics = None
                self.color_intrinsics = None
                self.depth_scale = None
                self.pointcloud = None
        else:
            logger.info("RealSense pipeline was not active/initialized, nothing to stop.")

realsense_service = RealSenseService()
