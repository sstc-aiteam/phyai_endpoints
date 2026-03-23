import base64
import cv2
import numpy as np
import pyrealsense2 as rs
import logging

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
    
    # Define default stream parameters as class/instance attributes for easy access
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS = 640, 480, 30

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RealSenseService, cls).__new__(cls)

            cls._instance.is_initialized = False
            cls._instance.align = None # Initialize to None
            cls._instance.depth_intrinsics = None
            cls._instance.color_intrinsics = None
            cls._instance.depth_scale = None
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
            
            # 1. Check for device connection
            context = rs.context()
            if len(context.devices) == 0:
                logger.warning("No RealSense device connected.")
                raise NoDeviceError("No RealSense device connected.")
        
            # 2. Configure streams
            width, height, fps = self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT, self.DEFAULT_FPS
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

    def capture_images(self, use_hole_filling: bool = False):
        """
        Captures and aligns one pair of color and depth frames.
        Will attempt to re-initialize the device if not connected.

        Args:
            use_hole_filling (bool): If True, applies a hole-filling filter to the depth frame. Defaults to False.
        Returns: 
            A tuple containing (color_image, depth_image). Depth values are raw depth units.
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

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return color_image, depth_image

        except RuntimeError as e:
            # This handles errors during frame capture (e.g., device unplugged mid-run)
            logging.error(f"RealSense runtime error during capture: {e}. Device is now considered uninitialized.", exc_info=True)
            raise RealSenseError(f"Error with RealSense camera: {e}") from e

    def deproject_pixel_to_point(self, pixel: list[int], depth: float) -> list[float]:
        """
        Deprojects a 2D pixel with a given depth into a 3D point in the camera's coordinate space.

        Args:
            pixel (list[int]): The [u, v] pixel coordinates.
            depth (float): The depth at that pixel, in meters.

        Returns:
            list[float]: The [x, y, z] coordinates of the 3D point.
            
        Raises:
            RealSenseError: If the camera intrinsics are not available.
        """
        if not self.color_intrinsics:
            raise RealSenseError("Cannot deproject point: color intrinsics not available. Ensure the camera is initialized.")
        
        # pyrealsense2.rs2_deproject_pixel_to_point takes intrinsics, pixel, and depth
        return rs.rs2_deproject_pixel_to_point(self.color_intrinsics, pixel, depth)

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
        else:
            logger.info("RealSense pipeline was not active/initialized, nothing to stop.")

realsense_service = RealSenseService()
