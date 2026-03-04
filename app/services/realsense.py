import base64
import cv2
import numpy as np
import pyrealsense2 as rs
import logging

logger = logging.getLogger(__name__)

class RealSenseService:
    def __init__(self):
        pass

    def capture_image_base64(self) -> str:
        """
        Starts the RealSense pipeline, captures a color frame, encodes it as JPEG,
        and returns the Base64 representation of the image.
        """
        pipeline = rs.pipeline()
        config = rs.config()
        # Configure the pipeline to stream color format (640x480 at 30 fps)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            # Start streaming
            pipeline.start(config)

            # Wait for a few frames to allow auto-exposure to stabilize
            for _ in range(5):
                pipeline.wait_for_frames()
                
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("Could not retrieve color frame from RealSense pipeline.")

            # Convert image to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Encode image to jpg using OpenCV
            success, encoded_image = cv2.imencode('.jpg', color_image)
            if not success:
                raise RuntimeError("Failed to encode image to JPEG.")

            # Convert to base64 string
            b64_string = base64.b64encode(encoded_image).decode('utf-8')
            return b64_string

        finally:
            # Ensure the pipeline is stopped regardless of errors
            pipeline.stop()

realsense_service = RealSenseService()
