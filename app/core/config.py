from pathlib import Path
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    PROJECT_NAME: str = "PhyAI Endpoints"
    API_V1_STR: str = "/api/v1"
    ROBOT_IP: str = "192.168.50.75"

    # RealSense camera settings
    # Setting1 - lower resolution and higher FPS for better real-time performance during testing
    # RS_STREAM_WIDTH: int = 640
    # RS_STREAM_HEIGHT: int = 480
    # RS_STREAM_FPS: int = 30
    # Setting2 - higher resolution and lower FPS for better hand-eye calibration accuracy
    RS_STREAM_WIDTH: int = 1280
    RS_STREAM_HEIGHT: int = 720
    RS_STREAM_FPS: int = 5

    # Directory for saving captured images
    CAPTURE_DIR: Path = PROJECT_ROOT / "captured_images"

    # Hand-eye calibration file
    CALIBRATION_FILE: str = str(PROJECT_ROOT / "handeye_result.npy")

    # Object detection model settings
    BOTTLE_YOLO_MODEL_PATH: str = str(PROJECT_ROOT / "bottle.pt")
    BOTTLE_CLASS_ID: int = 0
    
    WARD_ITEM_YOLO_MODEL_PATH: str = str(PROJECT_ROOT / "best.pt")

    # Distance from TCP (wrist3) offset for the gripper length along the camera's forward axis
    GRIPPER_LEN_OFFSET_IN_METERS: float = 0.15

    # Distance to stay away from the object when approaching
    APPROACH_OFFSET_IN_METERS: float = 0.15

    class Config:
        case_sensitive = True


settings = Settings()