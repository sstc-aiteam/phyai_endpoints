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

    # OpenAI settings for grasp verification
    OPENAI_API_KEY: str | None = None
    OPENAI_GRASP_VERIFICATION_MODEL: str = "gpt-5.6"

    # Hand-eye calibration file
    CALIBRATION_FILE: str = str(PROJECT_ROOT / "handeye_result.npy")

    # Bottle object detection model settings
    BOTTLE_YOLO_MODEL_PATH: str = str(PROJECT_ROOT / "bottle.pt")
    BOTTLE_CLASS_ID: int = 0
    
    # Ward item detection/segmentation model settings
    WARD_ITEM_DET_MODEL_PATH: str = str(PROJECT_ROOT / "ward_item.pt")
    WARD_ITEM_SEG_MODEL_PATH: str = str(PROJECT_ROOT / "ward_item_seg.pt")

    WARD_ITEM_CLASS_NAMES: list[str] = [
        'ac_remotecontrol', 'bottle_alcohol_spray', 'chair_surface', 'cotton_swab', 'cotton_swabs_pp',
        'disposable_mask', 'gauze_pp', 'saline', 'syringe_nipro', 'waterproof_bandages_ppb', 'unknown'
    ]

    # Ward object pipeline (RF-DETR + SAM2 + DINOv2) settings, used to detect ward items
    # including ones unrecognized by the fixed WARD_ITEM_CLASS_NAMES list (reported as "unknown").
    # See https://github.com/sstc-aiteam/ward_object_pipeline (branch: feature/yolo26-detector-backend)
    WARD_OBJECT_PIPELINE_BACKEND: str = "rfdetr"
    WARD_OBJECT_PIPELINE_RFDETR_WEIGHTS_PATH: str = str(PROJECT_ROOT / "checkpoint_best_total.pth")
    WARD_OBJECT_PIPELINE_YOLO_WEIGHTS_PATH: str = str(PROJECT_ROOT / "ward_item_seg.pt")
    WARD_OBJECT_PIPELINE_DINOV2_CACHE_PATH: str = str(PROJECT_ROOT / "dinov2_reference_cache.pt")
    WARD_OBJECT_PIPELINE_NUM_CLASSES: int = 11
    WARD_OBJECT_PIPELINE_DETECTOR_THRESHOLD: float = 0.3
    WARD_OBJECT_PIPELINE_DINOV2_THRESHOLD: float = 0.75
    WARD_OBJECT_PIPELINE_MIN_DETECTOR_CONFIDENCE: float = 0.50

    # Classes excluded from drawing
    ANNOTATION_SKIP_CLASSES: list[str] = ["chair_surface"]

    # Distance from TCP (wrist3) offset for the gripper length along the camera's forward axis
    GRIPPER_LEN_OFFSET_IN_METERS: float = 0.15

    # Distance to stay away from the object when approaching
    APPROACH_OFFSET_IN_METERS: float = 0.15

    class Config:
        case_sensitive = True


settings = Settings()