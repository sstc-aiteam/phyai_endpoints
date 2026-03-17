from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

class YoloService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(YoloService, cls).__new__(cls)
            cls._instance._model = None
        return cls._instance

    def _load_model(self):
        if self._model is None:
            logger.info("Loading YOLOv8n model...")
            # 'yolov8n.pt' is the Nano model (fastest).
            self._model = YOLO('yolov8n.pt')
            logger.info("✅ YOLOv8n model loaded.")

    def get_model(self):
        self._load_model()
        return self._model

yolo_service = YoloService()