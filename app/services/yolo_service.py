from ultralytics import YOLO
import logging

from app.core.config import settings

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
            logger.info(f"Loading YOLO model from {settings.YOLO_MODEL_PATH}...")
            self._model = YOLO(settings.YOLO_MODEL_PATH)
            logger.info("✅ YOLO model loaded.")

    def get_model(self):
        self._load_model()
        return self._model

yolo_service = YoloService()


class BestYoloService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BestYoloService, cls).__new__(cls)
            cls._instance._model = None
        return cls._instance

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading best YOLO model from {settings.BEST_MODEL_PATH}...")
            self._model = YOLO(settings.BEST_MODEL_PATH)
            logger.info("✅ Best YOLO model loaded.")

    def get_model(self):
        self._load_model()
        return self._model

best_yolo_service = BestYoloService()
