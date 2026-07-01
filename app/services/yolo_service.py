from ultralytics import YOLO
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

class YoloService:
    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading YOLO model from {self._model_path}...")
            self._model = YOLO(self._model_path)
            logger.info("✅ YOLO model loaded.")

    def get_model(self):
        self._load_model()
        return self._model

bottle_yolo_service = YoloService(settings.BOTTLE_YOLO_MODEL_PATH)
ward_item_yolo_service = YoloService(settings.WARD_ITEM_DET_MODEL_PATH)
ward_item_seg_yolo_service = YoloService(settings.WARD_ITEM_SEG_MODEL_PATH)
