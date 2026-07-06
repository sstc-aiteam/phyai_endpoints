import logging

import torch

from app.core.config import settings
from app.services.ward_object_pipeline import WardObjectPipeline

logger = logging.getLogger(__name__)


class WardObjectPipelineService:
    def __init__(self):
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading ward object pipeline (backend={settings.WARD_OBJECT_PIPELINE_BACKEND}, device={device})...")
            self._pipeline = WardObjectPipeline(
                detector_backend=settings.WARD_OBJECT_PIPELINE_BACKEND,
                detector_weights_path=settings.WARD_OBJECT_PIPELINE_RFDETR_WEIGHTS_PATH,
                dinov2_database_path=settings.WARD_OBJECT_PIPELINE_DINOV2_CACHE_PATH,
                num_classes=settings.WARD_OBJECT_PIPELINE_NUM_CLASSES,
                detector_threshold=settings.WARD_OBJECT_PIPELINE_DETECTOR_THRESHOLD,
                sam2_device=0 if device == "cuda" else "cpu",
                dinov2_device=device,
                dinov2_similarity_threshold=settings.WARD_OBJECT_PIPELINE_DINOV2_THRESHOLD,
                min_rfdetr_confidence=settings.WARD_OBJECT_PIPELINE_MIN_DETECTOR_CONFIDENCE,
            )
            logger.info("✅ Ward object pipeline loaded.")

    def get_pipeline(self) -> WardObjectPipeline:
        self._load_pipeline()
        return self._pipeline


ward_object_pipeline_service = WardObjectPipelineService()
