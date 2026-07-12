# detectors/detector_output.py

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import supervision as sv


@dataclass
class DetectorOutput:
    """
    Normalized detector output shared by RF-DETRSeg and YOLO26-Seg.

    All coordinates are full-image coordinates.
    Masks are full-image binary masks with shape NxHxW.
    """

    xyxy: np.ndarray
    mask: Optional[np.ndarray]
    class_id: np.ndarray
    confidence: np.ndarray
    class_names: List[str]

    def __len__(self):
        if self.xyxy is None:
            return 0
        return len(self.xyxy)

    def to_supervision(self) -> sv.Detections:
        if len(self) == 0:
            return sv.Detections(
                xyxy=np.empty((0, 4), dtype=float),
                confidence=np.empty((0,), dtype=float),
                class_id=np.empty((0,), dtype=int),
            )

        kwargs = {
            "xyxy": self.xyxy.astype(float),
            "confidence": self.confidence.astype(float),
            "class_id": self.class_id.astype(int),
        }

        if self.mask is not None:
            kwargs["mask"] = self.mask.astype(bool)

        return sv.Detections(**kwargs)

    @classmethod
    def from_supervision(cls, detections: sv.Detections, class_names: List[str]):
        return cls(
            xyxy=detections.xyxy,
            mask=detections.mask,
            class_id=detections.class_id,
            confidence=detections.confidence,
            class_names=class_names,
        )
