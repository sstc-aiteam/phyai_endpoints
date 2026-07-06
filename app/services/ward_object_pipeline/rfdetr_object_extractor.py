# rfdetr_object_extractor.py

import numpy as np
import supervision as sv


class RFDETRObjectExtractor:
    def __init__(
        self,
        model=None,
        class_names=None,
        exclude_classes=None,
        threshold=0.3
    ):
        self.model = model

        if class_names is not None:
            self.class_names = class_names
        elif model is not None:
            self.class_names = model.class_names
        else:
            raise ValueError(
                "Either class_names or model must be provided."
            )

        self.threshold = threshold

        if exclude_classes is None:
            exclude_classes = ["chair_surface", "ward-item-seg"]

        self.exclude_classes = set(exclude_classes)

    def predict_from_detections(self, detections):
        """
        Extract non-ROI object detections from an existing detector prediction.

        Args:
            detections:
                supervision.Detections from RF-DETRSeg or YOLO26-Seg adapter.

        Returns:
            sv.Detections containing only object classes.
        """

        keep_indices = []

        for idx, class_id in enumerate(detections.class_id):
            class_id = int(class_id)

            if class_id < 0 or class_id >= len(self.class_names):
                continue

            class_name = self.class_names[class_id]

            if class_name in self.exclude_classes:
                continue

            keep_indices.append(idx)

        if len(keep_indices) == 0:
            return self._empty_detections()

        xyxy = detections.xyxy[keep_indices]
        confidence = detections.confidence[keep_indices]
        class_id = detections.class_id[keep_indices]

        mask = None
        if detections.mask is not None:
            mask = detections.mask[keep_indices]

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            mask=mask
        )

    def predict(self, image):
        """
        Legacy convenience method.

        For adapter-based production, use predict_from_detections().
        """

        if self.model is None:
            raise ValueError(
                "predict() requires a model. "
                "Use predict_from_detections() when using detector adapters."
            )

        detections = self.model.predict(
            image,
            threshold=self.threshold
        )

        return self.predict_from_detections(detections)

    def get_labels(self, detections):
        labels = []

        for class_id, confidence in zip(
            detections.class_id,
            detections.confidence
        ):
            class_id = int(class_id)

            if 0 <= class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = "unknown"

            labels.append(f"{class_name} {confidence:.2f}")

        return labels

    def _empty_detections(self):
        return sv.Detections(
            xyxy=np.empty((0, 4)),
            confidence=np.empty((0,)),
            class_id=np.empty((0,), dtype=int)
        )
