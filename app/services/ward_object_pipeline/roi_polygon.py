# roi_polygon.py

import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon


class ChairROIPolygonExtractor:

    def __init__(
        self,
        model=None,
        class_names=None,
        roi_class="chair_surface",
        threshold=0.3,
        simplify_eps=3.0
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

        self.roi_class = roi_class
        self.threshold = threshold
        self.simplify_eps = simplify_eps

    def predict(self, image):
        """
        Convenience method:
        Runs detector internally, then extracts chair ROI.

        Good for standalone testing.
        In production, prefer predict_from_detections().
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

    def predict_from_detections(self, detections):
        """
        Extract chair_surface ROI from an existing detector prediction.

        This intentionally uses the detector's actual chair_surface mask.
        It does NOT fill holes, does NOT use convex hull, and does NOT expand
        the platform.

        Anything outside this mask/polygon is outside the ROI.
        """

        candidates = []

        for idx, class_id in enumerate(detections.class_id):
            class_id = int(class_id)

            if class_id < 0 or class_id >= len(self.class_names):
                continue

            class_name = self.class_names[class_id]

            if class_name == self.roi_class:
                candidates.append(idx)

        if len(candidates) == 0:
            return None

        best_idx = max(
            candidates,
            key=lambda i: detections.confidence[i]
        )

        if detections.mask is None:
            print("Detector output has no mask output.")
            return None

        mask = detections.mask[best_idx].astype(bool)
        confidence = float(detections.confidence[best_idx])
        class_id = int(detections.class_id[best_idx])

        polygon = self._mask_to_polygon(mask)

        if polygon is None:
            return None

        polygon = polygon.simplify(self.simplify_eps)
        polygon = self._ensure_single_polygon(polygon)

        if polygon is None or polygon.is_empty:
            return None

        polygon_coords = np.array(
            polygon.exterior.coords,
            dtype=np.float32
        )

        return {
            "mask": mask,
            "polygon": polygon_coords,
            "confidence": confidence,
            "class_id": class_id,
            "det_index": best_idx,
        }

    def _mask_to_polygon(self, mask):
        """
        Convert binary mask to polygon using OpenCV contours.

        This keeps the original logic:
        - find external contours
        - use the largest contour
        - convert it to a polygon
        """

        mask_uint8 = mask.astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        largest = max(
            contours,
            key=cv2.contourArea
        )

        coords = largest.squeeze()

        if coords.ndim != 2:
            return None

        if len(coords) < 3:
            return None

        polygon = Polygon(coords)

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        polygon = self._ensure_single_polygon(polygon)

        if polygon is None or polygon.is_empty:
            return None

        return polygon

    def _ensure_single_polygon(self, geometry):
        """
        YOLO masks can occasionally produce MultiPolygon after buffer/simplify.
        To keep the old behavior, choose the largest polygon only.
        """

        if geometry is None:
            return None

        if isinstance(geometry, Polygon):
            return geometry

        if isinstance(geometry, MultiPolygon):
            if len(geometry.geoms) == 0:
                return None

            return max(
                geometry.geoms,
                key=lambda p: p.area
            )

        if hasattr(geometry, "geoms"):
            polygons = [
                g for g in geometry.geoms
                if isinstance(g, Polygon) and not g.is_empty
            ]

            if len(polygons) == 0:
                return None

            return max(
                polygons,
                key=lambda p: p.area
            )

        return None
