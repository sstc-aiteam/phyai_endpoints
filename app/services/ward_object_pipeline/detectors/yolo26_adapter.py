# detectors/yolo26_adapter.py

import cv2
import numpy as np
from ultralytics import YOLO

from .detector_output import DetectorOutput


class YOLO26SegAdapter:
    def __init__(
        self,
        weights_path,
        threshold=0.3,
        device=0,
        imgsz=960,
        half=False,
        retina_masks=True,
        verbose=False,
    ):
        self.weights_path = weights_path
        self.threshold = threshold
        self.device = device
        self.imgsz = imgsz
        self.half = half
        self.retina_masks = retina_masks
        self.verbose = verbose

        print("Loading YOLO26-Seg detector...")

        self.model = YOLO(weights_path)

        names = self.model.names

        if isinstance(names, dict):
            max_id = max(names.keys())
            self.class_names = [
                names.get(i, f"class_{i}")
                for i in range(max_id + 1)
            ]
        else:
            self.class_names = list(names)

        print("YOLO26 classes:")
        print(self.class_names)

    def predict(self, image_pil, image_np=None):
        if image_np is None:
            image_np = np.array(image_pil.convert("RGB"))

        height, width = image_np.shape[:2]

        results = self.model.predict(
            source=image_np,
            conf=self.threshold,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            retina_masks=self.retina_masks,
            verbose=self.verbose,
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return DetectorOutput(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                mask=np.zeros((0, height, width), dtype=bool),
                class_id=np.zeros((0,), dtype=np.int64),
                confidence=np.zeros((0,), dtype=np.float32),
                class_names=self.class_names,
            )

        xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        class_id = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        confidence = result.boxes.conf.detach().cpu().numpy().astype(np.float32)

        if result.masks is None:
            masks = np.zeros((len(xyxy), height, width), dtype=bool)
        else:
            raw_masks = result.masks.data.detach().cpu().numpy()

            normalized_masks = []

            for mask in raw_masks:
                mask = mask.astype(np.float32)

                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(
                        mask,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )

                normalized_masks.append(mask > 0.5)

            masks = np.stack(normalized_masks, axis=0).astype(bool)

        return DetectorOutput(
            xyxy=xyxy,
            mask=masks,
            class_id=class_id,
            confidence=confidence,
            class_names=self.class_names,
        )
