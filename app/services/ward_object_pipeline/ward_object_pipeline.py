# ward_object_pipeline.py

import time
import numpy as np
from PIL import Image

from .detectors.rfdetr_adapter import RFDETRSegAdapter
from .detectors.yolo26_adapter import YOLO26SegAdapter

from .roi_polygon import ChairROIPolygonExtractor
from .roi_cropper import create_sam2_bounded_region
from .sam2_roi_segmenter import SAM2ROISegmenter
from .detection_mapper import map_detections_to_full_image
from .rfdetr_object_extractor import RFDETRObjectExtractor
from .detection_matcher import match_sam2_to_rfdetr
from .object_cropper import save_object_crops, create_object_crop_records
from .dinov2_verifier import DINOv2Verifier
from .decision_fusion import ClassificationFusion
from .api_output_formatter import format_api_output, format_failure_output


class WardObjectPipeline:
    def __init__(
        self,

        # Backward-compatible RF-DETR arguments
        rfdetr_weights_path=None,
        dinov2_database_path=None,
        num_classes=11,
        rfdetr_threshold=None,

        # New detector backend arguments
        detector_backend="rfdetr",
        detector_weights_path=None,
        detector_threshold=0.3,

        # YOLO26-specific arguments
        yolo_weights_path=None,
        yolo_imgsz=960,
        yolo_device=0,
        yolo_half=False,

        # Shared pipeline arguments
        roi_class="chair_surface",
        exclude_classes=None,
        roi_simplify_eps=3.0,
        roi_padding=10,

        # SAM2
        sam2_device=0,
        sam2_model_name="facebook/sam2-hiera-small",
        sam2_min_area_ratio=0.002,
        sam2_max_roi_coverage=0.30,
        sam2_min_inside_ratio=0.30,
        sam2_nested_contained_threshold=0.85,
        sam2_nms_iou_threshold=0.5,

        # DINOv2
        dinov2_model_name="facebook/dinov2-base",
        dinov2_device="cuda",
        dinov2_similarity_threshold=0.75,

        # Classification fusion
        min_rfdetr_confidence=0.50,
        require_agreement_for_classification=True,

        # Crops/debug
        crop_padding=8,
        crop_background_value=0,
        crop_output_dir="pipeline_object_crops",
    ):
        """
        Full ward object detection pipeline.

        Supports detector backends:
            - detector_backend="rfdetr"
            - detector_backend="yolo26"

        Final classification rule:
            If detector and DINOv2 both pass and agree:
                final_class = class name
            Otherwise:
                final_class = unknown
        """

        if exclude_classes is None:
            exclude_classes = ["chair_surface", "ward-item-seg"]

        if rfdetr_threshold is not None:
            detector_threshold = rfdetr_threshold

        if detector_weights_path is None:
            if detector_backend == "rfdetr":
                detector_weights_path = rfdetr_weights_path
            elif detector_backend == "yolo26":
                detector_weights_path = yolo_weights_path

        if detector_weights_path is None:
            raise ValueError(
                "detector_weights_path must be provided. "
                "For backward compatibility, rfdetr_weights_path also works "
                "when detector_backend='rfdetr'."
            )

        if dinov2_database_path is None:
            raise ValueError(
                "dinov2_database_path must be provided."
            )

        self.detector_backend = detector_backend
        self.detector_weights_path = detector_weights_path
        self.dinov2_database_path = dinov2_database_path
        self.num_classes = num_classes
        self.roi_class = roi_class
        self.detector_threshold = detector_threshold
        self.roi_padding = roi_padding
        self.crop_padding = crop_padding
        self.crop_background_value = crop_background_value
        self.crop_output_dir = crop_output_dir

        # --------------------------------------------------
        # Load detector backend
        # --------------------------------------------------

        if detector_backend == "rfdetr":
            self.detector = RFDETRSegAdapter(
                weights_path=detector_weights_path,
                num_classes=num_classes,
                threshold=detector_threshold,
                optimize_for_inference=True,
            )

        elif detector_backend == "yolo26":
            self.detector = YOLO26SegAdapter(
                weights_path=detector_weights_path,
                threshold=detector_threshold,
                device=yolo_device,
                imgsz=yolo_imgsz,
                half=yolo_half,
                retina_masks=True,
                verbose=False,
            )

        else:
            raise ValueError(
                f"Unsupported detector_backend: {detector_backend}. "
                "Use 'rfdetr' or 'yolo26'."
            )

        self.class_names = self.detector.class_names

        # --------------------------------------------------
        # ROI extractor
        # --------------------------------------------------

        self.roi_extractor = ChairROIPolygonExtractor(
            model=None,
            class_names=self.class_names,
            roi_class=roi_class,
            threshold=detector_threshold,
            simplify_eps=roi_simplify_eps
        )

        # --------------------------------------------------
        # Object extractor
        # --------------------------------------------------

        self.object_extractor = RFDETRObjectExtractor(
            model=None,
            class_names=self.class_names,
            exclude_classes=exclude_classes,
            threshold=detector_threshold
        )

        # --------------------------------------------------
        # SAM2 ROI segmenter
        # --------------------------------------------------

        self.sam2_segmenter = SAM2ROISegmenter(
            device=sam2_device,
            model_name=sam2_model_name,
            min_area_ratio=sam2_min_area_ratio,
            max_roi_coverage=sam2_max_roi_coverage,
            min_inside_ratio=sam2_min_inside_ratio,
            nested_contained_threshold=sam2_nested_contained_threshold,
            nms_iou_threshold=sam2_nms_iou_threshold,
        )

        # --------------------------------------------------
        # DINOv2 verifier
        # --------------------------------------------------

        self.dinov2_verifier = DINOv2Verifier(
            database_path=dinov2_database_path,
            model_name=dinov2_model_name,
            device=dinov2_device,
            similarity_threshold=dinov2_similarity_threshold
        )

        # --------------------------------------------------
        # Classification fusion
        # --------------------------------------------------

        self.classification_fusion = ClassificationFusion(
            min_rfdetr_confidence=min_rfdetr_confidence,
            min_dinov2_similarity=dinov2_similarity_threshold,
            require_agreement=require_agreement_for_classification
        )

    def predict(
        self,
        image,
        save_crops=False,
        crop_output_dir=None,
        return_intermediate=False,
        verbose=True,
    ):
        """
        Run the full pipeline on one image.
        """

        total_t0 = time.perf_counter()

        image_pil, image_np = self._load_image(image)

        # --------------------------------------------------
        # 1. Detector runs once
        # --------------------------------------------------

        t0 = time.perf_counter()

        if verbose:
            print(f"Running detector once: {self.detector_backend}")

        detector_output = self.detector.predict(
            image_pil=image_pil,
            image_np=image_np,
        )

        detector_all_detections = detector_output.to_supervision()

        t1 = time.perf_counter()

        if verbose:
            print(f"[TIME] Detector ({self.detector_backend}): {t1 - t0:.3f}s")

        # --------------------------------------------------
        # 2. Extract chair ROI from same detector result
        # --------------------------------------------------

        roi = self.roi_extractor.predict_from_detections(
            detector_all_detections
        )

        if roi is None:
            return self._failure_result(
                reason=f"No {self.roi_class} ROI found",
                image_np=image_np,
                detector_all_detections=detector_all_detections,
                return_intermediate=return_intermediate
            )

        # --------------------------------------------------
        # 3. Extract object detections from same detector result
        # --------------------------------------------------

        detector_objects = self.object_extractor.predict_from_detections(
            detector_all_detections
        )

        t2 = time.perf_counter()

        if verbose:
            print(f"[TIME] ROI + detector objects: {t2 - t1:.3f}s")
            print("Detector object detections:", len(detector_objects))

        # --------------------------------------------------
        # 4. Create bounded region for SAM2
        # --------------------------------------------------

        bounded = create_sam2_bounded_region(
            image_np=image_np,
            polygon=roi["polygon"],
            padding=self.roi_padding
        )

        # --------------------------------------------------
        # 5. SAM2 runs once inside chair ROI crop
        # --------------------------------------------------

        local_sam = self.sam2_segmenter.predict(
            crop_image=bounded["crop_image"],
            local_polygon_mask=bounded["local_polygon_mask"]
        )

        full_sam = map_detections_to_full_image(
            local_detections=local_sam,
            full_image_shape=image_np.shape,
            offset=bounded["offset"]
        )

        t3 = time.perf_counter()

        if verbose:
            print("SAM2 object proposals:", len(full_sam))
            print(f"[TIME] SAM2: {t3 - t2:.3f}s")

        # --------------------------------------------------
        # 6. Match SAM2 masks to detector detections
        # --------------------------------------------------

        matched = match_sam2_to_rfdetr(
            sam_detections=full_sam,
            rfdetr_detections=detector_objects,
            class_names=self.class_names,
            min_box_iou=0.15,
            min_mask_inside_box=0.30
        )

        # --------------------------------------------------
        # 7. Create masked crops for DINOv2
        # --------------------------------------------------

        if crop_output_dir is None:
            crop_output_dir = self.crop_output_dir

        if save_crops:
            crop_records = save_object_crops(
                image_np=image_np,
                matched_results=matched,
                output_dir=crop_output_dir,
                padding=self.crop_padding,
                background_value=self.crop_background_value
            )
        else:
            crop_records = create_object_crop_records(
                image_np=image_np,
                matched_results=matched,
                padding=self.crop_padding,
                background_value=self.crop_background_value
            )

        t4 = time.perf_counter()

        if verbose:
            print(f"[TIME] Matching + crops: {t4 - t3:.3f}s")

        # --------------------------------------------------
        # 8. DINOv2 verification
        # --------------------------------------------------

        verified = self.dinov2_verifier.verify_crops(
            crop_records
        )

        t5 = time.perf_counter()

        if verbose:
            print(f"[TIME] DINOv2: {t5 - t4:.3f}s")

        # --------------------------------------------------
        # 9. Classification fusion
        # --------------------------------------------------

        classified = self.classification_fusion.classify_all(
            verified
        )

        t6 = time.perf_counter()

        if verbose:
            print(f"[TIME] Classification fusion: {t6 - t5:.3f}s")
            print(f"[TIME] Total predict: {t6 - total_t0:.3f}s")

        # --------------------------------------------------
        # 10. Minimal API output
        # --------------------------------------------------

        api_output = format_api_output(
            classified_items=classified
        )

        messages = self._build_messages(classified)

        api_output["messages"] = messages

        # Keep internal full detail for review/debugging.
        api_output["final_results"] = classified

        if return_intermediate:
            api_output["intermediate"] = {
                "roi": roi,
                "bounded": bounded,
                "detector_backend": self.detector_backend,
                "detector_all_detections": detector_all_detections,
                "detector_objects": detector_objects,
                "local_sam": local_sam,
                "full_sam": full_sam,
                "matched": matched,
                "crop_records": crop_records,
                "verified": verified,
            }

        return api_output

    def _load_image(self, image):
        if isinstance(image, str):
            image_pil = Image.open(image).convert("RGB")
            image_np = np.array(image_pil)
            return image_pil, image_np

        if isinstance(image, Image.Image):
            image_pil = image.convert("RGB")
            image_np = np.array(image_pil)
            return image_pil, image_np

        if isinstance(image, np.ndarray):
            image_np = image.astype(np.uint8)

            if image_np.ndim != 3 or image_np.shape[2] != 3:
                raise ValueError(
                    "Numpy image must be RGB with shape HxWx3"
                )

            image_pil = Image.fromarray(image_np).convert("RGB")
            return image_pil, image_np

        raise TypeError(
            f"Unsupported image type: {type(image)}"
        )

    def _build_messages(self, classified_items):
        messages = []

        detector_label = (
            "RF-DETR"
            if self.detector_backend == "rfdetr"
            else self.detector_backend.upper()
        )

        for idx, item in enumerate(classified_items):
            messages.append(
                (
                    f"obj_{idx:02d} | "
                    f"class={item.get('final_class', 'unknown')} | "
                    f"{detector_label}={item.get('class_name', 'unknown')} "
                    f"({item.get('rfdetr_confidence', 0.0):.2f}) | "
                    f"DINOv2={item.get('dinov2_label', 'unknown')} "
                    f"({item.get('dinov2_score', 0.0):.3f}) | "
                    f"agreement={item.get('agreement', False)} | "
                    f"reason={item.get('classification_reason', '')}"
                )
            )

        return messages

    def _failure_result(
        self,
        reason,
        image_np=None,
        detector_all_detections=None,
        return_intermediate=False
    ):
        output = format_failure_output(reason)

        output["messages"] = [
            f"Pipeline failed: {reason}"
        ]

        output["final_results"] = []

        if return_intermediate:
            output["intermediate"] = {
                "image_np": image_np,
                "detector_backend": self.detector_backend,
                "detector_all_detections": detector_all_detections,
            }

        return output
