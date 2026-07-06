# sam2_roi_segmenter.py

import numpy as np
import torch
import torchvision
import supervision as sv
from PIL import Image
from transformers import pipeline
import cv2


class SAM2ROISegmenter:

    def __init__(
        self,
        device=0,
        model_name="facebook/sam2-hiera-small",
        min_area_ratio=0.002,
        max_roi_coverage=0.30,
        min_inside_ratio=0.30,
        nested_contained_threshold=0.85,
        nms_iou_threshold=0.5,
        keep_largest_component=True,
    ):
        self.min_area_ratio = min_area_ratio
        self.max_roi_coverage = max_roi_coverage
        self.min_inside_ratio = min_inside_ratio
        self.nested_contained_threshold = nested_contained_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.keep_largest_component = keep_largest_component

        self.sam_generator = pipeline(
            "mask-generation",
            model=model_name,
            device=device,
            points_per_batch=128,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            crop_nms_thresh=0.3,
        )

    def predict(self, crop_image, local_polygon_mask):
        """
        Run SAM2 on cropped chair ROI.

        Args:
            crop_image:
                RGB numpy image in crop/local coordinates.

            local_polygon_mask:
                Binary mask in crop/local coordinates.
                1 = chair surface
                0 = outside chair surface

        Returns:
            sv.Detections in crop/local coordinates.
        """

        image_area = crop_image.shape[0] * crop_image.shape[1]

        roi_mask_bool = local_polygon_mask.astype(bool)
        roi_area = np.sum(roi_mask_bool)

        if roi_area == 0:
            return self._empty_detections(crop_image)

        print("Running SAM2 inside chair ROI crop...")
        print("crop_image shape:", crop_image.shape)
        print("local_polygon_mask shape:", local_polygon_mask.shape)

        crop_pil = Image.fromarray(
            crop_image.astype(np.uint8)
        ).convert("RGB")

        sam_output = self.sam_generator(crop_pil)

        filtered_masks = []
        filtered_xyxy = []
        scores = []

        for mask_data in sam_output["masks"]:

            if isinstance(mask_data, dict):
                mask_array = np.array(mask_data["mask"])
            else:
                mask_array = np.array(mask_data)

            mask_array = mask_array.astype(bool)
            raw_mask_area = np.sum(mask_array)

            if raw_mask_area == 0:
                continue

            # --------------------------------------------------
            # Filter 1:
            # Ignore tiny raw SAM masks.
            # --------------------------------------------------
            if (raw_mask_area / image_area) < self.min_area_ratio:
                continue

            # --------------------------------------------------
            # Filter 2:
            # Check how much of the SAM mask is inside chair ROI.
            # --------------------------------------------------
            pixels_inside_roi = np.sum(
                np.logical_and(mask_array, roi_mask_bool)
            )

            inside_ratio = pixels_inside_roi / raw_mask_area

            if inside_ratio < self.min_inside_ratio:
                continue

            # --------------------------------------------------
            # Clip SAM mask to chair ROI polygon.
            # --------------------------------------------------
            clipped_mask = np.logical_and(
                mask_array,
                roi_mask_bool
            )

            clipped_area = np.sum(clipped_mask)

            if clipped_area == 0:
                continue

            # --------------------------------------------------
            # IMPORTANT FIX:
            # Remove disconnected mask fragments.
            # This prevents boxes from becoming huge because of
            # small pixels far away from the actual object.
            # --------------------------------------------------
            if self.keep_largest_component:
                clipped_mask = self._largest_connected_component(
                    clipped_mask
                )

            cleaned_area = np.sum(clipped_mask)

            if cleaned_area == 0:
                continue

            # --------------------------------------------------
            # Filter 3:
            # Reject masks that cover too much of the chair surface.
            # Usually this means SAM detected the chair/floor instead
            # of an object.
            # --------------------------------------------------
            if (cleaned_area / roi_area) > self.max_roi_coverage:
                continue

            # --------------------------------------------------
            # Compute bbox AFTER clipping + component cleanup.
            # --------------------------------------------------
            clean_box = self._bbox_from_mask(clipped_mask)

            if clean_box is None:
                continue

            filtered_masks.append(clipped_mask)
            filtered_xyxy.append(clean_box)
            scores.append(float(cleaned_area))

        # ------------------------------------------------------
        # Remove sub-part masks.
        # Example:
        # remote control + screen detected separately.
        # Keep the larger object mask.
        # ------------------------------------------------------
        filtered_masks, filtered_xyxy, scores = self._remove_nested_masks(
            filtered_masks,
            filtered_xyxy,
            scores,
            contained_threshold=self.nested_contained_threshold
        )

        # ------------------------------------------------------
        # NMS for duplicate overlapping SAM proposals.
        # ------------------------------------------------------
        if len(filtered_xyxy) > 0:

            boxes_tensor = torch.tensor(
                filtered_xyxy,
                dtype=torch.float32
            )

            scores_tensor = torch.tensor(
                scores,
                dtype=torch.float32
            )

            keep_indices = torchvision.ops.nms(
                boxes_tensor,
                scores_tensor,
                iou_threshold=self.nms_iou_threshold
            )

            sam_masks = [
                filtered_masks[int(i)]
                for i in keep_indices
            ]

            sam_xyxy = [
                filtered_xyxy[int(i)]
                for i in keep_indices
            ]

            sam_scores = [
                scores[int(i)]
                for i in keep_indices
            ]

        else:
            sam_masks = []
            sam_xyxy = []
            sam_scores = []

        num_masks = len(sam_xyxy)

        return sv.Detections(
            xyxy=np.array(sam_xyxy)
            if num_masks > 0
            else np.empty((0, 4)),

            mask=np.array(sam_masks)
            if num_masks > 0
            else np.empty(
                (0, crop_image.shape[0], crop_image.shape[1]),
                dtype=bool
            ),

            class_id=np.arange(num_masks)
            if num_masks > 0
            else np.empty((0,), dtype=int),

            confidence=np.array(sam_scores)
            if num_masks > 0
            else np.empty((0,))
        )

    def _largest_connected_component(self, mask):
        """
        Keep only the largest connected component in a binary mask.

        This removes disconnected fragments that can make the bbox too large.
        """

        mask_uint8 = mask.astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_uint8,
            connectivity=8
        )

        # label 0 is background
        if num_labels <= 1:
            return mask.astype(bool)

        component_areas = stats[1:, cv2.CC_STAT_AREA]

        largest_component_label = 1 + np.argmax(component_areas)

        largest_component_mask = labels == largest_component_label

        return largest_component_mask.astype(bool)

    def _bbox_from_mask(self, mask):
        """
        Compute tight xyxy bbox from cleaned binary mask.
        """

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        return [xmin, ymin, xmax, ymax]

    def _remove_nested_masks(
        self,
        masks,
        boxes,
        scores,
        contained_threshold=0.85
    ):
        """
        Remove masks that are mostly contained inside larger masks.

        Example:
            remote control
                └── screen

        The screen mask is removed if it is mostly inside the remote mask.
        """

        if len(masks) <= 1:
            return masks, boxes, scores

        keep = [True] * len(masks)

        areas = [
            np.sum(m.astype(bool))
            for m in masks
        ]

        for i in range(len(masks)):

            if not keep[i]:
                continue

            for j in range(len(masks)):

                if i == j or not keep[j]:
                    continue

                if areas[i] == 0:
                    keep[i] = False
                    break

                intersection = np.logical_and(
                    masks[i],
                    masks[j]
                ).sum()

                contained_ratio = intersection / areas[i]

                if (
                    contained_ratio >= contained_threshold
                    and areas[i] < areas[j]
                ):
                    keep[i] = False
                    break

        new_masks = [
            m for m, k in zip(masks, keep)
            if k
        ]

        new_boxes = [
            b for b, k in zip(boxes, keep)
            if k
        ]

        new_scores = [
            s for s, k in zip(scores, keep)
            if k
        ]

        return new_masks, new_boxes, new_scores

    def _empty_detections(self, image):
        return sv.Detections(
            xyxy=np.empty((0, 4)),

            mask=np.empty(
                (0, image.shape[0], image.shape[1]),
                dtype=bool
            ),

            class_id=np.empty((0,), dtype=int),

            confidence=np.empty((0,))
        )
