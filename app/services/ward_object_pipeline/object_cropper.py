# object_cropper.py

import os
import numpy as np
from PIL import Image


def crop_object_from_mask(
    image_np,
    mask,
    bbox=None,
    padding=8,
    background_value=0
):
    """
    Create a masked crop for one object.

    Args:
        image_np:
            Full RGB image as numpy array, shape HxWx3.

        mask:
            Full-image binary object mask, shape HxW.

        bbox:
            Optional xyxy bbox. If None, bbox is derived from mask.

        padding:
            Extra pixels around object crop.

        background_value:
            Pixel value outside mask. 0 = black background.

    Returns:
        {
            "crop": masked RGB crop,
            "raw_crop": raw RGB crop,
            "crop_mask": local binary mask,
            "crop_box": [x1, y1, x2, y2]
        }
    """

    h, w = image_np.shape[:2]
    mask_bool = mask.astype(bool)

    if bbox is None:
        bbox = bbox_from_mask(mask_bool)

    if bbox is None:
        return None

    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    raw_crop = image_np[y1:y2, x1:x2].copy()
    crop_mask = mask_bool[y1:y2, x1:x2]

    masked_crop = raw_crop.copy()
    masked_crop[~crop_mask] = background_value

    return {
        "crop": masked_crop,
        "raw_crop": raw_crop,
        "crop_mask": crop_mask,
        "crop_box": [x1, y1, x2, y2]
    }


def bbox_from_mask(mask):
    """
    Compute tight xyxy bbox from binary mask.
    """

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return [x1, y1, x2, y2]


def create_object_crop_records(
    image_np,
    matched_results,
    padding=8,
    background_value=0
):
    """
    Create in-memory crop records for DINOv2.

    Important:
        This keeps the original full-image mask and bbox,
        so the final API output can return:
            - class_name
            - bbox
            - rle_mask
    """

    crop_records = []

    for idx, result in enumerate(matched_results):

        class_name = result["class_name"]
        mask = result["mask"]
        bbox = result["bbox"]

        crop_data = crop_object_from_mask(
            image_np=image_np,
            mask=mask,
            bbox=bbox,
            padding=padding,
            background_value=background_value
        )

        if crop_data is None:
            continue

        record = {
            "index": idx,

            # --------------------------------------------------
            # KEEP THESE FOR FINAL API OUTPUT
            # --------------------------------------------------
            "mask": mask,
            "bbox": bbox,

            # --------------------------------------------------
            # RF-DETR / matching metadata
            # --------------------------------------------------
            "class_name": class_name,
            "matched": result.get("matched", False),
            "class_id": result.get("class_id", None),
            "rfdetr_confidence": result.get("rfdetr_confidence", 0.0),
            "box_iou": result.get("box_iou", 0.0),
            "mask_inside_box": result.get("mask_inside_box", 0.0),
            "match_score": result.get("match_score", 0.0),

            # --------------------------------------------------
            # DINOv2 crop data
            # --------------------------------------------------
            "crop_box": crop_data["crop_box"],
            "masked_crop": crop_data["crop"],
            "raw_crop": crop_data["raw_crop"],
            "crop_mask": crop_data["crop_mask"],

            # Optional debug paths
            "masked_path": None,
            "raw_path": None,
            "mask_path": None,
        }

        crop_records.append(record)

    return crop_records


def save_object_crops(
    image_np,
    matched_results,
    output_dir="object_crops",
    padding=8,
    background_value=0
):
    """
    Save object crops to disk for debugging.

    Also returns the same in-memory records used by DINOv2 and final API output.
    """

    os.makedirs(output_dir, exist_ok=True)

    crop_records = create_object_crop_records(
        image_np=image_np,
        matched_results=matched_results,
        padding=padding,
        background_value=background_value
    )

    for record in crop_records:

        idx = record["index"]
        class_name = record["class_name"]

        safe_class_name = class_name.replace("/", "_").replace(" ", "_")

        masked_path = os.path.join(
            output_dir,
            f"obj_{idx:02d}_{safe_class_name}_masked.png"
        )

        raw_path = os.path.join(
            output_dir,
            f"obj_{idx:02d}_{safe_class_name}_raw.png"
        )

        mask_path = os.path.join(
            output_dir,
            f"obj_{idx:02d}_{safe_class_name}_mask.png"
        )

        Image.fromarray(
            record["masked_crop"].astype(np.uint8)
        ).save(masked_path)

        Image.fromarray(
            record["raw_crop"].astype(np.uint8)
        ).save(raw_path)

        Image.fromarray(
            (record["crop_mask"].astype(np.uint8) * 255)
        ).save(mask_path)

        record["masked_path"] = masked_path
        record["raw_path"] = raw_path
        record["mask_path"] = mask_path

    return crop_records
