# detection_matcher.py

import numpy as np
import supervision as sv


def box_iou(box_a, box_b):
    """
    Compute IoU between two xyxy boxes.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def mask_box_overlap(mask, box):
    """
    Measure how much of a SAM2 mask lies inside a detector bbox.

    This is often better than bbox IoU because SAM2 gives precise masks.
    """

    x1, y1, x2, y2 = map(int, box)

    h, w = mask.shape[:2]

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    mask_area = np.sum(mask)

    if mask_area == 0:
        return 0.0

    inside_area = np.sum(mask[y1:y2, x1:x2])

    return inside_area / mask_area


def match_sam2_to_rfdetr(
    sam_detections,
    rfdetr_detections,
    class_names=None,
    model=None,
    min_box_iou=0.15,
    min_mask_inside_box=0.30
):
    """
    Match each SAM2 object proposal to the best detector object detection.

    Kept function name for backward compatibility.

    Args:
        sam_detections:
            sv.Detections from SAM2 in full-image coordinates.

        rfdetr_detections:
            sv.Detections from RF-DETRSeg or YOLO26-Seg adapter,
            excluding chair_surface.

        class_names:
            Detector class names.

        model:
            Optional legacy RF-DETR model. Used only if class_names is None.

    Returns:
        list of dicts.
    """

    if class_names is None:
        if model is None:
            raise ValueError(
                "Either class_names or model must be provided."
            )
        class_names = model.class_names

    results = []

    for sam_idx in range(len(sam_detections)):
        sam_box = sam_detections.xyxy[sam_idx]
        sam_mask = sam_detections.mask[sam_idx]

        best = {
            "rfdetr_index": None,
            "score": 0.0,
            "box_iou": 0.0,
            "mask_inside_box": 0.0
        }

        for det_idx in range(len(rfdetr_detections)):
            det_box = rfdetr_detections.xyxy[det_idx]

            iou = box_iou(sam_box, det_box)
            inside = mask_box_overlap(sam_mask, det_box)

            score = (0.4 * iou) + (0.6 * inside)

            if score > best["score"]:
                best = {
                    "rfdetr_index": det_idx,
                    "score": score,
                    "box_iou": iou,
                    "mask_inside_box": inside
                }

        matched = (
            best["rfdetr_index"] is not None
            and (
                best["box_iou"] >= min_box_iou
                or best["mask_inside_box"] >= min_mask_inside_box
            )
        )

        if matched:
            det_idx = best["rfdetr_index"]
            class_id = int(rfdetr_detections.class_id[det_idx])

            if 0 <= class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = "unknown"

            rfdetr_conf = float(rfdetr_detections.confidence[det_idx])
        else:
            det_idx = None
            class_id = None
            class_name = "unknown"
            rfdetr_conf = 0.0

        results.append(
            {
                "sam_index": sam_idx,
                "rfdetr_index": det_idx,
                "bbox": sam_box,
                "mask": sam_mask,
                "matched": matched,
                "class_id": class_id,
                "class_name": class_name,
                "rfdetr_confidence": rfdetr_conf,
                "box_iou": float(best["box_iou"]),
                "mask_inside_box": float(best["mask_inside_box"]),
                "match_score": float(best["score"]),
            }
        )

    return results


def matched_results_to_detections(matched_results):
    """
    Convert matched result dicts into sv.Detections for visualization.
    Uses SAM2 masks/boxes, but class_id/confidence from matched detector.
    Unknown objects get class_id=-1 and confidence=0.
    """

    if len(matched_results) == 0:
        return sv.Detections(
            xyxy=np.empty((0, 4)),
            mask=np.empty((0, 1, 1), dtype=bool),
            class_id=np.empty((0,), dtype=int),
            confidence=np.empty((0,))
        )

    xyxy = np.array([r["bbox"] for r in matched_results])
    masks = np.array([r["mask"] for r in matched_results])

    class_ids = np.array([
        r["class_id"] if r["class_id"] is not None else -1
        for r in matched_results
    ])

    confidence = np.array([
        r["rfdetr_confidence"]
        for r in matched_results
    ])

    return sv.Detections(
        xyxy=xyxy,
        mask=masks,
        class_id=class_ids,
        confidence=confidence
    )
