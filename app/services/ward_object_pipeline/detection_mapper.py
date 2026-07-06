# detection_mapper.py

import numpy as np
import supervision as sv


def map_detections_to_full_image(local_detections, full_image_shape, offset):
    """
    Convert SAM2 detections from crop/local coordinates back to full-image coordinates.

    Args:
        local_detections:
            sv.Detections from SAM2 on crop image.

        full_image_shape:
            Original image shape, e.g. image_np.shape.

        offset:
            (x_offset, y_offset) from crop_box.

    Returns:
        sv.Detections in original full-image coordinates.
    """

    x_offset, y_offset = offset
    full_h, full_w = full_image_shape[:2]

    if len(local_detections) == 0:
        return sv.Detections(
            xyxy=np.empty((0, 4)),
            mask=np.empty((0, full_h, full_w), dtype=bool),
            class_id=np.empty((0,), dtype=int),
            confidence=np.empty((0,))
        )

    # -----------------------------
    # Map boxes
    # -----------------------------
    full_xyxy = local_detections.xyxy.copy()

    full_xyxy[:, [0, 2]] += x_offset
    full_xyxy[:, [1, 3]] += y_offset

    # -----------------------------
    # Map masks
    # -----------------------------
    full_masks = []

    for local_mask in local_detections.mask:
        full_mask = np.zeros((full_h, full_w), dtype=bool)

        local_h, local_w = local_mask.shape[:2]

        y1 = y_offset
        y2 = y_offset + local_h
        x1 = x_offset
        x2 = x_offset + local_w

        # Safety clamp
        y2 = min(y2, full_h)
        x2 = min(x2, full_w)

        valid_h = y2 - y1
        valid_w = x2 - x1

        full_mask[y1:y2, x1:x2] = local_mask[:valid_h, :valid_w]

        full_masks.append(full_mask)

    full_masks = np.array(full_masks)

    return sv.Detections(
        xyxy=full_xyxy,
        mask=full_masks,
        class_id=local_detections.class_id.copy()
        if local_detections.class_id is not None
        else np.arange(len(full_xyxy)),
        confidence=local_detections.confidence.copy()
        if local_detections.confidence is not None
        else np.ones(len(full_xyxy))
    )
