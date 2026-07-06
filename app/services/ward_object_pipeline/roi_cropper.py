# roi_cropper.py

import numpy as np
import cv2


def polygon_to_full_mask(image_shape, polygon):
    height, width = image_shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)
    polygon_int = polygon.astype(np.int32)

    cv2.fillPoly(mask, [polygon_int], 1)

    return mask


def get_polygon_crop_box(image_shape, polygon, padding=10):
    height, width = image_shape[:2]

    x1 = int(np.floor(polygon[:, 0].min())) - padding
    y1 = int(np.floor(polygon[:, 1].min())) - padding
    x2 = int(np.ceil(polygon[:, 0].max())) + padding
    y2 = int(np.ceil(polygon[:, 1].max())) + padding

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    return [x1, y1, x2, y2]


def create_sam2_bounded_region(image_np, polygon, padding=10):
    """
    Creates the minimum data SAM2 needs:
    - cropped image region
    - local polygon mask for filtering SAM2 results
    - offset for mapping back to full image
    """

    crop_box = get_polygon_crop_box(
        image_shape=image_np.shape,
        polygon=polygon,
        padding=padding
    )

    x1, y1, x2, y2 = crop_box

    crop_image = image_np[y1:y2, x1:x2]

    full_polygon_mask = polygon_to_full_mask(
        image_shape=image_np.shape,
        polygon=polygon
    )

    local_polygon_mask = full_polygon_mask[y1:y2, x1:x2]

    local_polygon = polygon.copy()
    local_polygon[:, 0] -= x1
    local_polygon[:, 1] -= y1

    return {
        "crop_image": crop_image,
        "local_polygon_mask": local_polygon_mask,
        "local_polygon": local_polygon,
        "full_polygon_mask": full_polygon_mask,
        "crop_box": crop_box,
        "offset": (x1, y1)
    }


def filter_mask_by_polygon(sam_mask, local_polygon_mask, min_inside_ratio=0.8):
    """
    Check whether a SAM2 mask mostly lies inside the chair polygon.
    """

    sam_mask_bool = sam_mask.astype(bool)
    roi_mask_bool = local_polygon_mask.astype(bool)

    sam_area = sam_mask_bool.sum()

    if sam_area == 0:
        return False, 0.0, sam_mask_bool

    inside = sam_mask_bool & roi_mask_bool
    inside_area = inside.sum()

    inside_ratio = inside_area / sam_area

    clipped_mask = inside

    keep = inside_ratio >= min_inside_ratio

    return keep, inside_ratio, clipped_mask
