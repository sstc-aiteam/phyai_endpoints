import math
import cv2
import numpy as np

# Vibrant HSV-full-saturation palette in BGR for per-object coloring
_PALETTE: list[tuple[int, int, int]] = [
    (  0,   0, 255),  # Red
    (  0, 128, 255),  # Orange
    (  0, 220, 255),  # Gold
    (  0, 255, 128),  # Spring Green
    (  0, 255,   0),  # Green
    (255, 255,   0),  # Cyan
    (255, 128,   0),  # Azure
    (255,   0,   0),  # Blue
    (255,   0, 128),  # Blue-Violet
    (255,   0, 255),  # Magenta
    (128,   0, 255),  # Rose
    (  0,   0, 200),  # Dark Red (fallback contrast)
]


def palette_color(idx: int) -> tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


def draw_detection_annotation(
    image,
    bbox,
    pixel_coords=None,
    label=None,
    color=(0, 255, 0),
    class_name: str = "",
    skip_classes: list[str] | None = None,
):
    if image is None or bbox is None:
        return
    if skip_classes and class_name in skip_classes:
        return

    x1, y1, x2, y2 = map(int, bbox)
    if pixel_coords is None:
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
    else:
        u, v = map(int, pixel_coords)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.circle(image, (u, v), 6, (255, 255, 255), -1)
    cv2.circle(image, (u, v), 4, color, -1)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        lx, ly = x1, max(y1 - 10, th + 4)
        cv2.rectangle(image, (lx, ly - th - baseline), (lx + tw + 4, ly + baseline), color, -1)
        cv2.putText(image, label, (lx + 2, ly), font, font_scale, (0, 0, 0), thickness)


def draw_yaw_annotation(
    image,
    bbox,
    pixel_coords,
    object_yaw_deg,
    object_yaw_rad,
    show_label=False,
    show_unavailable_label=False,
    label_position="below",
    color=(255, 0, 255),
    class_name: str = "",
    skip_classes: list[str] | None = None,
):
    if image is None or bbox is None or pixel_coords is None:
        return
    if skip_classes and class_name in skip_classes:
        return

    x1, y1, x2, y2 = map(int, bbox)
    u, v = map(int, pixel_coords)
    image_h = image.shape[0]

    if label_position == "above":
        label_origin = (x1, max(20, y1 - 10))
    else:
        label_origin = (x1, min(image_h - 10, y2 + 22))

    if object_yaw_deg is None or object_yaw_rad is None:
        if show_unavailable_label:
            cv2.putText(image, "yaw: n/a", label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return

    axis_len = int(max(x2 - x1, y2 - y1) * 0.45)
    dx = math.sin(object_yaw_rad) * axis_len
    dy = math.cos(object_yaw_rad) * axis_len
    cv2.line(image, (int(u - dx), int(v - dy)), (int(u + dx), int(v + dy)), color, 2)

    if show_label:
        yaw_text = f"yaw: {object_yaw_deg:.1f} deg"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(yaw_text, font, font_scale, thickness)
        lx, ly = label_origin
        cv2.rectangle(image, (lx, ly - th - baseline), (lx + tw + 4, ly + baseline), color, -1)
        cv2.putText(image, yaw_text, (lx + 2, ly), font, font_scale, (0, 0, 0), thickness)


def draw_seg_mask_annotation(
    image,
    mask_contour: list,
    color=(0, 255, 0),
    alpha=0.4,
    class_name: str = "",
    skip_classes: list[str] | None = None,
):
    if image is None or not mask_contour:
        return
    if skip_classes and class_name in skip_classes:
        return
    pts = np.array(mask_contour, dtype=np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
