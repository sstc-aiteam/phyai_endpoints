import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "ward_item_seg.pt"


def predict_and_draw(image_path: str, output_path: str = None):
    model = YOLO(MODEL_PATH)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    results = model(img, verbose=False)[0]
    overlay = img.copy()

    for i, (box, mask) in enumerate(zip(results.boxes, results.masks.xy)):
        color = tuple(int(c) for c in np.random.randint(50, 230, 3))

        # Draw filled segmentation mask
        pts = mask.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], color)

        # Draw bbox
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{results.names[cls_id]} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Blend mask overlay with original
    result_img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    if output_path is None:
        p = Path(image_path)
        output_path = str(p.with_stem(p.stem + "_seg"))

    cv2.imwrite(output_path, result_img)
    print(f"Saved result to: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python yolo_seg_predict.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    predict_and_draw(image_path, output_path)
