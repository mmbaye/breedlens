"""
predict.py YOLOv11 inference script for sorghum grain counting
Called from R via: python predict.py --image ... --weights ... --output ... [--conf 0.25]
Writes result.json and annotated.jpg into --output directory.
Visualization: clean blue bounding boxes only (no labels, no confidence scores).
"""

import argparse
import json
import sys
from pathlib import Path

def run_inference(image_path, weights_path, output_dir, conf_threshold=0.5, max_det=10000):
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
    except ImportError as e:
        sys.exit(f"Import error: {e}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model   = YOLO(weights_path)
    results = model(image_path, conf=conf_threshold, max_det=max_det,iou=0.5, verbose=False)
    result  = results[0]

    # ── Count detections ──────────────────────────────────────────────────────
    count = len(result.boxes)

    # ── Per-class counts ──────────────────────────────────────────────────────
    class_counts = {}
    if result.boxes is not None and len(result.boxes) > 0:
        names = model.names
        for cls_id in result.boxes.cls.tolist():
            label = names[int(cls_id)]
            class_counts[label] = class_counts.get(label, 0) + 1

    # ── Draw ONLY clean blue rectangles (no labels, no scores) ───────────────
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        # fallback: try via numpy (handles some edge cases)
        import numpy as np
        arr     = np.frombuffer(open(image_path, "rb").read(), np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    BLUE      = (255, 0, 0)   # BGR — vivid blue
    thickness = 5              # box line thickness (pixels)

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)  # [x1, y1, x2, y2]
        for x1, y1, x2, y2 in boxes_xyxy:
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), BLUE, thickness)

    annotated_path = output_dir / "annotated.jpg"
    cv2.imwrite(str(annotated_path), img_bgr)

    # ── Confidence summary ────────────────────────────────────────────────────
    confs = []
    if result.boxes is not None and len(result.boxes) > 0:
        confs = [round(float(c), 4) for c in result.boxes.conf.tolist()]

    result_data = {
        "count":           count,
        "class_counts":    class_counts,
        "confidences":     confs,
        "conf_mean":       round(sum(confs) / len(confs), 4) if confs else 0.0,
        "conf_min":        round(min(confs), 4)              if confs else 0.0,
        "conf_max":        round(max(confs), 4)              if confs else 0.0,
        "annotated_image": str(annotated_path)
    }

    result_json_path = output_dir / "result.json"
    with open(result_json_path, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"OK: {count} detections -> {annotated_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 sorghum grain inference")
    parser.add_argument("--image",   required=True,              help="Path to input image")
    parser.add_argument("--weights", required=True,              help="Path to .pt weights file")
    parser.add_argument("--output",  required=True,              help="Directory for outputs")
    parser.add_argument("--conf",    type=float, default=0.25,   help="Confidence threshold")
    parser.add_argument("--max_det", type=int,   default=10000,  help="Max detections per image")
    args = parser.parse_args()

    run_inference(args.image, args.weights, args.output, args.conf, args.max_det)
