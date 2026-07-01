import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO inference on a single image and save the annotated result.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("model", nargs="?", default="runs_train/yolo26/yolo26s/weights/best.pt", help="Path to the trained YOLO model weights (.pt)")
    parser.add_argument("output_dir", nargs="?", default="outputs/inference", help="Directory where the annotated image will be saved")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size")
    return parser.parse_args()


def draw_boxes(image, result, names):
    annotated = image.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return annotated

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, cls_idx in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{names.get(cls_idx, cls_idx)} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated


def main():
    args = parse_args()

    image_path = Path(args.image).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = model(str(image_path), conf=args.conf, imgsz=args.img_size, stream=False)
    result = results[0]

    annotated = draw_boxes(image, result, model.names)

    output_path = output_dir / image_path.name
    success = cv2.imwrite(str(output_path), annotated)
    if not success:
        raise RuntimeError(f"Failed to save output image to: {output_path}")

    print(f"Annotated image saved to: {output_path}")


if __name__ == "__main__":
    main()
