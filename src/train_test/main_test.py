import torch
from ultralytics import YOLO

from constants import MODEL, DATA, results_root
from functions import ModelEvaluator

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RESULTS_CSV_PATH = f"{results_root}/results.csv"

    MODEL_PATH = "results_data_MIDOG_CMC_CCMCT/runs_train/yolo26/yolo26s-2/weights/epoch20.pt"

    evaluator = ModelEvaluator(MODEL, DATA, device, conf=0.25, iou=0.5)
    evaluator.model = YOLO(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")

    # Evaluate the model
    evaluator.evaluate()

    # Print metrics
    evaluator.print_metrics()

    # Save results to CSV
    evaluator.save_results_to_csv(RESULTS_CSV_PATH)