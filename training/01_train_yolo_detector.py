from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path("configs/data.yaml")
BASE_MODEL = "yolov8n.pt"
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 20
SEED = 42


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing dataset configuration: {DATA_YAML}")

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        seed=SEED,
        pretrained=True,
        workers=2,
        project="runs/plantdoc_edge",
        name="yolo_leaf_detector",
        plots=True,
        save=True,
    )
    print("YOLO training completed successfully.")


if __name__ == "__main__":
    main()
