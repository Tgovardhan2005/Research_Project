import cv2
import argparse
from ultralytics import YOLO

def crop_objects(image_path, model_path):
    model = YOLO(model_path)
    results = model(image_path)
    
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Processing detections from: {image_path}")
    
    for r_idx, r in enumerate(results):
        boxes = r.boxes
        names = r.names

        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_object = original_image[y1:y2, x1:x2]
            
            if cropped_object.size == 0:
                continue

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = names[class_id]

            print(f"Displaying cropped object {box_idx + 1} from result {r_idx + 1}: Class='{label}', Confidence={confidence:.2f}")
            
            cv2.imshow(f"Crop: {label}", cropped_object)
            cv2.waitKey(0)
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Plant Cropping")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to the YOLO model")
    args = parser.parse_args()
    
    crop_objects(args.image, args.model)
