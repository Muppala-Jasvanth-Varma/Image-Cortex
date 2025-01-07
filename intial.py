import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import numpy as np

COCO_LABELS = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def query_match(predictions, query):
    matched_objects = []
    for label, score in zip(predictions['labels'], predictions['scores']):
        if label.item() < len(COCO_LABELS):
            label_name = COCO_LABELS[label.item()]
            if query.lower() in label_name.lower() and score > 0.5:
                matched_objects.append((label_name, score.item()))
    return matched_objects

def display_results(image_path, predictions, matched_objects, threshold=0.5):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    canvas = 255 * np.ones((height, width + 400, 3), dtype=np.uint8)
    canvas[:, :width, :] = image

    detected_objects = []
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box.tolist())
            label_name = COCO_LABELS[label.item()] if label.item() < len(COCO_LABELS) else "Unknown"
            detected_objects.append((label_name, score.item()))
            
            color = (102, 204, 255) if (label_name, score.item()) in matched_objects else (204, 229, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"{label_name} ({score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    y_offset = 20
    cv2.putText(canvas, "Detected Items:", (width + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    y_offset += 40
    for obj in detected_objects:
        cv2.putText(canvas, f"- {obj[0]} ({obj[1]:.2f})", (width + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        y_offset += 20

    if matched_objects:
        print("\nMatched Objects:")
        for obj in matched_objects:
            print(f"- {obj[0]} (confidence: {obj[1]:.2f})")
    else:
        print("\nNo matched objects found.")

    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def detect_objects_in_image(image_path, query):
    input_image = preprocess_image(image_path)
    if input_image is None:
        return
    with torch.no_grad():
        predictions = model(input_image)[0]
    matched_objects = query_match(predictions, query)
    display_results(image_path, predictions, matched_objects)

if __name__ == "__main__":
    image_path = r"W:\Computer vision\proj\input.jpg"
    query = "cat"
    detect_objects_in_image(image_path, query)
