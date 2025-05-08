# 🧠 Image Cortex

**Image Cortex** is a query-based object detection system built using PyTorch's pre-trained Faster R-CNN model on the COCO dataset. This project allows users to detect and highlight specific objects in an image by providing a custom text-based query (e.g., "cat", "dog", "car").

---

## 🚀 Features

- 🔍 Detects and labels objects using **Faster R-CNN**.
- 🧠 Filters detections based on a **text query** (e.g., “car”, “bottle”).
- 🎯 Highlights only the matching objects with bounding boxes and labels.
- 🖼️ Visual output with matplotlib and OpenCV.
- 📋 Displays matched objects and their confidence scores in the console.

---

## 📂 Project Structure

image-cortex/
│
├── image_cortex.py # Main object detection script
├── input.jpg # Sample input image
├── README.md # Documentation
└── requirements.txt # Python dependencies (optional)


---

## 🛠️ Installation

Install the required Python packages:

pip install torch torchvision matplotlib opencv-python pillow
torch
torchvision
matplotlib
opencv-python
Pillow

And install with:
pip install -r requirements.txt


📷 Usage
1. Add Your Image
Replace or rename your image as input.jpg, or modify the image path in the script:

image_path = r"W:\Computer vision\proj\input.jpg"  # Change as needed

2. Set Your Query
Set the object you want to search for in the image:

query = "cat"  # Change to any COCO object like "dog", "car", etc.

3. Run the Script

python image_cortex.py
🧪 Sample Output
✅ Console Output:

Matched Objects:
- cat (confidence: 0.89)
- cat (confidence: 0.76)

🖼️ Image Output:
Bounding boxes are drawn for all detected objects. Query-matched objects are highlighted more distinctly.

🎯 COCO Labels (Examples)
Some of the 91 COCO labels you can query include:

person, car, bicycle, dog, cat, chair, laptop, bottle, cup, book, etc.

See the full list in the COCO_LABELS array inside the script.

🔮 Future Improvements
 Real-time webcam support.

 Save results as JSON/CSV.

 Save annotated image output.

 Add GUI using Gradio or Streamlit.

 Use advanced models like YOLOv8 or DETR.

 🤝 Contributing
Contributions are welcome! Feel free to fork this repo, create a new branch, and submit a pull request with improvements or new features.

👨‍💻 Author
Jasvanth Varma Muppala
Engineer | ML Enthusiast.
