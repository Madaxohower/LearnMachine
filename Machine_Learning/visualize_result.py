import cv2
from ultralytics import YOLO

# === Path to your trained model ===
model_path = r"E:\ALL PYTHON PROJECTS\MachineLearning\results\train2\weights\best.pt"

# === Path to your test image ===
image_path = r"E:\ALL PYTHON PROJECTS\MachineLearning\dataset\images\test\IMG_0864.JPG"

# === Load model ===
model = YOLO(model_path)

# === Predict ===
results = model.predict(source=image_path, conf=0.9)

# === Draw and display ===
for r in results:
    annotated = r.plot()
    cv2.imshow("YOLO Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
