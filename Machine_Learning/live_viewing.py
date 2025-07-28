import gradio as gr
import cv2
from ultralytics import YOLO

model = YOLO(r"E:\ALL PYTHON PROJECTS\MachineLearning\results\train5\weights\best.pt")

def get_frame():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=0.1, verbose=False)
        yield results[0].plot()

gr.Interface(fn=get_frame, inputs=[], outputs="image", live=True).launch()
