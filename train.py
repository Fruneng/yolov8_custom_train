import os

from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='pothole_v8.yaml', epochs=100, imgsz=640, name='yolov8n_custom') 