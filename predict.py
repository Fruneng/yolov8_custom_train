from ultralytics import YOLO
from PIL import Image

# Load a model
model = YOLO("/Users/chucheng/code/ansa/runs/detect/yolov8n_custom8/weights/best.pt")  # load a pretrained model (recommended for training)

im1 = Image.open("1.png")
results = model.predict(source=im1, save=True)

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    print(boxes.cls)
    print(boxes.conf)
