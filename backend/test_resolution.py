import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(r"c:\Users\VICTUS\multi-camera retail foot-traffic heatmap tracker\istockphoto-1267814575-640_adpp_is.mp4")

ret, frame = cap.read()
if not ret: exit(1)

for size in [640, 1280, 2560, 3840]:
    results = model(frame, imgsz=size, classes=[0], verbose=False)
    boxes = results[0].boxes
    print(f"imgsz={size}: Detected {len(boxes)} people")

cap.release()
