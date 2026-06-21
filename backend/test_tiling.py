import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(r"c:\Users\VICTUS\multi-camera retail foot-traffic heatmap tracker\istockphoto-1267814575-640_adpp_is.mp4")

ret, frame = cap.read()
if not ret: exit(1)

# Tile it into 4x4
h, w = frame.shape[:2]
tile_h = h // 4
tile_w = w // 4

tiles = []
offsets = []

for row in range(4):
    for col in range(4):
        y1 = row * tile_h
        y2 = (row + 1) * tile_h
        x1 = col * tile_w
        x2 = (col + 1) * tile_w
        
        tile = frame[y1:y2, x1:x2]
        tiles.append(tile)
        offsets.append((x1, y1))

# Run batch inference
results = model(tiles, classes=[0], conf=0.25, verbose=False)

total_people = 0
for i, result in enumerate(results):
    total_people += len(result.boxes)
    
print(f"Tiling method detected: {total_people} people")

cap.release()
