import cv2
import os

source_path = r"c:\Users\VICTUS\multi-camera retail foot-traffic heatmap tracker\istockphoto-1267814575-640_adpp_is.mp4"
print(f"Opening {source_path}...")
cap = cv2.VideoCapture(source_path)

if not cap.isOpened():
    print("Error opening video")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cw, ch = w // 2, h // 2

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Output sizes: 640x360 to keep them decent resolution for YOLO
out_w, out_h = 640, 360
out1 = cv2.VideoWriter('cam1.mp4', fourcc, fps, (out_w, out_h))
out2 = cv2.VideoWriter('cam2.mp4', fourcc, fps, (out_w, out_h))
out3 = cv2.VideoWriter('cam3.mp4', fourcc, fps, (out_w, out_h))
out4 = cv2.VideoWriter('cam4.mp4', fourcc, fps, (out_w, out_h))

max_frames = 600
frame_count = 0

print(f"Processing video of size {w}x{h} into 4 cameras...")

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret: break
    
    # Strictly cut the frame into 4 exact quadrants
    crop1 = cv2.resize(frame[0:ch, 0:cw], (out_w, out_h))
    crop2 = cv2.resize(frame[0:ch, cw:w], (out_w, out_h))
    crop3 = cv2.resize(frame[ch:h, 0:cw], (out_w, out_h))
    crop4 = cv2.resize(frame[ch:h, cw:w], (out_w, out_h))
    
    out1.write(crop1)
    out2.write(crop2)
    out3.write(crop3)
    out4.write(crop4)
    
    frame_count += 1

cap.release()
out1.release()
out2.release()
out3.release()
out4.release()
print("Successfully generated 4 cameras!")
