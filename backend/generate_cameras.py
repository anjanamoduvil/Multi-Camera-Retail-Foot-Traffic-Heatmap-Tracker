import cv2

source_path = r"c:\Users\VICTUS\multi-camera retail foot-traffic heatmap tracker\store_cctv.mp4"
print(f"Opening {source_path}...")
cap = cv2.VideoCapture(source_path)

if not cap.isOpened():
    print("Error opening video")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cw, ch = 640, 360

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out1 = cv2.VideoWriter('cam1.mp4', fourcc, fps, (cw, ch))
out2 = cv2.VideoWriter('cam2.mp4', fourcc, fps, (cw, ch))
out3 = cv2.VideoWriter('cam3.mp4', fourcc, fps, (cw, ch))
out4 = cv2.VideoWriter('cam4.mp4', fourcc, fps, (cw, ch))

max_frames = 600
frame_count = 0

print("Generating 4 strictly synchronized, non-overlapping distinct camera angles...")

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret: break
    
    # Extract 4 completely distinct, non-overlapping regions
    # This ensures they look like different cameras, but remain perfectly synchronized in time
    
    # Cam 1: Top-Left (Checkout zone)
    crop1 = cv2.resize(frame[0:350, 0:600], (cw, ch))
    
    # Cam 2: Top-Right (Aisle zone)
    crop2 = cv2.resize(frame[0:350, 670:1270], (cw, ch))
    
    # Cam 3: Bottom-Left (Front shelves zone)
    crop3 = cv2.resize(frame[370:720, 0:600], (cw, ch))
    
    # Cam 4: Bottom-Right (Entrance mat zone)
    crop4 = cv2.resize(frame[370:720, 670:1270], (cw, ch))
    
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
print("Successfully generated 4 synchronized distinct cameras!")
