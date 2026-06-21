import cv2
import numpy as np
import json
import os
import argparse

def run_calibration(camera_image_path, floor_plan_path, output_json="homography_config.json", cam_id=0):
    cam_img = cv2.imread(camera_image_path)
    floor_img = cv2.imread(floor_plan_path)
    
    if cam_img is None:
        print(f"Error loading camera image: {camera_image_path}")
        return
    if floor_img is None:
        print(f"Error loading floor plan: {floor_plan_path}")
        return

    cam_pts = []
    floor_pts = []

    def get_cam_pts(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(cam_pts) < 4:
                cam_pts.append([x, y])
                cv2.circle(cam_img, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(cam_img, str(len(cam_pts)), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                cv2.imshow("Camera View - Select 4 Points", cam_img)

    def get_floor_pts(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(floor_pts) < 4:
                floor_pts.append([x, y])
                cv2.circle(floor_img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(floor_img, str(len(floor_pts)), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.imshow("Floor Plan - Select 4 Corresponding Points", floor_img)

    print("Step 1: Click 4 reference points on the Camera View in order (e.g. top-left, top-right, bottom-right, bottom-left).")
    cv2.imshow("Camera View - Select 4 Points", cam_img)
    cv2.setMouseCallback("Camera View - Select 4 Points", get_cam_pts)
    
    while True:
        cv2.waitKey(10)
        if len(cam_pts) == 4:
            break
            
    print("Step 2: Click the SAME 4 reference points on the Floor Plan in the EXACT SAME order.")
    cv2.imshow("Floor Plan - Select 4 Corresponding Points", floor_img)
    cv2.setMouseCallback("Floor Plan - Select 4 Corresponding Points", get_floor_pts)
    
    while True:
        cv2.waitKey(10)
        if len(floor_pts) == 4:
            break
            
    cv2.destroyAllWindows()
    
    src_pts = np.array(cam_pts, dtype=np.float32)
    dst_pts = np.array(floor_pts, dtype=np.float32)
    
    H, status = cv2.findHomography(src_pts, dst_pts)
    
    print("\nHomography Matrix Calculated:\n", H)
    
    data = {}
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            try:
                data = json.load(f)
            except:
                pass
                
    data[str(cam_id)] = H.tolist()
    
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"\nSaved matrix for Camera {cam_id} to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Homography Matrix for a camera.")
    parser.add_argument("--camera_image", type=str, required=True, help="Path to a snapshot from the camera.")
    parser.add_argument("--floor_plan", type=str, required=True, help="Path to the floor plan image.")
    parser.add_argument("--cam_id", type=int, default=0, help="Camera ID to save in the JSON config.")
    parser.add_argument("--output", type=str, default="homography_config.json", help="Output JSON file path.")
    
    args = parser.parse_args()
    run_calibration(args.camera_image, args.floor_plan, args.output, args.cam_id)
