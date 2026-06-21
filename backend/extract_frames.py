import cv2
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_frames(video_path, output_dir, interval_seconds=2.0):
    """
    Extracts frames from a video file at a given interval.
    
    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory where extracted frames will be saved.
        interval_seconds (float): Interval in seconds between extracted frames.
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: # Handle NaN or 0
        fps = 30.0
        
    frame_interval = max(1, int(fps * interval_seconds))
    
    frame_count = 0
    saved_count = 0
    
    logger.info(f"Extracting frames from {video_path} into {output_dir}")
    logger.info(f"Video FPS: {fps:.2f}, Extraction Interval: every {frame_interval} frames ({interval_seconds}s)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"{video_name}_frame_{saved_count:05d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    logger.info(f"Finished extracting {saved_count} frames from {video_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from CCTV footage for labeling")
    parser.add_argument('--video', type=str, required=True, help="Path to input video")
    parser.add_argument('--output', type=str, default='./dataset/images/raw', help="Output directory")
    parser.add_argument('--interval', type=float, default=2.0, help="Seconds between extracted frames")
    
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.interval)
