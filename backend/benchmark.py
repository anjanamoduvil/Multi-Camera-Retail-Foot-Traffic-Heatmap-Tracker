import time
import cv2
import logging
import argparse
from detector import YOLODetector
from ingestion import VideoIngester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark(video_source, models=['n', 's', 'm'], max_frames=100):
    logger.info(f"Starting benchmark on {video_source}")
    results = {}
    
    for model_size in models:
        logger.info(f"\n--- Benchmarking YOLOv8{model_size} ---")
        detector = YOLODetector(model_size=model_size)
        ingester = VideoIngester(video_source, is_rtsp=False)
        ingester.start()
        
        frame_count = 0
        total_latency = 0
        total_detections = 0
        
        start_time = time.time()
        
        while frame_count < max_frames:
            frame_idx, frame = ingester.get_frame(timeout=2.0)
            if frame is None:
                break
                
            dets, _, latency = detector.detect(frame)
            total_latency += latency
            total_detections += len(dets)
            frame_count += 1
            
            if frame_count % 10 == 0:
                logger.info(f"Processed {frame_count}/{max_frames} frames...")
                
        ingester.stop()
        
        end_time = time.time()
        wall_time = end_time - start_time
        avg_latency = total_latency / frame_count if frame_count > 0 else 0
        fps = frame_count / wall_time if wall_time > 0 else 0
        
        results[model_size] = {
            'fps': fps,
            'avg_latency_ms': avg_latency,
            'total_detections': total_detections,
            'frames_processed': frame_count
        }
        
        logger.info(f"Results for {model_size}: {fps:.2f} FPS, {avg_latency:.2f}ms/frame, Detections: {total_detections}")
        
    logger.info("\n=== Benchmark Summary ===")
    for m, res in results.items():
        logger.info(f"YOLOv8{m}: {res['fps']:.1f} FPS | Latency: {res['avg_latency_ms']:.1f} ms | Total Detections: {res['total_detections']}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 models on a video.")
    parser.add_argument('--source', type=str, required=True, help="Path to video file or RTSP stream")
    parser.add_argument('--frames', type=int, default=100, help="Number of frames to benchmark")
    args = parser.parse_args()
    
    benchmark(args.source, max_frames=args.frames)
