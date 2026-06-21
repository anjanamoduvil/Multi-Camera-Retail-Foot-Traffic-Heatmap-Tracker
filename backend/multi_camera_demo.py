import time
import cv2
import logging
from detector import YOLODetector
from ingestion import VideoIngester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_batch_inference():
    logger.info("Initializing multi-camera batch inference demo...")
    
    # We will use the same sample videos, but simulate multiple cameras
    sources = [
        'sample.mp4',
        'store.mp4'
    ]
    
    # Initialize ingesters
    ingesters = []
    for src in sources:
        ingester = VideoIngester(src, is_rtsp=False, target_fps=15)
        ingester.start()
        ingesters.append(ingester)
        
    # Initialize single detector for all streams
    detector = YOLODetector(model_size='n', conf_threshold=0.3)
    
    logger.info(f"Started {len(ingesters)} camera streams. Press 'q' in video windows to exit.")
    
    # Allow some frames to buffer
    time.sleep(1)
    
    try:
        while True:
            batch_frames = []
            active_ingesters = []
            
            # Gather exactly 1 frame from each active stream to form a batch
            for i, ingester in enumerate(ingesters):
                if ingester.running:
                    idx, frame = ingester.get_frame(timeout=0.1)
                    if frame is not None:
                        batch_frames.append(frame)
                        active_ingesters.append(i)
            
            if not batch_frames:
                # All streams ended or empty
                break
                
            # Perform batched inference
            # This pushes all frames through the GPU in a single forward pass
            batch_dets, batch_processed, latency = detector.detect_batch(batch_frames)
            
            # Display results
            for idx, processed_frame in zip(active_ingesters, batch_processed):
                # Resize for display so they fit on screen easily
                disp_frame = cv2.resize(processed_frame, (640, 360))
                cv2.imshow(f"Camera {idx} - {sources[idx]}", disp_frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        for ingester in ingesters:
            ingester.stop()
        cv2.destroyAllWindows()
        logger.info("Demo complete.")

if __name__ == "__main__":
    demo_batch_inference()
