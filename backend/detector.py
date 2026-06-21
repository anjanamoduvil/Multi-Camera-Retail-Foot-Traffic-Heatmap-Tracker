from ultralytics import YOLO
import cv2
import time
import logging
import torch

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, model_size='n', conf_threshold=0.5):
        """
        model_size: 'n', 's', 'm', 'l', 'x'
        conf_threshold: confidence threshold for person detection
        """
        self.model_size = model_size
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Load the specified YOLOv8 model (it will auto-download if missing)
        model_name = f'yolov8{model_size}.pt'
        logger.info(f"Loading YOLOv8 model: {model_name} on {self.device}")
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        # We only care about class 0 (person)
        self.target_classes = [0]
        
    def detect(self, frame):
        """
        Run inference on a single frame.
        Returns: 
        - detections: list of dicts with box, conf
        - processed_frame: frame with bounding boxes drawn
        - latency: ms taken for inference
        """
        start_time = time.time()
        
        # Run inference
        results = self.model(frame, classes=self.target_classes, conf=self.conf_threshold, verbose=False, device=self.device)
        
        latency = (time.time() - start_time) * 1000
        
        detections = []
        processed_frame = frame.copy()
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            for box in boxes:
                # Extract coordinates, confidence, and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": conf
                })
                
                # Draw box for visualization
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, f'Person {conf:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
        return detections, processed_frame, latency

    def detect_batch(self, frames):
        """
        Run inference on a list of frames.
        Returns:
        - batch_detections: list of (list of dicts with box, conf)
        - batch_processed_frames: list of frames with bounding boxes drawn
        - latency: ms taken for inference of the entire batch
        """
        if not frames:
            return [], [], 0.0

        start_time = time.time()
        
        # YOLOv8 supports batching implicitly when passed a list
        results = self.model(frames, classes=self.target_classes, conf=self.conf_threshold, verbose=False, device=self.device)
        
        latency = (time.time() - start_time) * 1000
        
        batch_detections = []
        batch_processed_frames = []
        
        for i, result in enumerate(results):
            frame_detections = []
            processed_frame = frames[i].copy()
            
            for box in result.boxes:
                # Extract coordinates, confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                
                frame_detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": conf
                })
                
                # Draw box for visualization
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, f'Person {conf:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            batch_detections.append(frame_detections)
            batch_processed_frames.append(processed_frame)
            
        return batch_detections, batch_processed_frames, latency

    def set_conf_threshold(self, conf):
        self.conf_threshold = conf
        
    def change_model(self, model_size):
        if self.model_size != model_size:
            self.model_size = model_size
            model_name = f'yolov8{model_size}.pt'
            logger.info(f"Changing model to: {model_name}")
            self.model = YOLO(model_name)
