from ultralytics import YOLO
import cv2
import time
import logging
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

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
        
        # Initialize DeepSORT Tracker
        # We use the MobileNet visual feature embedder to support cross-camera Re-ID.
        self.tracker = DeepSort(max_age=15, n_init=3, nms_max_overlap=0.5, max_cosine_distance=0.2, nn_budget=5, embedder='mobilenet')
        
    def _detect_tiled(self, frame, grid_x=3, grid_y=3):
        """
        Slice the frame into a grid, run batch inference on tiles to detect tiny objects without OOM.
        """
        h, w = frame.shape[:2]
        tile_h = h // grid_y
        tile_w = w // grid_x
        
        tiles = []
        offsets = []
        
        for row in range(grid_y):
            for col in range(grid_x):
                y1 = row * tile_h
                y2 = (row + 1) * tile_h if row < grid_y - 1 else h
                x1 = col * tile_w
                x2 = (col + 1) * tile_w if col < grid_x - 1 else w
                
                tiles.append(frame[y1:y2, x1:x2])
                offsets.append((x1, y1))
                
        # Run inference on all tiles in one batch pass (safe memory footprint)
        results = self.model(tiles, classes=self.target_classes, conf=self.conf_threshold, verbose=False, device=self.device, imgsz=640)
        
        all_detections = []
        for i, result in enumerate(results):
            offset_x, offset_y = offsets[i]
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                
                # Shift coordinates back to original frame
                # deep_sort_realtime expects ([left, top, w, h], confidence, detection_class)
                w_box = x2 - x1
                h_box = y2 - y1
                all_detections.append(
                    ([int(x1 + offset_x), int(y1 + offset_y), int(w_box), int(h_box)], conf, "person")
                )
        return all_detections
        
    def detect(self, frame):
        """
        Run inference on a single frame using tiling to detect small objects.
        """
        start_time = time.time()
        
        # Use 3x3 tiling for dense CCTV multi-camera views
        detections = self._detect_tiled(frame, grid_x=3, grid_y=3)
        
        latency = (time.time() - start_time) * 1000
        processed_frame = frame.copy()
        
        # Pass detections to DeepSORT and let it extract visual embeddings
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        tracked_detections = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            conf = track.get_det_conf()
            if conf is None:
                conf = 0.5
            
            feature = track.features[-1] if track.features else None
            tracked_detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "track_id": track_id,
                "feature": feature
            })
            
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(processed_frame, f'ID:{track_id}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
        return tracked_detections, processed_frame, latency

    def detect_batch(self, frames):
        """
        Run inference on a list of frames.
        """
        start_time = time.time()
        batch_detections = []
        batch_processed_frames = []
        
        for frame in frames:
            # Use imgsz=1280 to preserve detail for small people in dense multi-camera splits
            results = self.model(frame, verbose=False, imgsz=1280, classes=self.target_classes, conf=self.conf_threshold)
            detections = []
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    w, h = x2 - x1, y2 - y1
                    
                    if w > 0 and h > 0:
                        detections.append(([int(x1), int(y1), int(w), int(h)], float(box.conf[0]), 'person'))
            
            # Pass through DeepSORT to extract embeddings for Re-ID
            try:
                tracks = self.tracker.update_tracks(detections, frame=frame)
            except Exception as e:
                print(f"Tracker math error ({e}), resetting DeepSORT...")
                self.tracker = DeepSort(max_age=15, n_init=3, nms_max_overlap=0.5, max_cosine_distance=0.2, nn_budget=5, embedder='mobilenet')
                tracks = []
            
            tracked_detections = []
            processed_frame = frame.copy()
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                
                # Clip to frame boundaries to prevent giant boxes
                x1, y1 = max(0, ltrb[0]), max(0, ltrb[1])
                x2, y2 = min(frame.shape[1], ltrb[2]), min(frame.shape[0], ltrb[3])
                w, h = x2 - x1, y2 - y1
                
                # Ignore Kalman filter explosions (boxes larger than 85% width, 95% height)
                if w > frame.shape[1] * 0.85 or h > frame.shape[0] * 0.95 or w <= 0 or h <= 0:
                    continue
                    
                bbox = [x1, y1, w, h]
                
                # Draw bounding box
                cv2.rectangle(processed_frame, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 255), 2)
                cv2.putText(processed_frame, f"ID:{track_id}", (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            
                feature = track.features[-1] if track.features else None
                tracked_detections.append({
                    "bbox": bbox,
                    "conf": 1.0,
                    "track_id": track_id,
                    "feature": feature
                })
            batch_detections.append(tracked_detections)
            batch_processed_frames.append(processed_frame)
            
        latency = (time.time() - start_time) * 1000
        return batch_detections, batch_processed_frames, latency

    def set_conf_threshold(self, conf):
        self.conf_threshold = conf
        
    def change_model(self, model_size):
        if self.model_size != model_size:
            self.model_size = model_size
            model_name = f'yolov8{model_size}.pt'
            logger.info(f"Changing model to: {model_name}")
            self.model = YOLO(model_name)
