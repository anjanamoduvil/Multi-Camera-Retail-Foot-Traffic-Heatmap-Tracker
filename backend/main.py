from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import cv2
import base64
import logging
import time
import numpy as np
import datetime
import sqlite3
import time

# --- HOMOGRAPHY CALIBRATION ---
import os
import json

HOMOGRAPHY_MATRICES = {}

if os.path.exists('homography_config.json'):
    with open('homography_config.json', 'r') as f:
        data = json.load(f)
        for cam_id_str, h_list in data.items():
            HOMOGRAPHY_MATRICES[int(cam_id_str)] = np.array(h_list, dtype=np.float32)
else:
    # Map 16 virtual cameras (160x90 local pixels) to 1000x1000 floor plan
    for row in range(4):
        for col in range(4):
            cam_id = row * 4 + col
            H = np.array([
                [250.0/160.0, 0, col * 250.0],
                [0, 250.0/90.0, row * 250.0],
                [0, 0, 1]
            ], dtype=np.float32)
            HOMOGRAPHY_MATRICES[cam_id] = H

def apply_homography(H, local_x, local_y):
    pt = np.dot(H, np.array([local_x, local_y, 1.0]))
    return pt[0]/pt[2], pt[1]/pt[2]

# --- SQLITE TRAJECTORY DB ---
class TrajectoryDB:
    def __init__(self):
        self.conn = sqlite3.connect('trajectories.db', check_same_thread=False)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS trajectories 
                             (global_id TEXT, timestamp REAL, cam_id INT, floor_x REAL, floor_y REAL)''')
    def log(self, global_id, cam_id, fx, fy):
        self.conn.execute("INSERT INTO trajectories VALUES (?, ?, ?, ?, ?)", 
                          (global_id, time.time(), cam_id, fx, fy))

# --- CROSS-CAMERA RE-ID ---
class ReIDModule:
    def __init__(self):
        self.active_tracks = {} # local_id -> global_id
        self.global_positions = {} # global_id -> (time, fx, fy, feature)
        self.next_global_id = 1
        
    def _cosine_similarity(self, a, b):
        if a is None or b is None: return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

    def match(self, local_id, cam_id, fx, fy, feature):
        track_key = f"{cam_id}_{local_id}"
        now = time.time()
        
        if track_key in self.active_tracks:
            gid = self.active_tracks[track_key]
            self.global_positions[gid] = (now, fx, fy, feature)
            return gid
            
        # Re-ID logic: Look for recently disappeared global_ids near this floor location + appearance similarity
        best_match = None
        best_score = 0.0
        
        for gid, (last_time, last_x, last_y, last_feature) in list(self.global_positions.items()):
            dt = now - last_time
            if 0.5 < dt < 15.0: # Disappeared recently
                dist = np.hypot(fx - last_x, fy - last_y)
                if dist < 150.0: # geometrically plausible
                    sim = self._cosine_similarity(feature, last_feature)
                    if sim > 0.8 and sim > best_score:
                        best_score = sim
                        best_match = gid
                    
        if best_match:
            self.active_tracks[track_key] = best_match
            self.global_positions[best_match] = (now, fx, fy, feature)
            return best_match
            
        # Create new global ID
        gid = f"G{self.next_global_id}"
        self.next_global_id += 1
        self.active_tracks[track_key] = gid
        self.global_positions[gid] = (now, fx, fy, feature)
        return gid

from detector import YOLODetector
from ingestion import VideoIngester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLOv8 Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection Manager for broadcasting to multiple clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Error sending message to websocket, removing connection: {e}")
                self.disconnect(connection)

manager = ConnectionManager()

# Global state
detector = YOLODetector(model_size='n', conf_threshold=0.15)
ingesters = []
last_frames = {}
current_metrics = {
    "fps": 0,
    "latency_ms": 0,
    "detections": 0,
    "model": "n"
}

class AnalyticsEngine:
    def __init__(self):
        self.zones = {"Entrance": 0, "Electronics": 0, "Grocery": 0, "Snacks": 0, "Checkout": 0, "Apparel": 0}
        self.trend_history = []
        self.last_trend_update = time.time()
        self.alerts = []
        self.db = TrajectoryDB()
        self.reid = ReIDModule()
        
    def process(self, batch_dets):
        all_dets = []
        for d in batch_dets:
            all_dets.extend(d)
            
        total_count = len(all_dets)
            
        for k in self.zones:
            self.zones[k] = 0
            
        hotspots = []
        
        for det in all_dets:
            x, y, w, h = det["bbox"]
            track_id = det.get("track_id", "Unknown")
            
            # Use bottom-center for foot mapping
            cx = x + w / 2.0
            foot_y = float(y + h)
            
            # Grid layout mapping (16-way split screen is 640x360)
            col = min(int(cx // 160), 3)
            row = min(int(foot_y // 90), 3)
            cam_id = row * 4 + col
            
            local_x = cx - col * 160
            local_y = foot_y - row * 90
            
            # Apply Homography -> Floor Plan Coordinates (0-1000)
            H = HOMOGRAPHY_MATRICES[cam_id]
            floor_x, floor_y = apply_homography(H, local_x, local_y)
            
            # Cross-Camera Re-ID
            feature = det.get("feature", None)
            global_id = self.reid.match(track_id, cam_id, floor_x, floor_y, feature)
            
            # Log Trajectory
            self.db.log(global_id, cam_id, floor_x, floor_y)
            
            # Heatmap Overlay uses 0-10 scale
            normalized_x = (floor_x / 1000) * 10
            normalized_y = (floor_y / 1000) * 10
            
            hotspots.append({
                "x": normalized_x,
                "y": normalized_y,
                "r": 1.0,
                "intensity": 0.85
            })
            
            # True Dwell Zones via Polygon Test (simplified to bounding boxes on floor plan)
            if floor_y < 333:
                if floor_x < 500: self.zones["Checkout"] += 1
                else: self.zones["Entrance"] += 1
            elif floor_y < 666:
                if floor_x < 500: self.zones["Grocery"] += 1
                else: self.zones["Electronics"] += 1
            else:
                if floor_x < 500: self.zones["Apparel"] += 1
                else: self.zones["Snacks"] += 1
                
        # Generate Alerts dynamically
        self.alerts = []
        t_str = datetime.datetime.now().strftime("%H:%M")
        if self.zones["Checkout"] >= 6:
            self.alerts.append({"severity": "Critical", "time": t_str, "zone": "Checkout", "message": "Severe queue bottleneck detected!"})
        elif self.zones["Checkout"] >= 4:
            self.alerts.append({"severity": "Warning", "time": t_str, "zone": "Checkout", "message": "High density near checkout registers."})
            
        if self.zones["Entrance"] >= 5:
            self.alerts.append({"severity": "Warning", "time": t_str, "zone": "Entrance", "message": "Entrance congestion detected."})
            
        # Update Trend History (sample every 2 seconds)
        if time.time() - self.last_trend_update > 2:
            self.trend_history.append({
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "count": total_count
            })
            if len(self.trend_history) > 10:
                self.trend_history.pop(0)
            self.last_trend_update = time.time()
            
        peak_zone = max(self.zones, key=self.zones.get) if total_count > 0 else "None"
        
        return {
            "zones": self.zones,
            "hotspots": hotspots,
            "alerts": self.alerts,
            "trend": self.trend_history,
            "peak_zone": peak_zone,
            "total_visitors": total_count
        }

analytics_engine = AnalyticsEngine()

class ConfigUpdate(BaseModel):
    model_size: str = None
    conf_threshold: float = None
    source: str = None
    is_rtsp: bool = None
    target_fps: int = None

def build_image_grid(images, max_cols=2):
    if not images:
        return None
    target_size = (640, 360)
    resized = [cv2.resize(img, target_size) for img in images]
    num_images = len(resized)
    cols = min(num_images, max_cols)
    rows = (num_images + cols - 1) // cols
    
    empty_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    while len(resized) < rows * cols:
        resized.append(empty_frame)
        
    grid_rows = []
    for i in range(rows):
        row_imgs = resized[i*cols:(i+1)*cols]
        grid_rows.append(cv2.hconcat(row_imgs))
        
    if len(grid_rows) > 1:
        return cv2.vconcat(grid_rows)
    return grid_rows[0]

@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    global ingesters, detector, current_metrics, last_frames
    
    if config.model_size:
        detector.change_model(config.model_size)
        current_metrics["model"] = config.model_size
        
    if config.conf_threshold is not None:
        detector.set_conf_threshold(config.conf_threshold)
        
    if config.source:
        for ingester in ingesters:
            try:
                ingester.stop()
            except Exception as e:
                logger.error(f"Error stopping ingester: {e}")
        ingesters = []
        last_frames = {}
        
        import os
        sources = [s.strip() for s in config.source.split(',')]
        
        for src in sources:
            source_path = src
            if not (source_path.startswith("rtsp://") or source_path.startswith("http://") or source_path.startswith("https://")):
                if not os.path.isabs(source_path):
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    candidate = os.path.join(script_dir, source_path)
                    if os.path.exists(candidate):
                        source_path = candidate
                    else:
                        workspace_dir = os.path.dirname(script_dir)
                        candidate_ws = os.path.join(workspace_dir, source_path)
                        if os.path.exists(candidate_ws):
                            source_path = candidate_ws
            
            ingester = VideoIngester(
                source=source_path, 
                is_rtsp=config.is_rtsp or False,
                target_fps=config.target_fps
            )
            ingester.start()
            ingesters.append(ingester)
        
    return {"status": "success", "message": "Configuration updated"}

@app.get("/api/metrics")
async def get_metrics():
    return current_metrics

async def processing_loop():
    global ingesters, detector, current_metrics, last_frames
    logger.info("Background processing loop started.")
    while True:
        try:
            active_ingesters = [i for i in ingesters if i and i.running]
            if not active_ingesters:
                await asyncio.sleep(0.1)
                continue
                
            if not manager.active_connections:
                await asyncio.sleep(0.01)
                continue
                
            from queue import Empty
            for i, ingester in enumerate(active_ingesters):
                try:
                    idx, frame = ingester.frame_queue.get_nowait()
                    if frame is not None:
                        last_frames[i] = frame
                except Empty:
                    pass
            
            batch_to_process = []
            indices = []
            for i in range(len(active_ingesters)):
                if i in last_frames:
                    batch_to_process.append(last_frames[i])
                    indices.append(i)
                    
            if not batch_to_process:
                await asyncio.sleep(0.01)
                continue
                
            # Run batched detection asynchronously to avoid blocking the WebSocket event loop
            loop = asyncio.get_running_loop()
            batch_dets, batch_processed, latency = await loop.run_in_executor(None, detector.detect_batch, batch_to_process)
            
            display_frames = []
            processed_idx = 0
            total_dets = 0
            for i in range(len(active_ingesters)):
                if i in indices:
                    display_frames.append(batch_processed[processed_idx])
                    total_dets += len(batch_dets[processed_idx])
                    processed_idx += 1
                else:
                    empty_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    display_frames.append(empty_frame)
            
            def encode_frame(frames):
                grid = build_image_grid(frames, max_cols=2)
                if grid is None: return None
                ret, buf = cv2.imencode('.jpg', grid, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ret: return None
                return base64.b64encode(buf).decode('utf-8')
                
            jpg_as_text = await loop.run_in_executor(None, encode_frame, display_frames)
            if not jpg_as_text:
                continue
            
            current_metrics["latency_ms"] = latency
            current_metrics["detections"] = total_dets
            if latency > 0:
                current_metrics["fps"] = int(1000 / latency)
                
            analytics_payload = await loop.run_in_executor(None, analytics_engine.process, batch_dets)
                
            payload = {
                "image": jpg_as_text,
                "image_raw": "",
                "metrics": current_metrics,
                "analytics": analytics_payload
            }
            await manager.broadcast(payload)
            
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            await asyncio.sleep(0.1)
            
        await asyncio.sleep(0.001)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(processing_loop())

@app.websocket("/ws/video")
async def video_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Wait for client to send a message or close
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error for client: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
