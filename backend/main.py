from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import cv2
import base64
import logging
import time
import numpy as np

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
detector = YOLODetector(model_size='n', conf_threshold=0.5)
ingesters = []
last_frames = {}
current_metrics = {
    "fps": 0,
    "latency_ms": 0,
    "detections": 0,
    "model": "n"
}

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
                
            # Run batched detection
            batch_dets, batch_processed, latency = detector.detect_batch(batch_to_process)
            
            display_frames = []
            processed_idx = 0
            total_dets = 0
            for i in range(len(active_ingesters)):
                if i in indices:
                    display_frames.append(batch_processed[processed_idx])
                    total_dets += len(batch_dets[processed_idx])
                    processed_idx += 1
                else:
                    display_frames.append(np.zeros((360, 640, 3), dtype=np.uint8))
            
            grid_img = build_image_grid(display_frames, max_cols=2)
            if grid_img is None:
                continue
                
            ret, buffer = cv2.imencode('.jpg', grid_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                continue
                
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            current_metrics["latency_ms"] = latency
            current_metrics["detections"] = total_dets
            if latency > 0:
                current_metrics["fps"] = int(1000 / latency)
                
            payload = {
                "image": jpg_as_text,
                "metrics": current_metrics
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
