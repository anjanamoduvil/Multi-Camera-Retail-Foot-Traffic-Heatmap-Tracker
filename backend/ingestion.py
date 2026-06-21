import cv2
import time
import threading
from queue import Queue, Empty
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoIngester:
    def __init__(self, source, is_rtsp=False, target_fps=None):
        """
        source: path to video file or RTSP url
        is_rtsp: True if source is a live stream (drops frames to maintain low latency)
        target_fps: If set, will downsample to this FPS
        """
        self.source = source
        self.is_rtsp = is_rtsp
        self.target_fps = target_fps
        self.thread = None
        self.cap = None
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_rtsp else -1
        
        self.frame_queue = Queue(maxsize=30)
        self.running = False
        self.thread = None
        
        # Downsampling logic
        self.frame_skip = 1
        if self.target_fps and self.target_fps < self.fps:
            self.frame_skip = max(1, int(self.fps / self.target_fps))
            
        logger.info(f"Initialized Ingester: {self.width}x{self.height} @ {self.fps}fps. Target FPS: {self.target_fps}, Skip: {self.frame_skip}")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        frame_idx = 0
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if not self.is_rtsp:
                    logger.info("Looping video stream.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.info("End of video stream (failed to loop).")
                        self.running = False
                        break
                else:
                    logger.info("End of video stream.")
                    self.running = False
                    break
                
            # Handle downsampling
            if frame_idx % self.frame_skip == 0:
                # If RTSP, we want to drop old frames if queue is full to prevent latency buildup
                if self.is_rtsp and self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                        
                if not self.frame_queue.full():
                    self.frame_queue.put((frame_idx, frame))
            
            frame_idx += 1
            
            # Simulated real-time playback for local files if needed, but usually 
            # we let the queue backpressure handle it if not RTSP.
            if not self.is_rtsp:
                # Sleep briefly if queue is full to avoid reading entire file into memory instantly
                while self.frame_queue.full() and self.running:
                    time.sleep(0.01)

    def get_frame(self, timeout=1.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None, None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()

    def __del__(self):
        self.stop()
