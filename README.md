# Multi-Camera Retail Foot-Traffic Heatmap Tracker

A high-performance, real-time tracking pipeline designed for retail environments. This project ingests multiple synchronized CCTV camera streams, runs highly optimized batched inference using **YOLOv8**, and streams the live detections via WebSockets to a modern, dynamic frontend dashboard.

---

## 🚀 Key Features

* **Batched Multi-Camera Inference:** Process multiple camera feeds simultaneously in a single GPU pass using YOLOv8. Reduces overhead when scaling up to 4+ cameras.
* **Kalman Filter Tracking:** Integrates **DeepSORT** to maintain stable tracking IDs even when shoppers walk behind shelves or become occluded.
* **Cross-Camera Re-Identification:** Features a custom `ReIDModule` that uses **MobileNet embeddings** (Cosine Similarity) and geometric plausibility to recognize when a shopper leaves Camera A and enters Camera B.
* **Homography Mapping:** Converts 2D pixel coordinates from angled security cameras into accurate physical coordinates on a 1000x1000 top-down floor plan.
* **Spatial Analytics:** Automatically calculates Total Shoppers, Active Zones, and Average Dwell Times using polygon intersections.
* **Real-time WebSocket Streaming:** High-efficiency JPEG-encoded frames and telemetry data (Latency, FPS, Live Count) broadcast to the web UI.
* **Cyber-Physical Dashboard:** A premium, low-latency Vanilla JavaScript dashboard powered by WebSockets to render live video, charts, and glowing heatmaps at 30 FPS.

---

## 🏗️ System Architecture

### 1. Ingestion Layer (`ingestion.py`)
Responsible for reading RTSP streams, MP4 files, or webcam feeds. Frames are fetched asynchronously and placed into a non-blocking queue to prevent the AI engine from lagging behind the live video.

### 2. Neural Engine (`detector.py`)
Uses Ultralytics **YOLOv8** for person detection. The system can process frames using a Tiled approach (`_detect_tiled`) to catch small objects in high-resolution multi-camera grids. **DeepSORT** handles temporal tracking and extracts a 128-D visual feature embedding using MobileNet.

### 3. Spatial & Analytics Engine (`main.py`)
* **Homography Calibration:** Pre-configured matrices map 2D camera coordinates to 3D floor plan coordinates. 
* **ReIDModule:** Compares MobileNet embeddings across different cameras. If a person drops off Camera 1 and appears on Camera 2, the system calculates cosine similarity and geometrical plausibility to merge their trajectories under a single Global ID.
* **TrajectoryDB:** Logs all movements in a local SQLite database (`trajectories.db`) for historical analysis.

### 4. WebSocket Server & API
FastAPI serves the HTTP endpoints for configuration updates (`/api/config`) and maintains a bidirectional WebSocket connection (`/ws/video`) that streams base64 encoded JPEGs and JSON payload containing active zones, total visitors, trends, and alerts.

---

## 🛠️ Technology Stack

**Backend**
* Python 3.x
* FastAPI & Uvicorn
* PyTorch & Ultralytics (YOLOv8)
* OpenCV (Computer Vision & Grid Stitching)
* deep-sort-realtime
* SQLite3

**Frontend**
* HTML5 / CSS3 (Vanilla, Glassmorphism UI)
* JavaScript (WebSocket Client)
* Vite (Bundler)
* Chart.js

---

## 📁 Repository Structure

```
Multi-Camera-Retail-Foot-Traffic-Heatmap-Tracker/
│
├── backend/
│   ├── main.py                # FastAPI server, WebSockets, and ReIDModule
│   ├── detector.py            # YOLOv8 + DeepSORT inference engine
│   ├── ingestion.py           # Asynchronous multi-camera stream processing
│   ├── calibration_tool.py    # GUI for generating Homography matrices
│   ├── train.py               # Scripts for YOLOv8 fine-tuning
│   └── requirements.txt       # Python dependencies
│
├── frontend/
│   ├── index.html             # The Cyber-Physical Dashboard UI
│   ├── style.css              # Custom styling with Glassmorphism
│   ├── main.js                # WebSocket client and UI logic
│   ├── floor_plan.png         # Background image for the heatmap
│   └── package.json           # Vite dev server configuration
│
└── README.md                  # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.9+
* Node.js v16+
* (Optional but recommended) NVIDIA GPU with CUDA for real-time inference.

### 1. Clone the repository
```bash
git clone https://github.com/anjanamoduvil/Multi-Camera-Retail-Foot-Traffic-Heatmap-Tracker.git
cd Multi-Camera-Retail-Foot-Traffic-Heatmap-Tracker
```

### 2. Install Backend Dependencies
It is recommended to use a virtual environment:
```bash
cd backend
python -m venv venv

# Activate venv (Windows)
venv\Scripts\activate
# Activate venv (macOS/Linux)
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Install Frontend Dependencies
```bash
cd ../frontend
npm install
```

---

## 🚀 Running the System

### 1. Start the AI Backend Engine
Open a terminal, ensure your virtual environment is active, and run:
```bash
cd backend
python main.py
```
*The backend API and WebSocket server will run on `http://0.0.0.0:8000`.*

### 2. Start the Frontend Dashboard
Open a new terminal window:
```bash
cd frontend
npm run dev
```
*The frontend will be available at `http://localhost:5173`.*

---

## 💻 Usage & Utilities

### Launching the Pipeline
1. Open the dashboard in your browser (`http://localhost:5173`).
2. Enter the path to your video source(s) in the **Data Source** input field.
   * *For multi-camera batched inference, pass a comma-separated list of paths (e.g., `cam1.mp4, cam2.mp4` or RTSP links).*
3. Adjust the **Neural Engine** parameters (Model size `n/s/m/l/x`, Confidence threshold, Target FPS).
4. Click **Initialize Stream** to launch the pipeline! The dashboard will start receiving processed frames and heatmap data.

### Utility Scripts (Backend)
This repository includes several utilities in the `backend/` directory for advanced configuration and testing:

* **`calibration_tool.py`**: A GUI tool to generate Homography matrices. Maps 2D camera pixel coordinates to a top-down floor plan by selecting 4 corresponding points on both images.
  ```bash
  python calibration_tool.py --camera_image snap.jpg --floor_plan floor.png --cam_id 0
  ```
  *This updates `homography_config.json`, which the main server automatically reads on startup.*

* **`extract_frames.py`**: Extracts frames from CCTV footage at specified intervals, useful for creating custom datasets for YOLO fine-tuning.
* **`train.py`**: A script to fine-tune the YOLOv8 model on custom datasets.
* **`benchmark.py`**: Runs performance benchmarks on inference speeds.
* **`multi_camera_demo.py`**: CLI script to test multi-camera batch processing and inference independently from the web dashboard.

---

## 🛣️ Project Roadmap
This repository covers Detection and Tracking. Future milestones will introduce:
1. **Improved Re-ID Models:** Swapping MobileNet for OSNet for better cross-camera recognition.
2. **Zone Configurator GUI:** A web-based tool to draw polygons on the floor plan for dynamic zone assignment instead of hardcoded coordinates.
3. **Advanced Analytics:** Generate daily/weekly PDF reports based on the SQLite TrajectoryDB data.
