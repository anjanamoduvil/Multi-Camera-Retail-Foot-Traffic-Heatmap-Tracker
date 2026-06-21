# Multi-Camera Retail Foot-Traffic Heatmap Tracker

A high-performance, real-time tracking pipeline designed for retail environments. This project ingests multiple synchronized CCTV camera streams, runs highly optimized batched inference using **YOLOv8**, and streams the live detections via WebSockets to a modern, dynamic frontend dashboard.

## 🚀 Features

* **Batched Multi-Camera Inference:** Process multiple camera feeds simultaneously in a single GPU pass using YOLOv8.
* **Dynamic Grid Stitching:** Automatically resizes and stitches processed frames into a split-screen composite view.
* **Real-time WebSocket Streaming:** High-efficiency JPEG-encoded frames and telemetry data (Latency, FPS, Live Count) broadcast to the web UI.
* **Modern Web Dashboard:** A sleek, glassmorphic UI built with Vite and vanilla JavaScript for controlling the neural engine parameters in real-time.

## 🛠️ Technology Stack

**Backend**
* Python 3.x
* FastAPI & Uvicorn (WebSocket Server)
* PyTorch & Ultralytics (YOLOv8)
* OpenCV (Computer Vision & Grid Stitching)

**Frontend**
* HTML5 / CSS3 (Vanilla, Glassmorphism UI)
* JavaScript (WebSocket Client)
* Vite (Bundler)
* **Multi-Camera Detection:** Asynchronous, batched inference using **YOLOv8** to detect humans across multiple video streams without stuttering.
* **Kalman Filter Tracking:** Integrates **DeepSORT** to maintain stable tracking IDs even when shoppers walk behind shelves or become occluded.
* **Cross-Camera Re-Identification:** Features a custom `ReIDModule` that uses **MobileNet embeddings** (Cosine Similarity) and geometric plausibility to recognize when a shopper leaves Camera A and enters Camera B.
* **Homography Mapping:** Converts 2D pixel coordinates from angled security cameras into accurate physical coordinates on a top-down floor plan.
* **Spatial Analytics:** Automatically calculates Total Shoppers, Active Zones, and Average Dwell Times using polygon intersections.
* **Cyber-Physical Dashboard:** A premium, low-latency Vanilla JavaScript dashboard powered by WebSockets to render live video, charts, and glowing heatmaps at 30 FPS.

## 🛠 Tech Stack

* **Backend / AI:** Python, PyTorch, Ultralytics YOLOv8, deep-sort-realtime, OpenCV, NumPy
* **API & Data:** FastAPI, WebSockets, SQLite3 (Trajectory DB)
* **Frontend:** HTML5, CSS3 (Glassmorphism), Vanilla JavaScript, Chart.js

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
│   ├── floor_plan.png         # Background image for the heatmap
│   └── package.json           # Vite dev server configuration
│
└── README.md                  # Project documentation
```

## ⚙️ Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/anjanamoduvil/Multi-Camera-Retail-Foot-Traffic-Heatmap-Tracker.git
cd Multi-Camera-Retail-Foot-Traffic-Heatmap-Tracker
```

2. **Install Backend Dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

3. **Install Frontend Dependencies:**
```bash
cd ../frontend
npm install
```

## 🚀 Running the System

1. **Start the AI Backend Engine:**
```bash
cd backend
python main.py
```
*The backend will run on `http://0.0.0.0:8000`.*

### 3. Frontend Setup
Open a new terminal, navigate to the `frontend` directory, install Node dependencies, and start the Vite dev server:
```bash
cd frontend
npm install
npm run dev
```
*The frontend will be available at `http://localhost:5173`.*

## 💻 Usage
1. Open the dashboard in your browser (`http://localhost:5173`).
2. Enter the path to your video source(s) in the **Data Source** input field.
   * *For multi-camera batched inference, pass a comma-separated list of paths (e.g., `cam1.mp4, cam2.mp4`).*
3. Adjust the **Neural Engine** parameters (Model size, Confidence threshold, Target FPS).
4. Click **Initialize Stream** to launch the pipeline!

## 🛣️ Project Roadmap
This repository currently covers **Phase 1: Detection**. Future milestones will introduce:
1. **DeepSORT Tracking:** Assigning persistent IDs to shoppers.
2. **Cross-Camera Re-ID:** Maintaining consistent shopper identities across different camera views.
3. **Homography Floor Mapping:** Projecting 2D bounding boxes onto a 3D floor plan.
4. **Heatmap & Analytics Engine:** Generating actionable retail intelligence.
