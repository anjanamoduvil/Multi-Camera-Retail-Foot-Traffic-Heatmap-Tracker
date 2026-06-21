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

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/anjanamoduvil/Multi-Camera-Retail-Foot-Traffic-Heatmap-Tracker.git
cd Multi-Camera-Retail-Foot-Traffic-Heatmap-Tracker
```

### 2. Backend Setup
Navigate to the `backend` directory, install the required dependencies, and start the FastAPI server:
```bash
cd backend
pip install -r requirements.txt
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
