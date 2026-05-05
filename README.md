# TruthLens: Advanced Deepfake Detection System

TruthLens is a full-stack, AI-powered web application designed to detect synthetic media and deepfakes. It provides an enterprise-grade, clean user interface where users can upload images or videos and instantly receive a confidence score and authenticity verdict.

## 🎬 Demo

https://github.com/user-attachments/assets/e3e8f29f-745f-4ab1-a1c6-126a60f242d7




## 🚀 Architecture

The system is built on a modern, decoupled architecture:

### 1. Frontend (Client Tier)
- **Tech Stack:** React.js, Vite, Tailwind CSS, Lucide Icons.
- **Features:** Drag-and-drop media upload, beautiful Apple-style light mode UI, dynamic real-time processing animations, and responsive layout.
- **Deployment:** Vercel

### 2. Backend (Server Tier)
- **Tech Stack:** Python, Flask, Gunicorn.
- **Features:** RESTful API endpoint (`/predict` and `/predict-image`), CORS handling, and multipart form data processing.
- **Deployment:** Google Cloud Run (Containerized via Docker)

### 3. AI Inference Engine (Deep Learning Tier)
The backend dynamically routes requests to two distinct deep learning pipelines based on the file format:
- **Image Pipeline:** Uses a custom-trained Convolutional Neural Network (CNN) to preprocess images (resizing, normalization) and output a Softmax probability for Real vs. Fake classification.
- **Video Pipeline:** Uses OpenCV and MediaPipe to extract 15 frames from the uploaded video, detect and crop facial anchors, and pass the sequence into a custom PyTorch model (CNN + LSTM architecture) for temporal forgery detection.

---

## 🛠️ Local Development Setup

To run the application locally on your machine, you need to run both the frontend and backend servers.

### 1. Start the Backend (Flask)
The backend requires Python and several data science/ML libraries (TensorFlow, PyTorch, OpenCV).

```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment (Windows)
python -m venv myenv
myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the Flask server (runs on port 5000 by default)
python app.py
```

### 2. Start the Frontend (React + Vite)
Open a new terminal window to start the frontend.

```bash
# Navigate to the frontend directory
cd truthlens-app

# Install Node modules
npm install

# Start the Vite development server
npm run dev
```
Navigate to `http://localhost:5173` in your browser to view the application!

---

## 🧠 Model Weights
To keep the repository size manageable and avoid Git LFS issues, large pre-trained model weights (`.h5` and `.pth` files) are not tracked in this repository. 
The backend is configured to automatically download these weights directly into server memory on startup from secure cloud storage using `gdown`.

---

## 🛡️ Privacy & Security
TruthLens is built with privacy in mind. Media uploaded to the backend is processed entirely in memory or temporary storage and is **instantly deleted** after the AI inference is complete. No user media is permanently stored or saved on the server.