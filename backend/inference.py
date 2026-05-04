import os
import cv2
import numpy as np
import io
import torch
import torch.nn as nn
from PIL import Image
import torchvision.models as models
import gdown
import tensorflow as tf

# ----------------------------
# DEVICE (PyTorch — video model)
# ----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# GOOGLE DRIVE MODEL URLS (video only)
# ----------------------------

VID_URL = "https://drive.google.com/uc?id=1JIgN0hZ0kTjttiSosUd1NZ5D9S9b9rHd"

def download_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vid_path = os.path.join(base_dir, "deepfake_video_model.pth")

    if not os.path.exists(vid_path):
        print("Downloading video model from Google Drive...")
        gdown.download(VID_URL, vid_path, quiet=False)

# ----------------------------
# FACE DETECTION (Haar Cascade)
# ----------------------------

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------------------
# VIDEO MODEL ARCHITECTURE (ResNet50 + LSTM)
# ----------------------------

class VideoDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(weights=None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        self.lstm = nn.LSTM(2048, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, frames):
        B, T, C, H, W = frames.shape

        frames = frames.reshape(B * T, C, H, W)
        features = self.cnn(frames)
        features = features.reshape(B, T, -1)

        lstm_out, _ = self.lstm(features)
        last_out = lstm_out[:, -1, :]

        out = self.fc(last_out)
        return out

# ----------------------------
# LOAD MODELS
# ----------------------------

def load_video_model():
    print("Loading video model (ResNet50+LSTM)...")
    download_models()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "deepfake_video_model.pth")

    model = VideoDeepfakeDetector().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("✅ Video model loaded")
    return model


def load_image_model():
    """
    Load the local MobileNetV2-based Keras image model.
    Model output: (None, 1) sigmoid — value > 0.5 means FAKE.
    """
    print("Loading image model (MobileNetV2 Keras)...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "Img_detector.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Image model not found at {model_path}. "
            "Make sure Img_detector.keras is in the backend/ directory."
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False
    )
    print("✅ Image model loaded")
    return model


VIDEO_MODEL = None
IMAGE_MODEL = None

# ----------------------------
# FACE CROPPING
# ----------------------------

def pad_and_crop(image, x, y, w, h):
    H, W, _ = image.shape

    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(W, x + w + margin_x)
    y2 = min(H, y + h + margin_y)

    crop = image[y1:y2, x1:x2]

    if crop.size > 0:
        return crop

    size = min(H, W)
    return image[0:size, 0:size]


def detect_and_crop_face(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        return pad_and_crop(image_rgb, x, y, w, h)

    return image_rgb

# ----------------------------
# PREPROCESSING
# ----------------------------

def preprocess_frame(frame_rgb, target_size=(224, 224)):
    """For video frames — ImageNet normalization for PyTorch ResNet."""
    face = detect_and_crop_face(frame_rgb)
    face = cv2.resize(face, target_size)
    face = face.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    face = (face - mean) / std

    return face


def preprocess_image_for_keras(image_rgb, target_size=(224, 224)):
    """For image — MobileNetV2 expects [0, 1] float32."""
    face = detect_and_crop_face(image_rgb)
    face = cv2.resize(face, target_size)
    face = face.astype(np.float32) / 255.0   # [0, 1]
    face = np.expand_dims(face, axis=0)       # (1, 224, 224, 3)
    return face

# ----------------------------
# FRAME EXTRACTION (15 frames)
# ----------------------------

def extract_frames(video_path, num_frames=15):
    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames).astype(int)

    frames = []
    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if i in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess_frame(frame))

        i += 1

    cap.release()
    return np.array(frames)

# ----------------------------
# VIDEO INFERENCE
# ----------------------------

def run_video_inference(video_path):
    global VIDEO_MODEL
    if VIDEO_MODEL is None:
        VIDEO_MODEL = load_video_model()

    sequence = extract_frames(video_path)

    tensor = torch.tensor(sequence).permute(0, 3, 1, 2)
    tensor = tensor.unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        outputs = VIDEO_MODEL(tensor)
        prob = torch.sigmoid(outputs)
        fake_prob = prob.item()

    is_fake = fake_prob > 0.5

    return {
        "isDeepfake": is_fake,
        "confidence": round(fake_prob * 100, 2) if is_fake else round((1 - fake_prob) * 100, 2),
        "label": "Fake Video" if is_fake else "Real Video"
    }

# ----------------------------
# IMAGE INFERENCE
# ----------------------------

def run_image_inference(image_bytes):
    global IMAGE_MODEL
    if IMAGE_MODEL is None:
        IMAGE_MODEL = load_image_model()

    # Open and convert to RGB numpy array
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_rgb = np.array(image)

    # Preprocess: detect face, crop, resize, normalize
    tensor = preprocess_image_for_keras(image_rgb)

    # Predict — output shape (1, 1), sigmoid activation
    prediction = IMAGE_MODEL.predict(tensor, verbose=0)
    fake_prob = float(prediction[0][0])

    is_fake = fake_prob > 0.5

    return {
        "isDeepfake": is_fake,
        "confidence": round(fake_prob * 100, 2) if is_fake else round((1 - fake_prob) * 100, 2),
        "label": "Fake" if is_fake else "Real"
    }
