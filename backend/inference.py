import os
import cv2
import numpy as np
import io
import torch
import torch.nn as nn
from PIL import Image
from tensorflow.keras.models import load_model
import torchvision.models as models


# ----------------------------
# DEVICE
# ----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# FACE DETECTION — MediaPipe with Haar cascade fallback
# ----------------------------

mp_face_detection = None
face_detection = None
USE_MEDIAPIPE = False

try:
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        USE_MEDIAPIPE = True
    else:
        print("MediaPipe solutions not available. Using Haar cascade.")
except ImportError:
    print("MediaPipe not installed. Using Haar cascade.")

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------------------
# VIDEO MODEL (ResNet50 + LSTM)
# ----------------------------

class VideoDeepfakeDetector(nn.Module):
    """
    Architecture matches deepfake_video_model.pth:
      cnn.*  -> ResNet50 (without final fc)
      lstm.* -> nn.LSTM(input_size=2048, hidden_size=128, batch_first=True)
      fc.*   -> nn.Linear(128, 1)
    """
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        frames = frames.reshape(B * T, C, H, W)
        features = self.cnn(frames)           # (B*T, 2048, 1, 1)
        features = features.reshape(B, T, -1) # (B, T, 2048)
        lstm_out, _ = self.lstm(features)     # (B, T, 128)
        last_out = lstm_out[:, -1, :]         # (B, 128)
        return self.fc(last_out)              # (B, 1)


# ----------------------------
# LOAD MODELS
# ----------------------------

VIDEO_MODEL = None
IMAGE_MODEL = None


def load_video_model():
    print("Loading PyTorch video model...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "deepfake_video_model.pth")
    model = VideoDeepfakeDetector().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("✅ Video model loaded")
    return model


def load_image_model():
    print("Loading TensorFlow image model (Img_detector.keras)...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "Img_detector.keras")
    model = load_model(model_path, compile=False)
    print("✅ Image model loaded")
    return model


# ----------------------------
# FACE DETECTION HELPER
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
    h, w, _ = image_rgb.shape

    if USE_MEDIAPIPE and face_detection:
        results = face_detection.process(image_rgb)
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            return pad_and_crop(image_rgb, x, y, width, height)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, fw, fh = faces[0]
        return pad_and_crop(image_rgb, x, y, fw, fh)

    return image_rgb


# ----------------------------
# PREPROCESSING
# ----------------------------

def preprocess_for_video(frame_rgb, target_size=(224, 224)):
    """ImageNet normalization for ResNet50 backbone."""
    face = detect_and_crop_face(frame_rgb)
    face = cv2.resize(face, target_size).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (face - mean) / std


def preprocess_for_image(image_rgb, target_size=(224, 224)):
    """MobileNetV2 preprocessing: maps [0, 255] -> [-1, 1]."""
    face = detect_and_crop_face(image_rgb)
    face = cv2.resize(face, target_size).astype(np.float32)
    face = (face - 127.5) / 127.5          # [-1, 1]
    return np.expand_dims(face, axis=0)    # (1, 224, 224, 3)


# ----------------------------
# FRAME EXTRACTION
# ----------------------------

def extract_frames(video_path, num_frames=15):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(np.linspace(0, total - 1, num_frames).astype(int))
    frames = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess_for_video(frame_rgb))
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
        fake_prob = prob.item() if prob.numel() == 1 else prob[0][0].item()

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

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_rgb = np.array(image)

    tensor = preprocess_for_image(image_rgb)     # (1, 224, 224, 3) in [-1,1]
    outputs = IMAGE_MODEL.predict(tensor, verbose=0)
    preds = outputs[0]

    if len(preds) == 1:
        fake_prob = float(preds[0])
        real_prob = 1 - fake_prob
    else:
        fake_prob = float(preds[1])
        real_prob = float(preds[0])

    is_fake = fake_prob > real_prob
    return {
        "isDeepfake": is_fake,
        "confidence": round(max(fake_prob, real_prob) * 100, 2),
        "label": "Fake" if is_fake else "Real"
    }
