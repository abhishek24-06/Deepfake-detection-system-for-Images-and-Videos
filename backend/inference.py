import os
import cv2
import numpy as np
import io
from PIL import Image
import gdown
import onnxruntime as ort

# ----------------------------
# GOOGLE DRIVE MODEL URLS
# ----------------------------

# Image model (9 MB) is committed to git — no download needed
# Video model (94 MB) is downloaded from Google Drive at first request
VID_ONNX_URL = os.environ.get(
    "VID_ONNX_URL",
    "https://drive.google.com/uc?id=1FMT1eEI2gFuI4d4vK8Kq3Fl0Gmm-wMVb"
)

def download_video_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vid_path = os.path.join(base_dir, "deepfake_video_model.onnx")

    if not os.path.exists(vid_path):
        print("Downloading video ONNX model from Google Drive...")
        gdown.download(VID_ONNX_URL, vid_path, quiet=False)
        print("✅ Video model downloaded")
    return vid_path

# ----------------------------
# FACE DETECTION (Haar Cascade)
# ----------------------------

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------------------
# LOAD MODELS
# ----------------------------

IMAGE_SESSION = None
VIDEO_SESSION = None

def load_image_session():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "Img_detector.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Image ONNX model not found at {model_path}")
    print("Loading image ONNX model...")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("✅ Image model loaded")
    return sess

def load_video_session():
    vid_path = download_video_model()
    print("Loading video ONNX model...")
    sess = ort.InferenceSession(vid_path, providers=["CPUExecutionProvider"])
    print("✅ Video model loaded")
    return sess

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

def preprocess_for_video(frame_rgb, target_size=(224, 224)):
    """ImageNet normalization for ResNet50 backbone."""
    face = detect_and_crop_face(frame_rgb)
    face = cv2.resize(face, target_size).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (face - mean) / std   # (224, 224, 3)


def preprocess_for_image(image_rgb, target_size=(224, 224)):
    """MobileNetV2 expects float32 in [0, 1]."""
    face = detect_and_crop_face(image_rgb)
    face = cv2.resize(face, target_size).astype(np.float32) / 255.0
    return np.expand_dims(face, axis=0)   # (1, 224, 224, 3)

# ----------------------------
# FRAME EXTRACTION (15 frames)
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
    return np.array(frames)   # (15, 224, 224, 3)

# ----------------------------
# VIDEO INFERENCE
# ----------------------------

def run_video_inference(video_path):
    global VIDEO_SESSION
    if VIDEO_SESSION is None:
        VIDEO_SESSION = load_video_session()

    frames = extract_frames(video_path)           # (15, 224, 224, 3)
    # ONNX model expects (batch, T, C, H, W)
    tensor = frames.transpose(0, 3, 1, 2)         # (15, 3, 224, 224)
    tensor = tensor[np.newaxis, ...].astype(np.float32)  # (1, 15, 3, 224, 224)

    input_name = VIDEO_SESSION.get_inputs()[0].name
    logit = VIDEO_SESSION.run(None, {input_name: tensor})[0]   # (1, 1)

    # Apply sigmoid manually (model outputs raw logit)
    fake_prob = float(1 / (1 + np.exp(-logit[0][0])))
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
    global IMAGE_SESSION
    if IMAGE_SESSION is None:
        IMAGE_SESSION = load_image_session()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_rgb = np.array(image)

    tensor = preprocess_for_image(image_rgb)   # (1, 224, 224, 3)

    input_name = IMAGE_SESSION.get_inputs()[0].name
    out = IMAGE_SESSION.run(None, {input_name: tensor})[0]  # (1, 1) already sigmoid

    fake_prob = float(out[0][0])
    is_fake = fake_prob > 0.5

    return {
        "isDeepfake": is_fake,
        "confidence": round(fake_prob * 100, 2) if is_fake else round((1 - fake_prob) * 100, 2),
        "label": "Fake" if is_fake else "Real"
    }
