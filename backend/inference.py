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
# MEDIAPIPE FACE DETECTION
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

# fallback detector
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------------------
# PYTORCH MODEL DEFINITIONS
# (Architecture matches deepfake_video_model.pth)
# ----------------------------

class VideoDeepfakeDetector(nn.Module):
    """
    Architecture reverse-engineered from state_dict keys:
      cnn.*              -> ResNet50 (without final fc) as self.cnn
      lstm.*             -> nn.LSTM(input_size=2048, hidden_size=128, batch_first=True)
      fc.weight [1,128]  -> nn.Linear(128, 1) with sigmoid output
    """
    def __init__(self):
        super().__init__()

        # ResNet50 backbone without final FC layer
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # outputs (B, 2048, 1, 1)

        # Single-layer LSTM: input_size=2048, hidden_size=128
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        # Final binary classification head
        self.fc = nn.Linear(128, 1)

    def forward(self, frames):
        B, T, C, H, W = frames.shape

        # Run CNN on all frames at once
        frames = frames.reshape(B * T, C, H, W)
        features = self.cnn(frames)           # (B*T, 2048, 1, 1)
        features = features.reshape(B, T, -1) # (B, T, 2048)

        # Run LSTM over frame sequence
        lstm_out, _ = self.lstm(features)     # (B, T, 128)

        # Use last timestep output
        last_out = lstm_out[:, -1, :]         # (B, 128)

        # Classify
        out = self.fc(last_out)               # (B, 1)
        return out


# ----------------------------
# LOAD MODELS
# ----------------------------

def load_video_model():

    print("Loading PyTorch video model...")

    model = VideoDeepfakeDetector().to(DEVICE)

    model.load_state_dict(
        torch.load("deepfake_video_model.pth", map_location=DEVICE)
    )

    model.eval()

    return model


def load_image_model():

    print("Loading TensorFlow image model...")

    return load_model("Img_detector.h5")


VIDEO_MODEL = load_video_model()
IMAGE_MODEL = load_image_model()

# ----------------------------
# FACE DETECTION
# ----------------------------

def pad_and_crop(image,x,y,w,h):

    H,W,_ = image.shape

    margin_x = int(w*0.1)
    margin_y = int(h*0.1)

    x1 = max(0,x-margin_x)
    y1 = max(0,y-margin_y)
    x2 = min(W,x+w+margin_x)
    y2 = min(H,y+h+margin_y)

    crop = image[y1:y2 , x1:x2]

    if crop.size > 0:
        return crop

    size = min(H,W)
    return image[0:size , 0:size]


def detect_and_crop_face(image_rgb):

    h,w,_ = image_rgb.shape

    if USE_MEDIAPIPE and face_detection:

        results = face_detection.process(image_rgb)

        if results.detections:

            bbox = results.detections[0].location_data.relative_bounding_box

            x = int(bbox.xmin*w)
            y = int(bbox.ymin*h)
            width = int(bbox.width*w)
            height = int(bbox.height*h)

            return pad_and_crop(image_rgb,x,y,width,height)

    gray = cv2.cvtColor(image_rgb,cv2.COLOR_RGB2GRAY)

    faces = face_classifier.detectMultiScale(
        gray,1.1,5,minSize=(30,30)
    )

    if len(faces)>0:

        x,y,w,h = faces[0]

        return pad_and_crop(image_rgb,x,y,w,h)

    return image_rgb

# ----------------------------
# PREPROCESS FRAME
# ----------------------------

def preprocess_frame(frame_rgb,target_size=(224,224)):

    face = detect_and_crop_face(frame_rgb)

    face = cv2.resize(face,target_size)

    face = face.astype(np.float32)/255.0
    
    # Normalize using ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    face = (face - mean) / std

    return face

# ----------------------------
# FRAME EXTRACTION
# ----------------------------

def extract_frames(video_path,num_frames=15):

    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = np.linspace(0,total-1,num_frames).astype(int)

    frames = []

    i = 0

    while cap.isOpened():

        ret,frame = cap.read()

        if not ret:
            break

        if i in indices:

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            frames.append(preprocess_frame(frame))

        i+=1

    cap.release()

    return np.array(frames)

# ----------------------------
# VIDEO INFERENCE
# ----------------------------

def run_video_inference(video_path):

    sequence = extract_frames(video_path)

    # numpy -> torch

    tensor = torch.tensor(sequence).permute(0,3,1,2)

    tensor = tensor.unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():

        outputs = VIDEO_MODEL(tensor)

        # Apply sigmoid to output
        prob = torch.sigmoid(outputs)
        
        # Support single logit output securely
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

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_np = np.array(image)

    processed = preprocess_frame(image_np)

    tensor = np.expand_dims(processed,axis=0)

    outputs = IMAGE_MODEL.predict(tensor)

    preds = outputs[0]

    if len(preds)==1:

        fake_prob = float(preds[0])
        real_prob = 1-fake_prob

    else:

        fake_prob = float(preds[1])
        real_prob = float(preds[0])

    is_fake = fake_prob > real_prob

    return {
        "isDeepfake": is_fake,
        "confidence": round(max(fake_prob,real_prob)*100,2),
        "label": "Fake" if is_fake else "Real"
    }