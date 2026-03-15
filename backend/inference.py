import cv2
import numpy as np
import io
import torch
import torch.nn as nn
from PIL import Image
from tensorflow.keras.models import load_model

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
# (Required by Antigravity)
# ----------------------------

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 256 * 14 * 14 = 50176
        self.projection = nn.Linear(50176, 256)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.projection(x)
        return x


class Attention(nn.Module):

    def __init__(self,hidden_size):
        super().__init__()

        self.attention = nn.Linear(hidden_size,1)

    def forward(self,lstm_out):

        weights = torch.softmax(self.attention(lstm_out),dim=1)

        context = torch.sum(weights * lstm_out , dim=1)

        return context


class DeepfakeLSTMAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )

        self.attention = Attention(256)

        self.fc = nn.Linear(256,2)

    def forward(self,x):

        lstm_out,_ = self.lstm(x)

        context = self.attention(lstm_out)

        out = self.fc(context)

        return out


class VideoDeepfakeDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = CNNFeatureExtractor()
        self.lstm = DeepfakeLSTMAttention()

    def forward(self,frames):

        B,T,C,H,W = frames.shape

        frames = frames.reshape(B*T,C,H,W)

        features = self.cnn(frames)

        features = features.reshape(B,T,-1)

        output = self.lstm(features)

        return output


# ----------------------------
# LOAD MODELS
# ----------------------------

def load_video_model():

    print("Loading PyTorch video model...")

    model = VideoDeepfakeDetector().to(DEVICE)

    model.load_state_dict(
        torch.load("deepfake_model.pth", map_location=DEVICE)
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

        probs = torch.softmax(outputs,dim=1)

        real_prob = probs[0][0].item()
        fake_prob = probs[0][1].item()

    is_fake = fake_prob > real_prob

    return {
        "isDeepfake": is_fake,
        "confidence": round(max(fake_prob,real_prob)*100,2),
        "label": "Fake" if is_fake else "Real"
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