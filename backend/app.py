import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import run_video_inference, run_image_inference

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'temp_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return "TruthLens backend is running 🚀"


@app.route('/predict', methods=['POST'])
def predict_video():
    """
    Endpoint for analyzing video files.
    Pipeline: Video -> Frame Extraction -> Face Detection -> Crop ->
    Resize (224x224) -> 15-frame sequence -> CNN+LSTM -> Prediction
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        temp_filename = str(uuid.uuid4()) + "_" + file.filename
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        file.save(temp_path)

        result = run_video_inference(temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify(result)

    except Exception as e:
        print(f"Error during video inference: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500


@app.route('/predict-image', methods=['POST'])
def predict_image():
    """
    Endpoint for analyzing static image files.
    Pipeline: Image -> Face Detection -> Crop -> Resize -> Model -> Prediction
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        result = run_image_inference(image_bytes)
        return jsonify(result)

    except Exception as e:
        print(f"Error during image inference: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting TruthLens backend on http://localhost:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)
