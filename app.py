from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# ── CNN Model (must match training) ──────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), 
            nn.Conv2d(8, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = "cnn4.pth"

device = torch.device("cpu")  # Render free tier uses CPU

model = CNN().to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)

# ── Transform (must match training) ──────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ── Routes ────────────────────────────────────────────────────────────────────

# 🔹 Frontend route
@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        return f"ERROR: {str(e)}"

# 🔹 Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        # Read and process image
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(tensor).item()

        label = "Real" if output > 0.5 else "Fake"
        confidence = output if output > 0.5 else 1 - output

        return jsonify({
            "label": label,
            "confidence": round(confidence * 100, 2),
            "raw_score": round(output, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔹 Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ── Run app ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)