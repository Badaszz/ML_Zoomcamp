from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import base64  # <-- missing in your version
from model import TinyUNet, DoubleConv

app = Flask(__name__)

# -----------------------------
# Load your trained model
# -----------------------------
device = torch.device("cpu")  # CPU only
model = torch.load("oil_spill_unet_full_3_layer.pth", map_location=device, weights_only=False)
model.eval()

# -----------------------------
# Transform (grayscale + tensor)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),  # convert to grayscale
    transforms.ToTensor()
])

# -----------------------------
# Prediction function
# -----------------------------
def predict_mask(image_bytes):
    # Load image and convert to grayscale
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    img_tensor = transform(image).unsqueeze(0)
    img_tensor = img_tensor / 255.0
    # Model inference
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))[0].cpu().numpy()[0]
    mask = (pred > 0.001).astype(np.uint8) * 255

    # Convert PIL image to numpy for display
    img_np = np.array(image.resize((128, 128)))  # <-- match mask size here
    

    # Show side-by-side using OpenCV
    combined = np.hstack([img_np, mask])
    cv2.imshow("Original (grayscale) | Predicted Mask", combined)
    cv2.waitKey(1)
    
    print(f"Tensor shape before final index: {pred.shape}")

    return combined,pred

# -----------------------------
# Flask route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    mask,pred = predict_mask(file.read())

    # Encode mask as base64
    _, buffer = cv2.imencode('.png', mask)
    mask_base64 = base64.b64encode(buffer).decode('utf-8')
    print(f"DEBUG: Min Value: {pred.min():.6f}")
    print(f"DEBUG: Max Value: {pred.max():.6f}")
    print(f"DEBUG: Mean Value: {pred.mean():.6f}")

    return jsonify({'min': mask_base64, 'message': 'Prediction successful', 'pred_shape': pred.shape})

# -----------------------------
# Run the Flask app
# -----------------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
