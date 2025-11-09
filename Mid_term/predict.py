from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import numpy as np
import base64
import cv2

# --- Flask app ---
app = Flask(__name__)

# --- Load model ---
model = torch.load("oil_spill_unet_full.pth")
model.eval()

# --- Image preprocessing ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))[0].cpu().numpy()[0]
    
    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    # Convert mask to base64 to return as JSON
    _, buffer = cv2.imencode('.png', pred_mask)
    mask_base64 = base64.b64encode(buffer).decode('utf-8')
    return mask_base64

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    mask_base64 = predict_mask(file.read())
    
    return jsonify({'mask': mask_base64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
