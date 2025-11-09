# 🛢️ Oil Spill Detection and Segmentation

## 📖 Description of the Problem
This project detects and segments oil spills in satellite radar images over water. The goal is to identify regions affected by oil contamination to assist environmental monitoring efforts.  

I trained a lightweight **U-Net segmentation model** on a publicly available dataset of oil spill images and masks. The trained model is then served via a **Flask API** as a web service for inference.

---

## ⚙️ How to Run the Project

### **1. Clone the Repository**
```bash
git clone https://github.com/<your-username>/oil-spill-segmentation.git
cd oil-spill-segmentation
```

### **2. Set Up the Environment**
Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate      # (Linux/Mac)
venv\Scripts\activate       # (Windows)
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

### **3. Download the Dataset**
The dataset is publicly available on **Zenodo**:
> [Oil Spill Detection Dataset (Zenodo)](https://zenodo.org/records/15298010)

Download both `images.zip` and `mask.zip` and extract them into:
```
/content/oil_spill_dataset/
    ├── images/
    │   ├── images/train/
    │   └── images/val/
    └── mask/
        ├── masks/train/
        └── masks/val/
```

---

### **4. Train the Model**
The training script is defined in `train.py`.

To train:
```bash
python train.py
```

This script:
- Prepares and splits the data.
- Trains a lightweight **Tiny U-Net** model.
- Saves the final model as `oil_spill_unet_full_3_layer.pth`.

---

### **5. Run the Flask Web Service**
Start the API service:
```bash
python app.py
```
The Flask server will start at:
```
http://127.0.0.1:5000/predict
```

---

### **6. Make Predictions**
You can use the provided CLI script `use.py` to send an image and visualize the prediction.

```bash
python use.py
```

This will:
- Send `download.png` (or another image) to the Flask API.
- Receive a predicted segmentation mask (base64 encoded).
- Decode and display it using OpenCV.

---

## 🧠 Data

- **Source:** [Zenodo Oil Spill Dataset](https://zenodo.org/records/15298010)
- **Structure:**
  - `train/`: training images and masks.
  - `val/`: validation images and masks (used as test set).
- Each image has a corresponding binary mask:
  - **White (255):** Oil spill area  
  - **Black (0):** Water/background  

---

## 📘 Notebook
The notebook `notebook.ipynb` includes:
- Data loading and visualization.
- Data preprocessing (resizing, normalization).
- Model architecture definition.
- Model training and validation.
- IoU and Dice score evaluations.

---

## 🧩 Scripts

### `train.py`
- Prepares training and validation data.
- Defines and trains the **Tiny U-Net** model.
- Saves the trained model to disk.

### `app.py`
- Loads the saved model.
- Provides an HTTP endpoint `/predict` for inference.
- Converts uploaded images to grayscale.
- Returns the predicted mask (base64 encoded) and shows it via OpenCV.

### `use.py`
- Sends an image to the Flask endpoint.
- Decodes and displays the predicted segmentation mask.

---

## 📦 Dependencies
List of key dependencies (in `requirements.txt`):
```
flask
torch
torchvision
pillow
numpy
opencv-python
requests
```

---

## 🐳 Docker Setup

### **Build the Docker Image**
```bash
docker build -t oil-spill-api .
```

### **Run the Container**
```bash
docker run -p 5000:5000 oil-spill-api
```

Then send requests to:
```
http://127.0.0.1:5000/predict
```

---

## 🚀 Deployment
- **Local Deployment:** Flask + Docker (CPU-only)
- **Inference Interface:** Command-line client (`use.py`)
- **Visualization:** OpenCV popup and base64 return

---

