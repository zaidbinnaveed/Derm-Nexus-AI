import sys
sys.path.insert(0, '/app/src')

import os
import io
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)
CORS(app)

MODEL_PATH = '/tmp/dermai_nexus_model.pth'
GDRIVE_FILE_ID = '1zN1J7YvFZGaHo7rrlhUh0d6ByNhkalAl'
IMAGE_SIZE = 224

RISK_MAP = {
    'mel':   'high',
    'bcc':   'high',
    'akiec': 'high',
    'bkl':   'medium',
    'df':    'low',
    'nv':    'low',
    'vasc':  'medium',
}

CONDITION_INFO = {
    'mel':   'Melanoma is a serious form of skin cancer. Urgent dermatologist review is strongly advised.',
    'bcc':   'Basal Cell Carcinoma is the most common skin cancer. Usually treatable when caught early.',
    'akiec': 'Actinic Keratoses are pre-cancerous lesions caused by sun damage. Should be reviewed soon.',
    'bkl':   'Benign Keratosis is a non-cancerous skin growth. Still worth confirming with a dermatologist.',
    'df':    'Dermatofibroma is a benign skin nodule. Generally harmless but monitor for changes.',
    'nv':    'Melanocytic Nevi (common mole) is typically benign. Monitor for changes in size or color.',
    'vasc':  'Vascular Lesion involves blood vessels in the skin. Usually benign but review is advised.',
}

CLASS_NAMES = {
    'akiec': 0, 'bcc': 1, 'bkl': 2,
    'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
}
INDEX_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}
CLASS_DISPLAY_NAMES = {
    'akiec': 'Actinic Keratoses',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevi',
    'vasc':  'Vascular Lesions'
}

transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def download_model():
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        if file_size > 1000000:
            print(f"Model already downloaded: {file_size / 1024 / 1024:.1f} MB")
            return
        else:
            os.remove(MODEL_PATH)
    
    print("Downloading model from Google Drive using gdown...")
    import gdown
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    
    file_size = os.path.getsize(MODEL_PATH)
    print(f"Model downloaded: {file_size / 1024 / 1024:.1f} MB")
    if file_size < 1000000:
        raise Exception(f"Download failed — file too small: {file_size} bytes")

# Download and load model at startup
# Download and load model at startup
download_model()
device = torch.device('cpu')

from torchvision import models
import torch.nn as nn

def build_model(num_classes=7):
    m = models.efficientnet_b3(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )
    return m

model = build_model(num_classes=7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval()
print("Model loaded and ready.")

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    image_np = np.array(image)

    transformed = transform(image=image_np)['image']
    tensor = transformed.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    indices = np.argsort(probs)[::-1]
    results = []
    for i in indices[:3]:
        cls = INDEX_TO_CLASS[i]
        results.append({
            'class': cls,
            'displayName': CLASS_DISPLAY_NAMES[cls],
            'probability': round(float(probs[i]) * 100, 1),
            'risk': RISK_MAP[cls],
            'info': CONDITION_INFO[cls],
        })

    top = results[0]
    regions = [
        {
            'id': 'r1',
            'x': 20, 'y': 22, 'width': 22, 'height': 24,
            'confidence': round(float(probs[indices[0]]) * 100),
            'label': top['displayName'],
            'risk': top['risk'],
        }
    ]
    if results[1]['probability'] > 20:
        regions.append({
            'id': 'r2',
            'x': 55, 'y': 40, 'width': 18, 'height': 18,
            'confidence': round(results[1]['probability']),
            'label': results[1]['displayName'],
            'risk': results[1]['risk'],
        })

    return jsonify({
        'regions': regions,
        'topPrediction': top,
        'allPredictions': results,
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)