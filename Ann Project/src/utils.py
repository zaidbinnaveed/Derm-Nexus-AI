import sys
sys.path.insert(0, r'D:\Python_packages')
import os
import torch
import random
import numpy as np

CLASS_NAMES = {
    'akiec': 0,
    'bcc':   1,
    'bkl':   2,
    'df':    3,
    'mel':   4,
    'nv':    5,
    'vasc':  6
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    return device

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f'Model loaded from {path}')
    return model
