import sys
sys.path.insert(0, r'D:\Python_packages')
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import json
import os
from tqdm import tqdm
from utils import get_device, INDEX_TO_CLASS, CLASS_DISPLAY_NAMES
from model import build_model

OUTPUT_DIR = r'D:\Derm AI_Nexus\outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds), np.vstack(all_probs)

def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    class_names = [CLASS_DISPLAY_NAMES[INDEX_TO_CLASS[i]] for i in range(7)]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - DermAI Nexus', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print('Confusion matrix saved.')

def plot_roc_curves(labels, probs):
    classes = list(range(7))
    labels_bin = label_binarize(labels, classes=classes)
    plt.figure(figsize=(10, 8))
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf']
    for i, color in zip(classes, colors):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        auc = roc_auc_score(labels_bin[:, i], probs[:, i])
        class_name = CLASS_DISPLAY_NAMES[INDEX_TO_CLASS[i]]
        plt.plot(fpr, tpr, color=color, label=f'{class_name} (AUC = {auc:.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - DermAI Nexus')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'), dpi=150)
    plt.close()
    print('ROC curves saved.')

def plot_training_history(history_path):
    with open(history_path) as f:
        history = json.load(f)
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   label='Val Loss')
    ax1.axvline(x=10, color='gray', linestyle='--', label='Phase 2 starts')
    ax1.set_title('Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'],   label='Val Accuracy')
    ax2.axvline(x=10, color='gray', linestyle='--', label='Phase 2 starts')
    ax2.set_title('Accuracy Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150)
    plt.close()
    print('Training history plot saved.')

def run_evaluation(model, test_loader):
    device = get_device()
    labels, preds, probs = get_predictions(model, test_loader, device)
    class_names = [CLASS_DISPLAY_NAMES[INDEX_TO_CLASS[i]] for i in range(7)]
    print('\n' + '='*60)
    print('CLASSIFICATION REPORT')
    print('='*60)
    print(classification_report(labels, preds, target_names=class_names))
    labels_bin = label_binarize(labels, classes=list(range(7)))
    macro_auc = roc_auc_score(labels_bin, probs, multi_class='ovr', average='macro')
    print(f'Macro ROC-AUC Score: {macro_auc:.4f}')
    plot_confusion_matrix(labels, preds)
    plot_roc_curves(labels, probs)
    plot_training_history(r'D:\Derm AI_Nexus\models\training_history.json')
    print('\nAll evaluation outputs saved to D:\\Derm AI_Nexus\\outputs\\')
