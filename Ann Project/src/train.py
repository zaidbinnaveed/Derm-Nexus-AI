import sys
sys.path.insert(0, r'D:\Python_packages')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import json
from tqdm import tqdm
from utils import get_device, save_model, set_seed
from dataset import prepare_dataloaders
from model import build_model, unfreeze_backbone

CONFIG = {
    'data_dir':          r'D:\Derm AI_Nexus\data',
    'model_save_path':   r'D:\Derm AI_Nexus\models\dermai_nexus_model.pth',
    'history_save_path': r'D:\Derm AI_Nexus\models\training_history.json',
    'batch_size':        32,
    'epochs_phase1':     10,
    'epochs_phase2':     15,
    'lr_phase1':         1e-3,
    'lr_phase2':         1e-4,
    'patience':          5,
    'use_class_weights': True,
}

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100. * correct / total

def run_training():
    set_seed(42)
    device = get_device()
    train_loader, val_loader, test_loader, class_weights = prepare_dataloaders(CONFIG['data_dir'], CONFIG['batch_size'])
    model = build_model(num_classes=7, freeze_backbone=True)
    model = model.to(device)
    if CONFIG['use_class_weights']:
        from utils import CLASS_NAMES
        weights = torch.tensor(
            [class_weights[cls] for cls in sorted(CLASS_NAMES, key=CLASS_NAMES.get)],
            dtype=torch.float32
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    print('\n' + '='*50)
    print('PHASE 1: Training classifier head')
    print('='*50)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr_phase1'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs_phase1'])
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)

    for epoch in range(CONFIG['epochs_phase1']):
        print(f'\nEpoch [{epoch+1}/{CONFIG["epochs_phase1"]}]')
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, CONFIG['model_save_path'])
            print('Best model saved')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print('Early stopping triggered.')
                break

    print('\n' + '='*50)
    print('PHASE 2: Fine-tuning with unfrozen backbone')
    print('='*50)
    model = unfreeze_backbone(model, unfreeze_last_n_blocks=3)
    model = model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr_phase2'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs_phase2'])
    patience_counter = 0

    for epoch in range(CONFIG['epochs_phase2']):
        print(f'\nEpoch [{epoch+1}/{CONFIG["epochs_phase2"]}]')
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, CONFIG['model_save_path'])
            print('Best model saved')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print('Early stopping triggered.')
                break

    os.makedirs(os.path.dirname(CONFIG['history_save_path']), exist_ok=True)
    with open(CONFIG['history_save_path'], 'w') as f:
        json.dump(history, f)
    print('\nTraining complete. History saved.')
    return model, test_loader, history

if __name__ == '__main__':
    model, test_loader, history = run_training()
    from evaluate import run_evaluation
    run_evaluation(model, test_loader)
