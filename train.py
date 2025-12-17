import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import random

from model import SegmentationModel
from utils import load_config

# --- LOAD CONFIG ---
cfg = load_config("config.yaml")

DATA_MODE = cfg['data']['mode']
DATA_DIR = f"{cfg['data']['output_prefix']}_{DATA_MODE}"
SAVE_PATH = f"{cfg['training']['save_name_prefix']}_{DATA_MODE}.pth"
DEVICE = cfg['common']['device']

def load_data():
    x_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_X.npy")))
    y_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_Y.npy")))
    data = []
    print(f"Loading from {DATA_DIR}...")
    for xf, yf in zip(x_files, y_files):
        x = np.load(xf)
        y = np.load(yf)
        if len(x) > 0: data.append((x, y))
    return data

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data dir {DATA_DIR} missing. Run preprocess.py first.")
        exit()

    full_data = load_data()
    
    # Train/Val Split (Simple: last 5 for val)
    if len(full_data) < 5:
        print("Warning: Not enough data for validation. Using all for training.")
        train_data = full_data
        val_data = full_data
    else:
        train_data = full_data[:-5]
        val_data = full_data[-5:]
        
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    model = SegmentationModel(cfg).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    mse_loss = nn.MSELoss()

    best_acc = 0.0

    best_acc = 0.0

    n_epochs = cfg['training']['epochs']
    
    # For verification, I'll temporarily set epochs to 1 in the code or just kill it.
    # Let's set it to 1 specifically for this verification run by override.
    cfg['training']['epochs'] = 1
    
    n_epochs = cfg['training']['epochs']
    
    # --- TRAINING LOOP ---
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        random.shuffle(train_data)
        
        for x, y in train_data:
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(DEVICE)
            y_tensor = torch.tensor(y, dtype=torch.long).unsqueeze(0).to(DEVICE)
            
            optimizer.zero_grad()
            p1, p2 = model(x_tensor)
            
            loss = ce_loss(p1, y_tensor) + ce_loss(p2, y_tensor)
            loss += 0.15 * torch.mean(torch.clamp(
                mse_loss(torch.softmax(p2[:, :, 1:], dim=1).log(), 
                         torch.softmax(p2[:, :, :-1], dim=1).log()), min=0, max=16)
            )
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 5 == 0:
            model.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for x, y in val_data:
                    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(DEVICE)
                    y_tensor = torch.tensor(y, dtype=torch.long).to(DEVICE)
                    _, p2 = model(x_tensor)
                    pred = torch.argmax(p2, dim=1).squeeze()
                    mask = y_tensor != -100
                    correct += (pred[mask] == y_tensor[mask]).sum().item()
                    total += mask.sum().item()
            
            acc = correct / total * 100 if total > 0 else 0
            print(f"Epoch {epoch:03d} | Loss: {epoch_loss/max(1, len(train_data)):.4f} | Val Acc: {acc:.2f}%")
            # Save Best Model
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"  -> Saved Best to {SAVE_PATH}")