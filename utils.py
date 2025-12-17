import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.transform import Rotation as R

class FeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, frames):
        if not frames: return torch.empty(0, 512).to(self.device)
        batch_tensor = torch.stack([self.preprocess(f) for f in frames]).to(self.device)
        with torch.no_grad():
            features = self.backbone(batch_tensor)
        return features.squeeze() 

def compute_kinematics(traj_df):
    pos = traj_df[['x', 'y', 'z']].values.astype(np.float32)
    quats = traj_df[['q_x', 'q_y', 'q_z', 'q_w']].values.astype(np.float32)
    
    norms = np.linalg.norm(quats, axis=1)
    bad_indices = norms < 1e-6
    if np.any(bad_indices):
        if bad_indices[0]: quats[0] = [0, 0, 0, 1]
        for i in range(1, len(quats)):
            if np.linalg.norm(quats[i]) < 1e-6: quats[i] = quats[i-1]

    vel_vec = np.diff(pos, axis=0, prepend=pos[0:1])
    rots = R.from_quat(quats)
    rot_diffs = rots[1:] * rots[:-1].inv()
    rot_vecs = rot_diffs.as_rotvec()
    rot_vecs = np.vstack([[0,0,0], rot_vecs]) 
    
    if np.any(bad_indices):
        mask = bad_indices.flatten()
        vel_vec[mask] = 0; rot_vecs[mask] = 0
        
    return np.hstack([vel_vec, rot_vecs]).astype(np.float32)

def load_video_robust(path, resize_dim=(224, 224)):
    if not os.path.exists(path): return []
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(cv2.resize(f, resize_dim))
    cap.release()
    return frames

def load_config(path="config.yaml"):
    import yaml
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
