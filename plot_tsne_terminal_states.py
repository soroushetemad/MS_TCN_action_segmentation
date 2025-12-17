import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# --- CONFIG ---
CONFIG_PATH = "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

# --- FEATURE EXTRACTOR ---
class FeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.device = device
        self.to(device)
        self.eval()

    def forward(self, frame):
        # frame: numpy array (H, W, C) BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(frame_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.backbone(tensor)
        return features.squeeze().cpu().numpy()

# --- KINEMATICS HELPER ---
def get_kinematics_at_frame(traj_df, frame_idx):
    # This is a simplified version of compute_kinematics_6d for a single frame.
    # Ideally, we should compute the full trajectory features and then pick the index,
    # because velocity/rotation diffs depend on neighbors.
    # For efficiency, we'll compute it for the window around the frame or the whole traj.
    # Let's compute for the whole traj once per episode to be safe and consistent.
    
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
        
    kinematics = np.hstack([vel_vec, rot_vecs]).astype(np.float32)
    
    if frame_idx < len(kinematics):
        return kinematics[frame_idx]
    else:
        return kinematics[-1]

def extract_frames_from_video(vid_path, frame_indices):
    if not os.path.exists(vid_path):
        return {}
    
    cap = cv2.VideoCapture(vid_path)
    frames = {}
    target_indices = set(frame_indices)
    max_idx = max(target_indices) if target_indices else -1
    
    current_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if current_idx in target_indices:
            frames[current_idx] = frame
            
        if current_idx >= max_idx:
            break
            
        current_idx += 1
            
    cap.release()
    return frames

def main():
    cfg = load_config()
    
    base_dir = cfg['data']['dataset_root']
    data_mode = cfg['data']['mode']
    use_traj = cfg['data'].get('use_trajectory', True)
    files = cfg['data']['files']
    vid_name = files['raw_video']
    traj_name = files['trajectory']
    lbl_prefix = files['pred_labels_prefix']
    tac0_name = files['tactile_0']
    tac1_name = files['tactile_1']
    tsne_out = files['tsne_plot']
    labels_list = cfg['common']['labels']
    colors_cfg = cfg['common']['colors']
    colors_map = [tuple(np.array(c) / 255.0) for c in colors_cfg]
    device = cfg['common']['device'] if torch.cuda.is_available() else "cpu"

    print(f"Mode: {data_mode}, Device: {device}")
    
    extractor = FeatureExtractor(device)
    
    terminal_features = []
    terminal_labels = []
    
    episodes = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    print(f"Found {len(episodes)} episodes.")
    
    for ep in tqdm(episodes):
        ep_dir = os.path.join(base_dir, ep)
        lbl_path = os.path.join(ep_dir, f"{lbl_prefix}_{data_mode}.csv")
        
        if not os.path.exists(lbl_path):
            continue
            
        df_lbl = pd.read_csv(lbl_path)
        
        # Identify terminal states: where label changes
        # We look for indices i where label[i] != label[i+1]
        # The "terminal state" for the action ending at i is frame i.
        
        labels_indices = df_lbl['label_idx'].values
        
        # Use frame_idx column if available to map row index to actual frame index
        if 'frame_idx' in df_lbl.columns:
            frame_mapping = df_lbl['frame_idx'].values
        else:
            frame_mapping = np.arange(len(labels_indices))

        # Create a mask for transitions
        # transition at i means i is the last frame of the current label
        transitions = np.where(labels_indices[:-1] != labels_indices[1:])[0]
        
        # Add the last frame index
        terminal_row_indices = np.append(transitions, len(labels_indices) - 1)
        
        # Map to actual frame indices
        terminal_frame_indices = frame_mapping[terminal_row_indices]
        
        # Load trajectory once if needed
        traj_df = None
        if use_traj:
            traj_path = os.path.join(ep_dir, traj_name)
            if os.path.exists(traj_path):
                traj_df = pd.read_csv(traj_path)
        
        # Batch extract frames
        vid_path = os.path.join(ep_dir, vid_name)
        tac0_path = os.path.join(ep_dir, tac0_name)
        tac1_path = os.path.join(ep_dir, tac1_name)
        
        rgb_frames = {}
        tac0_frames = {}
        tac1_frames = {}
        
        if data_mode == "rgb" or data_mode == "all":
            rgb_frames = extract_frames_from_video(vid_path, terminal_frame_indices)
            
        if data_mode == "tactile" or data_mode == "all":
            tac0_frames = extract_frames_from_video(tac0_path, terminal_frame_indices)
            tac1_frames = extract_frames_from_video(tac1_path, terminal_frame_indices)
            
        for i, row_idx in enumerate(terminal_row_indices):
            frame_idx = terminal_frame_indices[i]
            label_idx = labels_indices[row_idx]
            
            features_list = []
            
            # 1. RGB
            if data_mode == "rgb" or data_mode == "all":
                frame = rgb_frames.get(frame_idx)
                if frame is not None:
                    frame = cv2.resize(frame, (224, 224)) # Match preprocess_v4.py
                    features_list.append(extractor(frame))
                else:
                    continue

            # 2. Tactile
            if data_mode == "tactile" or data_mode == "all":
                f0 = tac0_frames.get(frame_idx)
                f1 = tac1_frames.get(frame_idx)
                
                if f0 is not None: 
                    f0 = cv2.resize(f0, (224, 224))
                    features_list.append(extractor(f0))
                else: features_list.append(np.zeros(512, dtype=np.float32))
                    
                if f1 is not None: 
                    f1 = cv2.resize(f1, (224, 224))
                    features_list.append(extractor(f1))
                else: features_list.append(np.zeros(512, dtype=np.float32))

            # 3. Kinematics
            if use_traj and traj_df is not None:
                kin = get_kinematics_at_frame(traj_df, frame_idx)
                features_list.append(kin)
            elif use_traj:
                features_list.append(np.zeros(6, dtype=np.float32)) # Assuming 6+8 dim? No, compute_kinematics returns 6. 
                # Wait, compute_kinematics_6d returns 6 dims (3 vel + 3 rot).
                # Let's check preprocess_v4.py again. It returns 6 dims.
                # But wait, config says kin_dim_raw: 6.
                # So 6 is correct.
                
            if features_list:
                feat_vec = np.hstack(features_list)
                terminal_features.append(feat_vec)
                terminal_labels.append(label_idx)

    if not terminal_features:
        print("No features collected.")
        return

    X = np.array(terminal_features)
    Y = np.array(terminal_labels)
    
    print(f"Collected {len(X)} terminal states.")
    print("Running t-SNE...")
    
    n_samples = len(X)
    perp = min(30, n_samples - 1) if n_samples > 1 else 1
    
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    print("Plotting...")
    
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(Y)
    
    for lbl_idx in unique_labels:
        mask = Y == lbl_idx
        lbl_name = labels_list[lbl_idx] if lbl_idx < len(labels_list) else f"Unknown ({lbl_idx})"
        c = colors_map[lbl_idx] if lbl_idx < len(colors_map) else (0,0,0)
        
        plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], label=lbl_name, color=c, alpha=0.7, edgecolors='w', s=60)
        
    plt.title(f"t-SNE of Terminal States (Transitions) - Mode: {data_mode}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_file = tsne_out
    plt.savefig(out_file, dpi=300)
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    main()
