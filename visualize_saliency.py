import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml

from model import SegmentationModel
from utils import compute_kinematics, load_video_robust, load_config

# --- LOAD CONFIG ---
cfg = load_config("config.yaml")

# USER SETTING: Which episode to visualize?
# Automatically pick the first one from data dir if available, otherwise manual override needed.
if os.path.exists(cfg['data']['dataset_root']):
    candidates = sorted([d for d in os.listdir(cfg['data']['dataset_root']) if os.path.isdir(os.path.join(cfg['data']['dataset_root'], d))])
    EPISODE_NAME = candidates[0] if candidates else "demo_placeholder"
else:
    EPISODE_NAME = "demo_placeholder" 

# Config-derived paths
BASE_DIR = cfg['data']['dataset_root']
DATA_MODE = cfg['data']['mode']
USE_TRAJ = cfg['data'].get('use_trajectory', True)
FILES = cfg['data']['files']
VID_NAME = FILES['raw_video']
TRAJ_NAME = FILES['trajectory']
TAC0_NAME = FILES['tactile_0']
TAC1_NAME = FILES['tactile_1']
SALIENCY_PREFIX = FILES['saliency_prefix']
MODEL_PATH = f"{cfg['training']['save_name_prefix']}_{DATA_MODE}.pth"
LABELS_LIST = cfg['common']['labels']
DEVICE = cfg['common']['device']

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        return self.backbone(x).squeeze()

# --- UTILS ---
def denoise_heatmap(grad):
    if grad is None: return None
    heatmap = torch.abs(grad).sum(dim=1).squeeze().detach().cpu().numpy()
    max_val = heatmap.max()
    if max_val > 0: heatmap /= max_val
    heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)
    heatmap[heatmap < 0.2] = 0 # Threshold noise
    return heatmap

def apply_viz(heatmap, img, global_max=None):
    if heatmap is None: return img
    if global_max:
        heatmap = heatmap / (global_max + 1e-8)
        heatmap = np.clip(heatmap, 0, 1)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colored_map = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(colored_map, 0.6, img, 0.4, 0)

def draw_live_plot(canvas, history_rgb, history_t0, history_t1, history_traj, current_frame_idx, w, h):
    plot_area = canvas[448:h, 0:w]
    plot_area[:] = (20, 20, 20)
    ph, pw = plot_area.shape[:2]
    
    all_vals = history_rgb + history_t0 + history_t1 + history_traj
    if not all_vals: max_val = 1.0
    else: max_val = max(all_vals) * 1.1 + 1e-6
    
    def to_pt(idx, val):
        x = int((idx / max(1, len(history_rgb))) * pw)
        y = int(ph - (val / max_val) * (ph - 10)) - 5
        return (x, y)
# -----------------------------------------------------------------------------
# 2. HELPER: Overlay Heatmap on Image
# -----------------------------------------------------------------------------
def apply_viz(image, heatmap, alpha=0.5):
    # image: H, W, 3 (BGR)
    # heatmap: H, W (0..1)
    
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay

# -----------------------------------------------------------------------------
# 3. HELPER: Draw Live Plot
# -----------------------------------------------------------------------------
def draw_live_plot(saliency_history, labels, frame_idx, colors):
    # saliency_history: (N, C) (N frames so far)
    # We draw a line chart for each class
    
    H, W = 300, 600
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    
    if len(saliency_history) < 2: return canvas
    
    # Normalize
    data = np.array(saliency_history)
    max_val = data.max() if data.max() > 0 else 1.0
    
    # Draw logic
    for c_idx in range(data.shape[1]):
        color = colors[c_idx] 
        # Convert RGB to BGR
        c_bgr = (color[2], color[1], color[0])
        
        pts = []
        for i, val in enumerate(data[:, c_idx]):
            x = int((i / max(1, len(data))) * W)
            y = int(H - (val / max_val) * (H-20) - 10)
            pts.append((x, y))
            
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts)], False, c_bgr, 2)
            
    return canvas

# --- MAIN ---
def main():
    if not os.path.exists(os.path.join(BASE_DIR, EPISODE_NAME)):
        print(f"Error: Episode {EPISODE_NAME} not found in {BASE_DIR}")
        return

    print("Loading Model...")
    model = SegmentationModel(cfg).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Saliency needs gradient, so we might need train mode or just require grad enabled
    
    # Initialize Feature Extractor (Backbone)
    resnet = ResNetBackbone().to(DEVICE)
    resnet.eval()
    
    # Load Data
    ep_path = os.path.join(BASE_DIR, EPISODE_NAME)
    print(f"Processing {ep_path}...")
    # 2. LOAD DATA
    # We load RGB even in tactile mode just for visualization background
    main_p = os.path.join(ep_path, VID_NAME)
    if not os.path.exists(main_p): main_p = os.path.join(ep_path, "segmented_raw_video.mp4")
    
    # 1. Kinematics
    kinematics = []
    traj_p = os.path.join(ep_path, TRAJ_NAME)
    if os.path.exists(traj_p):
        df = pd.read_csv(traj_p)
        kinematics = compute_kinematics(df)
        
    # 2. Video Frames (RGB)
    frames_main = load_video_robust(os.path.join(ep_path, VID_NAME))
    
    # 3. Video Frames (Tactile)
    frames_tac0 = load_video_robust(os.path.join(ep_path, TAC0_NAME))
    frames_tac1 = load_video_robust(os.path.join(ep_path, TAC1_NAME))
    
    # Pad Tactile to match RGB length
    def pad_f(l, t):
        if len(l) == 0: return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(t)]
        if len(l) < t: l.extend([l[-1]] * (t - len(l)))
        return l[:t]
    
    target_len = len(frames_main)
    frames_tac0 = pad_f(frames_tac0, target_len)
    frames_tac1 = pad_f(frames_tac1, target_len)
    
    # Kinematics
    kin_tensor = None
    if USE_TRAJ:
        traj_p = os.path.join(ep_path, "camera_trajectory.csv")
        if os.path.exists(traj_p): kin_np = compute_kinematics(pd.read_csv(traj_p))
        else: kin_np = np.zeros((target_len, 6), dtype=np.float32)
        
        if len(kin_np) < target_len:
            kin_np = np.vstack([kin_np, np.repeat(kin_np[-1:], target_len-len(kin_np), axis=0)])
        kin_tensor = torch.tensor(kin_np[:target_len].T).unsqueeze(0).to(DEVICE)
        kin_tensor.requires_grad_(True)

    # 3. PRE-COMPUTE FROZEN FEATURES
    preprocess = transforms.Compose([
        transforms.ToPILImage(), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"Pre-computing features for Mode: {DATA_MODE}...")
    feats_list = []
    tensor_main, tensor_t0, tensor_t1 = [], [], []

    for i in tqdm(range(target_len)):
        # Prepare inputs based on mode
        f_vecs = []
        
        # RGB
        tm = preprocess(frames_main[i]).to(DEVICE)
        tensor_main.append(tm)
        if DATA_MODE in ["rgb", "all"]:
            with torch.no_grad(): f_vecs.append(resnet(tm.unsqueeze(0)))

        # Tactile
        t0 = preprocess(frames_tac0[i]).to(DEVICE)
        t1 = preprocess(frames_tac1[i]).to(DEVICE)
        tensor_t0.append(t0); tensor_t1.append(t1)
        
        if DATA_MODE in ["tactile", "all"]:
            with torch.no_grad():
                f_vecs.append(resnet(t0.unsqueeze(0)))
                f_vecs.append(resnet(t1.unsqueeze(0)))

        feats_list.append(torch.cat(f_vecs))

    full_seq_frozen = torch.stack(feats_list).T.unsqueeze(0) # (1, Channels, T)

    # 4. SALIENCY LOOP
    out_w, out_h = 672, 600
    out = cv2.VideoWriter(f"{SALIENCY_PREFIX}_{DATA_MODE}_{EPISODE_NAME}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (out_w, out_h))
    
    history_rgb, history_t0, history_t1, history_traj = [], [], [], []
    
    print("Generating Saliency Maps...")
    for i in tqdm(range(target_len)):
        # Calculate gradients specifically for current frame
        active_feats = []
        
        # RGB Handling
        hm_m = None
        if DATA_MODE in ["rgb", "all"]:
            curr_m = tensor_main[i].unsqueeze(0).clone().detach().requires_grad_(True)
            active_feats.append(resnet(curr_m))
            
        # Tactile Handling
        hm_t0, hm_t1 = None, None
        if DATA_MODE in ["tactile", "all"]:
            curr_t0 = tensor_t0[i].unsqueeze(0).clone().detach().requires_grad_(True)
            curr_t1 = tensor_t1[i].unsqueeze(0).clone().detach().requires_grad_(True)
            active_feats.append(resnet(curr_t0))
            active_feats.append(resnet(curr_t1))
            
        # Splicing: Create current frame vector
        curr_feat_vec = torch.cat(active_feats).unsqueeze(0).unsqueeze(2) 

        # Splicing: Inject into sequence
        seq_input = full_seq_frozen.clone().detach()
        seq_input[:, :, i] = curr_feat_vec.squeeze()
        
        # Forward Pass
        if USE_TRAJ and kin_tensor is not None:
            full_input = torch.cat([seq_input, kin_tensor], dim=1)
        else:
            full_input = seq_input
            
        output = model(full_input)
        
        # Backward Pass
        logits = output[0, :, i]
        pred_idx = torch.argmax(logits)
        score = logits[pred_idx]
        
        model.zero_grad()
        resnet.zero_grad()
        if kin_tensor is not None and kin_tensor.grad is not None:
            kin_tensor.grad.zero_()
            
        score.backward()
        
        # Extract Gradients
        if DATA_MODE in ["rgb", "all"]:
            hm_m = denoise_heatmap(curr_m.grad)
            history_rgb.append(np.sum(hm_m) if hm_m is not None else 0)
        else:
            history_rgb.append(0) 

        if DATA_MODE in ["tactile", "all"]:
            hm_t0 = denoise_heatmap(curr_t0.grad)
            hm_t1 = denoise_heatmap(curr_t1.grad)
            history_t0.append(np.sum(hm_t0) if hm_t0 is not None else 0)
            history_t1.append(np.sum(hm_t1) if hm_t1 is not None else 0)
        else:
            history_t0.append(0)
            history_t1.append(0)
            
        if USE_TRAJ and kin_tensor is not None and kin_tensor.grad is not None:
            # Sum absolute gradients for the current timestep across all kinematic channels
            g_val = kin_tensor.grad[0, :, i].abs().sum().item()
            history_traj.append(g_val)
        else:
            history_traj.append(0)

        # Visualize
        vals = [v for v in [hm_m, hm_t0, hm_t1] if v is not None]
        g_max = max([v.max() for v in vals]) if vals else 1.0
        
        viz_m = apply_viz(hm_m, frames_main[i], g_max)
        viz_t0 = apply_viz(hm_t0, frames_tac0[i], g_max)
        viz_t1 = apply_viz(hm_t1, frames_tac1[i], g_max)
        
        # Layout
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        main_big = cv2.resize(viz_m, (448, 448))
        canvas[0:448, 0:448] = main_big
        canvas[0:224, 448:672] = viz_t0
        canvas[224:448, 448:672] = viz_t1
        
        canvas = draw_live_plot(canvas, history_rgb, history_t0, history_t1, history_traj, i, out_w, out_h)
        
        label_text = LABELS_LIST[pred_idx.item()]
        cv2.putText(canvas, f"Mode: {DATA_MODE}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(canvas, f"Act: {label_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(canvas)

    out.release()
    print(f"Saved {SALIENCY_PREFIX}_{DATA_MODE}_{EPISODE_NAME}.mp4")

if __name__ == "__main__":
    main()