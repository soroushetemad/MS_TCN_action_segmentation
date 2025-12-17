import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from model import SegmentationModel
from utils import FeatureExtractor, compute_kinematics, load_video_robust, load_config

# --- LOAD CONFIG ---
cfg = load_config("config.yaml")

DATA_ROOT = cfg['data']['dataset_root']
DATA_MODE = cfg['data']['mode']
USE_TRAJ = cfg['data'].get('use_trajectory', True)
VID_TYPE = cfg['data'].get('vid_type', 'raw_video.mp4') 
FILES = cfg['data']['files']
VID_NAME = FILES['raw_video']
TRAJ_NAME = FILES['trajectory']
LBL_NAME = FILES['labels']
TAC0_NAME = FILES['tactile_0']
TAC1_NAME = FILES['tactile_1']
AUTO_LABEL_PREFIX = FILES['pred_labels_prefix']
VIZ_PREFIX = FILES['viz_prefix']

DEVICE = cfg['common']['device']
MODEL_PATH = f"{cfg['training']['save_name_prefix']}_{DATA_MODE}.pth"
LABELS_LIST = cfg['common']['labels']
# Convert list of lists to list of tuples for OpenCV
COLORS = [tuple(c) for c in cfg['common']['colors']] 

def get_inference_data(extractor, ep_path):
    # 1. Kinematics
    kinematics = None
    n_frames = 0
    
    if USE_TRAJ:
        traj_p = os.path.join(ep_path, "camera_trajectory.csv")
        if not os.path.exists(traj_p): return None
        kinematics = compute_kinematics(pd.read_csv(traj_p))
        n_frames = len(kinematics)
    
    features_to_stack = []

    # 2. RGB
    if DATA_MODE == "rgb" or DATA_MODE == "all":
        main_p = os.path.join(ep_path, VID_TYPE)
        # Assuming load_video_robust returns frames, we need feature extraction
        # But wait, utils has load_video_robust which returns frames.
        traj_p = os.path.join(ep_path, TRAJ_NAME)
        if os.path.exists(traj_p):
            kinematics = compute_kinematics(pd.read_csv(traj_p))
    
    # 2. Extract Features
    feats = []
    
    if DATA_MODE in ["rgb", "all"]:
        main_p = os.path.join(ep_path, VID_NAME)
        frames = load_video_robust(main_p)
        if len(frames) > 0:
            # Batch extract
            curr = []
            tmp = []
            for f in frames:
                f = cv2.resize(f, (224, 224))
                curr.append(f)
                if len(curr) == 64:
                    tmp.append(extractor(curr).cpu().numpy())
                    curr = []
            if curr: tmp.append(extractor(curr).cpu().numpy())
            feats.append(np.vstack(tmp))
    
    if DATA_MODE in ["tactile", "all"]:
        t0_p = os.path.join(ep_path, TAC0_NAME)
        t1_p = os.path.join(ep_path, TAC1_NAME)
        # ... (Same logic for tactile)
        for p in [t0_p, t1_p]:
            frames = load_video_robust(p)
            if len(frames) > 0:
                curr = []
                tmp = []
                for f in frames:
                    f = cv2.resize(f, (224, 224))
                    curr.append(f)
                    if len(curr) == 64:
                        tmp.append(extractor(curr).cpu().numpy())
                        curr = []
                if curr: tmp.append(extractor(curr).cpu().numpy())
                feats.append(np.vstack(tmp))
            else:
                 # fallback if missing? usually assume valid data
                 pass

    # Pad kinematics to match visual frames if needed
    # (Assuming simple length match or padding as in preprocess)
    # Ideally should share exact same function as preprocess but inference might process live stream style.
    # For now, keeping logic similar to original inference but using extracted vars.
    
    if not feats: # No visual/tactile features extracted
        return None

    # Align lengths
    target_len = min([len(f) for f in feats])
    feats = [f[:target_len] for f in feats]
    
    if USE_TRAJ: # Kinematics handling
         if len(kinematics) > 0:
             if len(kinematics) < target_len:
                 # Pad kinematics
                 padding = np.repeat(kinematics[-1:], target_len - len(kinematics), axis=0)
                 kin_aligned = np.vstack([kinematics, padding])
             else:
                 # Crop kinematics
                 kin_aligned = kinematics[:target_len]
             feats.append(kin_aligned)
         else:
             # Fallback if trajectory enabled but missing file
             feats.append(np.zeros((target_len, 6), dtype=np.float32))

    return np.hstack(feats)

def generate_visualization(ep_path, predictions):
    # Load video for overlay
    cap = cv2.VideoCapture(os.path.join(ep_path, VID_NAME))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out_name = f"{VIZ_PREFIX}_{os.path.basename(ep_path)}_{DATA_MODE}.mp4"
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Color map
    colors = cfg['common']['colors']
    labels = cfg['common']['labels']
    
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if idx < len(predictions):
            pred_cls = predictions[idx]
            label_text = labels[pred_cls]
            color = colors[pred_cls]
            # Convert RGB to BGR
            c_bgr = (color[2], color[1], color[0])
            
            cv2.putText(frame, f"Pred: {label_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, c_bgr, 2)
            
        out.write(frame)
        idx += 1
        
    cap.release()
    out.release()

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} not found. Train first with mode={DATA_MODE}"); exit(1)

    if not os.path.exists(DATA_ROOT):
         print(f"Data root {DATA_ROOT} does not exist.")
         exit()

    print(f"--- INFERENCE MODE: {DATA_MODE} ---")
    
    # Load Model
    model = SegmentationModel(cfg).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    extractor = FeatureExtractor(DEVICE).to(DEVICE).eval()
    
    episodes = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    
    for ep in tqdm(episodes):
        try:
            ep_path = os.path.join(DATA_ROOT, ep)
            features = get_inference_data(extractor, ep_path)

            if features is None: continue
            
            x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(DEVICE)
            with torch.no_grad():
                out = model(x_tensor)
                # Model returns (out1, out2), we want out2
                preds = torch.argmax(out[1], dim=1).squeeze().cpu().numpy()
            
            # Save CSV
            pd.DataFrame({
                "frame_idx": np.arange(len(preds)),
                "label": [LABELS_LIST[i] for i in preds],
                "label_idx": preds
            }).to_csv(os.path.join(ep_path, f"{AUTO_LABEL_PREFIX}_{DATA_MODE}.csv"), index=False)
            
            # Save Video
            generate_visualization(ep_path, preds)
            
        except Exception as e:
            print(f"Error {ep}: {e}")