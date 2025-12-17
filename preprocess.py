import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import yaml

from utils import FeatureExtractor, compute_kinematics, load_config

# --- LOAD CONFIG ---
cfg = load_config("config.yaml")

# --- CONFIG PARSING ---
BASE_DIR = cfg['data']['dataset_root']
DATA_MODE = cfg['data']['mode']
USE_TRAJ = cfg['data'].get('use_trajectory', True)
FILES = cfg['data']['files']

VID_NAME = FILES['raw_video']
TRAJ_NAME = FILES['trajectory']
LBL_NAME = FILES['labels']
TAC0_NAME = FILES['tactile_0']
TAC1_NAME = FILES['tactile_1']

LABELS_LIST = cfg['common']['labels']
DEVICE = cfg['common']['device'] if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = f"{cfg['data']['output_prefix']}_{DATA_MODE}"

def extract_video_features(extractor, vid_path, target_length):
    if not os.path.exists(vid_path):
        return np.zeros((target_length, 512), dtype=np.float32)

    cap = cv2.VideoCapture(vid_path)
    feats_list = []
    batch = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (224, 224))
        batch.append(frame)
        if len(batch) == 64:
            feats_list.append(extractor(batch).cpu().numpy())
            batch = []
    if batch: feats_list.append(extractor(batch).cpu().numpy())
    cap.release()
    
    if not feats_list: return np.zeros((target_length, 512), dtype=np.float32)
    matrix = np.vstack(feats_list)
    
    curr = len(matrix)
    if curr < target_length:
        diff = target_length - curr
        padding = np.repeat(matrix[-1:], diff, axis=0)
        matrix = np.vstack([matrix, padding])
    elif curr > target_length:
        matrix = matrix[:target_length]
        
    return matrix

def process_episode(extractor, ep_dir):
    # 1. Labels
    lbl_path = os.path.join(ep_dir, LBL_NAME)
    if not os.path.exists(lbl_path): return None
    df_lbl = pd.read_csv(lbl_path)
    labels = []
    for txt in df_lbl['label']:
        if txt in LABELS_LIST: labels.append(LABELS_LIST.index(txt))
        else: labels.append(-100)
    labels = np.array(labels)
    n_frames = len(labels)
    
    # 2. Kinematics
    kinematics = None
    if USE_TRAJ:
        traj_path = os.path.join(ep_dir, TRAJ_NAME)
        if not os.path.exists(traj_path): return None
        kinematics = compute_kinematics(pd.read_csv(traj_path))
        if len(kinematics) < n_frames:
            padding = np.repeat(kinematics[-1:], n_frames - len(kinematics), axis=0)
            kinematics = np.vstack([kinematics, padding])
        else:
            kinematics = kinematics[:n_frames]

    # 3. Dynamic Extraction
    features_to_stack = []
    
    if DATA_MODE == "rgb" or DATA_MODE == "all":
        main_vid = os.path.join(ep_dir, VID_NAME)
        features_to_stack.append(extract_video_features(extractor, main_vid, n_frames))

    if DATA_MODE == "tactile" or DATA_MODE == "all":
        tac0 = os.path.join(ep_dir, TAC0_NAME)
        tac1 = os.path.join(ep_dir, TAC1_NAME)
        features_to_stack.append(extract_video_features(extractor, tac0, n_frames))
        features_to_stack.append(extract_video_features(extractor, tac1, n_frames))

    if USE_TRAJ and kinematics is not None:
        features_to_stack.append(kinematics)
    
    return np.hstack(features_to_stack), labels

if __name__ == "__main__":
    import pandas as pd # Delayed import
    
    if not os.path.exists(BASE_DIR):
        print(f"Error: BASE_DIR not found: {BASE_DIR}")
        print("Please check config.yaml or use command line args.")
        exit(1)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- PREPROCESSING MODE: {DATA_MODE} ---")
    print(f"Output: {OUTPUT_DIR}")
    
    extractor = FeatureExtractor(DEVICE).to(DEVICE).eval()
    episodes = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
    
    count = 0
    for ep in tqdm(episodes):
        res = process_episode(extractor, os.path.join(BASE_DIR, ep))
        if res:
            X, Y = res
            np.save(os.path.join(OUTPUT_DIR, f"{ep}_X.npy"), X)
            np.save(os.path.join(OUTPUT_DIR, f"{ep}_Y.npy"), Y)
            count += 1
    print(f"Processed {count} episodes.")