import os
import cv2
import numpy as np
import glob
import sys
import threading
import time
import queue
from datetime import datetime
import pandas as pd
from tkinter import filedialog
import tkinter as tk
import yaml
from scipy.spatial.transform import Rotation as R
import matplotlib.cm as cm

# --- LOAD CONFIG ---
def load_config(path="config.yaml"):
    with open(path, "r") as f: return yaml.safe_load(f)

# --- CONFIG PARSING ---
cfg = load_config()

# --- CONFIGURATION ---
BASE_DIR = cfg['data']['dataset_root']
FILES = cfg['data']['files']

VID_NAME = FILES['raw_video']
TRAJ_NAME = FILES['trajectory']
LBL_NAME = FILES['labels']
TAC0_NAME = FILES['tactile_0']
TAC1_NAME = FILES['tactile_1']

LABELS_LIST = cfg['common']['labels']

VIEW_W, VIEW_H = 960, 720 
TAC_W, TAC_H = 320, 240
N_FUTURE = 16

# Camera Intrinsics (Ideally move to config, but kept here for stability)
FX, FY = 642.9425, 641.5370
CX, CY = 641.8889, 398.0969
DIST_COEFFS = np.array([-0.0566365, 0.0680985, 0.0004173, 0.0008049, -0.0221657])
K_MATRIX = np.array([[FX, 0, CX],
                     [0, FY, CY],
                     [0, 0, 1]], dtype=np.float32)

SKIP_COMPLETED = False 

# Determine Target Folders
# Option 1: passed as args
# Option 2: scan base_dir
# Option 3: manual selection if neither
if len(sys.argv) > 1:
    TARGET_FOLDERS = sys.argv[1:]
else:
    if os.path.exists(BASE_DIR):
        TARGET_FOLDERS = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
        # Filter for standard date format if desired, or take all.
        # Taking all is safer for general use.
        # Exclude special folders?
        TARGET_FOLDERS = [d for d in TARGET_FOLDERS if not d.startswith('.')]
    else:
        TARGET_FOLDERS = []

print(f"Loaded {len(TARGET_FOLDERS)} folders from {BASE_DIR}")

COLORS = {
    "reach bottle cap": (0, 0, 255),  # Red
    "grasp bottle cap": (0, 255, 0),  # Green
    "reach bottle": (255, 0, 0),    # Blue
    "align bottle cap": (0, 255, 255), # Yellow
    "screw bottle cap": (255, 0, 255), # Magenta
    "final twist": (255, 255, 0),    # Cyan
    "release/retreat": (255, 255, 255) # White
} 
# Fallback colors for config labels if not in hardcoded map
CFG_COLORS = cfg['common']['colors']
if len(CFG_COLORS) == len(LABELS_LIST):
    for i, lbl in enumerate(LABELS_LIST):
        # Config colors are RGB list, OpenCV needs BGR tuple
        c = CFG_COLORS[i]
        COLORS[lbl] = (c[2], c[1], c[0]) 

# --- TRAJECTORY GENERATION FUNCTIONS ---

def generate_projected_video(exp_path):
    csv_path = os.path.join(exp_path, TRAJ_NAME)
    raw_vid_path = os.path.join(exp_path, VID_NAME)
    out_path = os.path.join(exp_path, "projected_trajectory.mp4")

    if not os.path.exists(csv_path) or not os.path.exists(raw_vid_path):
        # Silent fail is better here to allow the main loop to handle "corrupt" logic
        return False

    print(f"Generating projected video for: {os.path.basename(exp_path)}...")

    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(raw_vid_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps == 0 or width == 0:
        cap.release()
        return False

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    n_points = len(df)
    
    for i in range(n_points):
        ret, frame = cap.read()
        if not ret: break

        row = df.iloc[i]
        t = np.array([row["x"], row["y"], row["z"]], dtype=np.float32)
        quat = [row["q_x"], row["q_y"], row["q_z"], row["q_w"]]

        if np.linalg.norm(quat) < 1e-6:
            out.write(frame)
            continue

        R_wc = R.from_quat(quat).as_matrix()
        
        T_ref = np.eye(4)
        T_ref[:3, :3] = R_wc
        T_ref[:3, 3] = t
        T_ref_inv = np.linalg.inv(T_ref)

        rel_pos = []
        for j in range(i, i + min(N_FUTURE, n_points - i)):
            row_fut = df.iloc[j]
            t_fut = np.array([row_fut["x"], row_fut["y"], row_fut["z"]], dtype=np.float32)
            quat_fut = [row_fut["q_x"], row_fut["q_y"], row_fut["q_z"], row_fut["q_w"]]
            
            R_wc_fut = R.from_quat(quat_fut).as_matrix()
            T_fut = np.eye(4)
            T_fut[:3, :3] = R_wc_fut
            T_fut[:3, 3] = t_fut

            T_rel = T_ref_inv @ T_fut
            t_rel = T_rel[:3, 3]
            t_rel[2] = 0.18 
            rel_pos.append(t_rel)

        rel_pos = np.array(rel_pos)
        
        if len(rel_pos) > 0:
            pts_2d, _ = cv2.projectPoints(
                rel_pos, np.zeros((3, 1)), np.zeros((3, 1), dtype=np.float32),
                K_MATRIX, DIST_COEFFS
            )
            pts_2d = np.squeeze(pts_2d)

            if pts_2d.ndim == 1:
                pts_2d = np.expand_dims(pts_2d, axis=0)

            colors = cm.jet(np.linspace(0, 1, pts_2d.shape[0]))[:, :3] * 255

            for j, (u, v) in enumerate(pts_2d.astype(int)):
                if 0 <= u < width and 0 <= v < height:
                    color_idx = min(j, len(colors) - 1)
                    color = tuple(int(c) for c in colors[color_idx])
                    cv2.circle(frame, (u, v), 5, color, -1)

        out.write(frame)

    cap.release()
    out.release()
    print("Generation complete.")
    return True

# --- MAIN LABELER CLASSES ---

class EpisodeData:
    def __init__(self, exp_id):
        self.exp_id = exp_id
        self.main_frames = None 
        self.tac1_frames = None
        self.tac2_frames = None
        self.labels = []
        self.is_ready = False
        self.load_error = False
        self.undo_stack = []
        self.redo_stack = []

    def push_state(self, current_pending_idx):
        self.undo_stack.append((list(self.labels), current_pending_idx)) 
        if len(self.undo_stack) > 50: self.undo_stack.pop(0)
        self.redo_stack.clear() 

    def undo(self, current_pending_idx_from_ui):
        if not self.undo_stack: return None
        self.redo_stack.append((list(self.labels), current_pending_idx_from_ui))
        prev_labels, prev_pending_idx = self.undo_stack.pop()
        self.labels = prev_labels
        return prev_pending_idx

    def redo(self, current_pending_idx_from_ui):
        if not self.redo_stack: return None
        self.undo_stack.append((list(self.labels), current_pending_idx_from_ui))
        next_labels, next_pending_idx = self.redo_stack.pop()
        self.labels = next_labels
        return next_pending_idx

def load_video_frames_fast(path, resize_dim=None):
    if not os.path.exists(path): return np.array([], dtype=np.uint8)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return np.array([], dtype=np.uint8)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = resize_dim if resize_dim else (int(cap.get(3)), int(cap.get(4)))
    if n_frames <= 0: return load_video_frames_slow(path, resize_dim)
    buffer = np.empty((n_frames, h, w, 3), dtype=np.uint8)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx >= n_frames: break 
        if resize_dim: buffer[idx] = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_LINEAR)
        else: buffer[idx] = frame
        idx += 1
    cap.release()
    if idx < n_frames: return buffer[:idx]
    return buffer

def load_video_frames_slow(path, resize_dim=None):
    frames = []
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret: break
        if resize_dim: frame = cv2.resize(frame, resize_dim)
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.uint8)

class AsyncLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.next_data = None
        self.thread = None
        self.lock = threading.Lock()

    def start_loading(self, exp_id):
        if self.thread and self.thread.is_alive():
            self.thread.join()
        self.next_data = EpisodeData(exp_id)
        self.thread = threading.Thread(target=self._worker, args=(exp_id,))
        self.thread.daemon = True
        self.thread.start()

    def _worker(self, exp_id):
        try:
            exp_path = os.path.join(self.base_dir, exp_id)
            vid_path = os.path.join(exp_path, "projected_trajectory.mp4")
            
            if not os.path.exists(vid_path):
                success = generate_projected_video(exp_path)
                if not success:
                    vid_path = os.path.join(exp_path, VID_NAME)
            
            if not os.path.exists(vid_path): 
                # Be robust, don't crash thread, just mark error
                self.next_data.load_error = True
                self.next_data.is_ready = True
                return
                
            self.next_data.main_frames = load_video_frames_fast(vid_path, (VIEW_W, VIEW_H))
            self.next_data.tac1_frames = load_video_frames_fast(os.path.join(exp_path, TAC0_NAME), (TAC_W, TAC_H))
            self.next_data.tac2_frames = load_video_frames_fast(os.path.join(exp_path, TAC1_NAME), (TAC_W, TAC_H))
            
            n_frames = len(self.next_data.main_frames)
            self.next_data.labels = [-1] * n_frames
            lbl_path = os.path.join(exp_path, LBL_NAME)
            if os.path.exists(lbl_path):
                df = pd.read_csv(lbl_path)
                loaded_lbls = df['label'].tolist()
                limit = min(n_frames, len(loaded_lbls))
                for i in range(limit):
                    txt = loaded_lbls[i]
                    if txt in LABELS_LIST:
                        self.next_data.labels[i] = LABELS_LIST.index(txt)
            
            with self.lock:
                self.next_data.is_ready = True
                
        except Exception as e:
            print(f"Error loading {exp_id}: {e}")
            self.next_data.load_error = True
            self.next_data.main_frames = np.array([], dtype=np.uint8)

    def get_data(self):
        if self.thread is None: return None
        if not self.next_data.is_ready:
            print(f"Waiting for load ({self.next_data.exp_id})...")
            self.thread.join()
        return self.next_data

class Labeler:
    def __init__(self, base_dir, episodes):
        self.base_dir = base_dir
        self.episodes = episodes
        self.loader = AsyncLoader(base_dir)
        self.ep_idx = 0
        self.finished = False
        
        print(f"--- Starting Session: {len(episodes)} episodes to do ---")
        
        self.data = None
        # Robust Startup: find first good episode
        while self.ep_idx < len(self.episodes):
            print(f"Loading episode {self.ep_idx + 1}/{len(self.episodes)}: {self.episodes[self.ep_idx]}...")
            self.loader.start_loading(self.episodes[self.ep_idx])
            self.data = self.loader.get_data()
            
            if self.data.load_error or self.data.main_frames is None or len(self.data.main_frames) == 0:
                print(f"Skipping corrupted/empty episode: {self.episodes[self.ep_idx]}")
                self.ep_idx += 1
            else:
                break

        if self.ep_idx >= len(self.episodes):
            print("No valid episodes found. Exiting.")
            self.finished = True
            return

        # Pre-load next
        if self.ep_idx + 1 < len(self.episodes):
            self.loader.start_loading(self.episodes[self.ep_idx + 1])

        self.current_idx = 0
        self.playback_speed = 1.0
        self.is_playing = False
        self.pending_label_idx = 0
        self.show_help = False
        self.msg = "" 
        self.msg_timer = 0
        
        self.init_episode_state()

    def init_episode_state(self):
        self.current_idx = 0
        self.pending_label_idx = 0
        
        if self.data is None or self.data.main_frames is None or len(self.data.main_frames) == 0:
            return

        n = len(self.data.main_frames)
        labeled_indices = [i for i, x in enumerate(self.data.labels) if x != -1]
        if labeled_indices:
            last = labeled_indices[-1]
            self.current_idx = min(last + 1, n - 1)
            last_lbl = self.data.labels[last]
            if last_lbl != -1:
                self.pending_label_idx = min(last_lbl + 1, len(LABELS_LIST)-1)

    def save_current(self):
        if self.data is None or self.data.load_error: return
        exp_path = os.path.join(self.base_dir, self.data.exp_id)
        lbl_path = os.path.join(exp_path, LBL_NAME)
        str_labels = [LABELS_LIST[x] if x != -1 else "unlabeled" for x in self.data.labels]
        df = pd.DataFrame({"frame_idx": np.arange(len(str_labels)), "label": str_labels})
        df.to_csv(lbl_path, index=False)
        print(f"Saved {self.data.exp_id}")

    def next_episode(self):
        self.save_current()
        self.load_next_valid()

    def load_next_valid(self):
        while True:
            self.ep_idx += 1
            if self.ep_idx >= len(self.episodes):
                print("All episodes processed!")
                self.finished = True
                return

            self.data = self.loader.get_data() 
            
            if self.ep_idx + 1 < len(self.episodes):
                self.loader.start_loading(self.episodes[self.ep_idx + 1])
            
            if self.data.load_error or self.data.main_frames is None or len(self.data.main_frames) == 0:
                 print(f"Skipping corrupted episode: {self.data.exp_id}")
                 continue 
            else:
                 break 

        self.init_episode_state()
        cv2.setTrackbarPos("Scrub", "Labeler", 0)
        cv2.setTrackbarMax("Scrub", "Labeler", len(self.data.main_frames)-1)

    def reject_current_episode(self):
        if not self.data or not self.data.exp_id: return

        exp_id = self.data.exp_id
        src_path = os.path.join(self.base_dir, exp_id)
        rej_dir = os.path.join(self.base_dir, "rejected")
        dst_path = os.path.join(rej_dir, exp_id)

        print(f"Rejecting: {exp_id} -> {dst_path}")

        # Ensure rejected folder exists
        if not os.path.exists(rej_dir):
            os.makedirs(rej_dir)

        try:
            # Move the directory
            shutil.move(src_path, dst_path)
            self.set_msg("REJECTED")
            # We do NOT save the CSV, we just move it.
            # Directly load next without saving
            self.load_next_valid()
        except Exception as e:
            print(f"Error moving folder: {e}")
            self.set_msg("MOVE FAILED")

    def find_segment_start(self, idx):
        if self.data.labels[idx] == -1:
            for i in range(idx, -1, -1):
                if self.data.labels[i] != -1: return i+1
            return 0
        curr = self.data.labels[idx]
        for i in range(idx, -1, -1):
            if self.data.labels[i] != curr: return i+1
        return 0

    def apply_label(self):
        self.data.push_state(self.pending_label_idx)
        start_seg = self.find_segment_start(self.current_idx)
        label_val = self.pending_label_idx
        
        for i in range(start_seg, self.current_idx + 1):
            self.data.labels[i] = label_val
            
        if self.current_idx + 1 < len(self.data.main_frames):
            for k in range(self.current_idx + 1, len(self.data.main_frames)):
                if self.data.labels[k] == label_val or self.data.labels[k] == -1: 
                    self.data.labels[k] = -1 
                else: break
        
        self.pending_label_idx = min(self.pending_label_idx + 1, len(LABELS_LIST)-1)
        self.save_current()

    def set_msg(self, text):
        self.msg = text
        self.msg_timer = 30 

    def render(self):
        if self.data.main_frames is None or self.current_idx >= len(self.data.main_frames): 
            return np.zeros((720,1280,3), np.uint8)
            
        frame = self.data.main_frames[self.current_idx].copy()
        
        t1 = self.data.tac1_frames[self.current_idx] if self.current_idx < len(self.data.tac1_frames) else np.zeros((TAC_H, TAC_W, 3), np.uint8)
        t2 = self.data.tac2_frames[self.current_idx] if self.current_idx < len(self.data.tac2_frames) else np.zeros((TAC_H, TAC_W, 3), np.uint8)

        # UI Overlay
        cv2.rectangle(frame, (0, 0), (VIEW_W, 100), (30, 30, 30), -1)
        existing = self.data.labels[self.current_idx]
        exist_txt = LABELS_LIST[existing] if existing != -1 else "Unlabeled"
        target_txt = LABELS_LIST[self.pending_label_idx]
        target_col = COLORS[self.pending_label_idx]
        
        cv2.putText(frame, f"Ep: {self.data.exp_id} ({self.ep_idx+1}/{len(self.episodes)})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f"Existing: {exist_txt}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(frame, f"APPLY: {target_txt}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, target_col, 2)
        
        if self.msg_timer > 0:
            cv2.putText(frame, self.msg, (VIEW_W//2 - 100, VIEW_H//2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
            self.msg_timer -= 1

        if self.show_help:
             # UPDATED HELP TEXT
             lines = ["[Space] Play/Pause", "[Tab] SlowMo", "[Enter] Apply", "[u] UNDO", "[y] REDO", "[r] REJECT", "[N] Next Ep"]
             y = 130
             cv2.rectangle(frame, (50, 110), (400, 380), (0,0,0), -1)
             for l in lines:
                 cv2.putText(frame, l, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                 y += 30
        else:
            cv2.putText(frame, "[H] Help", (VIEW_W-100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)

        # Timeline
        h, w = VIEW_H, VIEW_W
        y_start = h - 30
        cv2.rectangle(frame, (0, y_start), (w, h), (50, 50, 50), -1)
        scale = w / len(self.data.main_frames)
        curr_c, start = -2, 0
        for i, idx in enumerate(self.data.labels):
            if idx != curr_c:
                if curr_c != -2:
                    c = COLORS[curr_c] if curr_c != -1 else (100, 100, 100)
                    cv2.rectangle(frame, (int(start*scale), y_start), (int(i*scale), h), c, -1)
                curr_c, start = idx, i
        c = COLORS[curr_c] if curr_c != -1 else (100, 100, 100)
        cv2.rectangle(frame, (int(start*scale), y_start), (int(len(self.data.main_frames)*scale), h), c, -1)
        cx = int(self.current_idx * scale)
        cv2.line(frame, (cx, y_start), (cx, h), (255, 255, 255), 2)

        final_h, final_w = max(VIEW_H, TAC_H*2), VIEW_W + TAC_W
        canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        canvas[:VIEW_H, :VIEW_W] = frame
        canvas[:TAC_H, VIEW_W:] = t1
        canvas[TAC_H:TAC_H*2, VIEW_W:] = t2
        return canvas

    def run(self):
        if self.finished: return

        if self.data.main_frames is None or len(self.data.main_frames) == 0:
            print("CRITICAL: Failed to load any valid episodes.")
            return

        cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Labeler", 1280, 720)
        def on_track(val): self.current_idx = val
        cv2.createTrackbar("Scrub", "Labeler", 0, len(self.data.main_frames)-1, on_track)

        while not self.finished:
            if self.data.main_frames is None: 
                self.next_episode()
                continue

            if self.is_playing and self.current_idx < len(self.data.main_frames)-1:
                self.current_idx += (1 if self.playback_speed >= 1.0 else 0)
                cv2.setTrackbarPos("Scrub", "Labeler", self.current_idx)
            elif self.is_playing: self.is_playing = False
            
            cv2.imshow("Labeler", self.render())
            wait = 33 if self.playback_speed >= 1.0 else 100
            k = cv2.waitKey(wait) & 0xFF
            
            if k == 27: self.save_current(); break
            elif k == ord('n'): self.next_episode()
            elif k == ord(' '): self.is_playing = not self.is_playing
            elif k == 9: self.playback_speed = 0.2 if self.playback_speed == 1.0 else 1.0
            elif k == ord('h'): self.show_help = not self.show_help
            
            # --- UPDATED KEY BINDINGS ---
            elif k == ord('r'): # REJECT
                self.reject_current_episode()
            elif k == ord('y'): # REDO (Moved from r)
                restored_tool = self.data.redo(self.pending_label_idx)
                if restored_tool is not None:
                    self.pending_label_idx = restored_tool
                    self.set_msg("REDO")
                else: self.set_msg("Can't Redo")

            elif k == ord('d'): self.current_idx = min(self.current_idx+1, len(self.data.main_frames)-1); self.is_playing = False; cv2.setTrackbarPos("Scrub", "Labeler", self.current_idx)
            elif k == ord('a'): self.current_idx = max(self.current_idx-1, 0); self.is_playing = False; cv2.setTrackbarPos("Scrub", "Labeler", self.current_idx)
            elif k == 82: self.pending_label_idx = max(self.pending_label_idx - 1, 0)
            elif k == 84: self.pending_label_idx = min(self.pending_label_idx + 1, len(LABELS_LIST)-1)
            
            elif k == ord('u'): 
                restored_tool = self.data.undo(self.pending_label_idx)
                if restored_tool is not None:
                    self.pending_label_idx = restored_tool
                    self.set_msg("UNDO")
                else: self.set_msg("Can't Undo")

            elif k == 13: self.apply_label()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- MAIN EXECUTION ---
    # BASE_DIR is already set from config
    
    if not os.path.exists(BASE_DIR):
        print(f"Error: Path {BASE_DIR} does not exist.")
    else: 
        # --- SKIP LOGIC ---
        final_list = []
        if SKIP_COMPLETED:
            for ep in TARGET_FOLDERS:
                label_file = os.path.join(BASE_DIR, ep, LBL_NAME)
                if not os.path.exists(label_file):
                    final_list.append(ep)
            print(f"Skipped {len(TARGET_FOLDERS) - len(final_list)} completed episodes.")
        else:
            final_list = TARGET_FOLDERS

        if not final_list:
            print("No episodes left to label! Good job.")
        else:
            Labeler(BASE_DIR, final_list).run()
