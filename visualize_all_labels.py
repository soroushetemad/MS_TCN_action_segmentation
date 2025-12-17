import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob

from utils import load_config

# --- LOAD CONFIG ---
cfg = load_config()

BASE_DIR = cfg['data']['dataset_root']
DATA_MODE = cfg['data']['mode']
FILES = cfg['data']['files']
LBL_PREFIX = FILES['pred_labels_prefix']
OUT_PREFIX = FILES['all_labels_plot']

LABELS_LIST = cfg['common']['labels']
# Colors in config are often [R, G, B] (0-255). Matplotlib needs (0-1).
COLORS = [tuple(np.array(c)/255.0) for c in cfg['common']['colors']]

def get_segments(labels):
    """
    Convert a list of labels into segments (start, length, label_idx).
    """
    if not labels: return []
    
    segments = []
    curr_label = labels[0]
    start = 0
    
    for i, lbl in enumerate(labels):
        if lbl != curr_label:
            segments.append((start, i - start, curr_label))
            curr_label = lbl
            start = i
            
    # Add last segment
    segments.append((start, len(labels) - start, curr_label))
    return segments

def main():
    if not os.path.exists(BASE_DIR):
        print(f"Error: Base dir {BASE_DIR} not found.")
        return

    episodes = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
    print(f"Found {len(episodes)} episodes.")

    # Collect data
    demo_data = []
    
    for ep in episodes:
        ep_path = os.path.join(BASE_DIR, ep)
        
        # Priority: auto_labels_{DATA_MODE}.csv -> auto_labels.csv -> frame_labels.csv
        
        target_auto = f"{LBL_PREFIX}_{DATA_MODE}.csv"
        candidates = glob.glob(os.path.join(ep_path, target_auto))
        if not candidates:
            # Fallback to ground truth if no auto labels found
            candidates = [os.path.join(ep_path, FILES['labels'])]
            
        # Pick the best candidate (e.g., most recent or specific mode)
        # For now, let's pick the first one that exists
        csv_path = None
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break
        
        if not csv_path:
            print(f"Skipping {ep}: No labels found.")
            continue
            
        try:
            df = pd.read_csv(csv_path)
            # Map string labels to indices if needed
            if 'label' in df.columns and df['label'].dtype == object:
                # Ensure we handle unknown labels gracefully
                label_indices = []
                for l in df['label']:
                    if l in LABELS_LIST:
                        label_indices.append(LABELS_LIST.index(l))
                    else:
                        label_indices.append(-1) # Unlabeled/Unknown
            elif 'label_idx' in df.columns:
                label_indices = df['label_idx'].tolist()
            else:
                print(f"Skipping {ep}: CSV format unknown.")
                continue
                
            demo_data.append({
                "id": ep,
                "labels": label_indices,
                "source": os.path.basename(csv_path)
            })
            
        except Exception as e:
            print(f"Error reading {ep}: {e}")

    if not demo_data:
        print("No data to plot.")
        return

    # --- PLOTTING ---
    plt.figure(figsize=(15, len(demo_data) * 0.5 + 2))
    
    # Create legend handles
    legend_handles = []
    for i, (name, color) in enumerate(zip(LABELS_LIST, COLORS)):
        legend_handles.append(mpatches.Patch(color=color, label=f"{i}: {name}"))
    # Add Unlabeled/Background
    legend_handles.append(mpatches.Patch(color=(0.8, 0.8, 0.8), label="Unlabeled"))

    # Plot bars
    # Y-axis: Episodes (reversed so first is at top)
    y_positions = range(len(demo_data))
    
    for i, data in enumerate(demo_data):
        y = i
        segments = get_segments(data['labels'])
        
        # Broken Barh expects list of (start, width)
        # We need to group by color
        
        for start, length, lbl_idx in segments:
            if lbl_idx >= 0 and lbl_idx < len(COLORS):
                color = COLORS[lbl_idx]
            else:
                color = (0.8, 0.8, 0.8) # Grey for unknown
            
            plt.broken_barh([(start, length)], (y - 0.4, 0.8), facecolors=color, edgecolor='none')

    plt.yticks(y_positions, [d['id'] for d in demo_data])
    plt.xlabel("Frame Index")
    plt.title(f"Action Segmentation Labels per Demo ({len(demo_data)} Demos Total)")
    plt.grid(True, axis='x', alpha=0.3)
    
    # Legend outside
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    out_file = f"{OUT_PREFIX}_{DATA_MODE}.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved visualization to {out_file}")
    # plt.show() # Cannot show in headless env

if __name__ == "__main__":
    main()
