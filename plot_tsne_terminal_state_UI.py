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
import base64
import io
from PIL import Image
import dash
from dash import dcc, html, Input, Output, no_update
import plotly.express as px
import pickle

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
        # Match preprocess_v4.py: No RGB conversion, explicit resize before ToPILImage (if done outside)
        tensor = self.preprocess(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.backbone(tensor)
        return features.squeeze().cpu().numpy()

# --- KINEMATICS HELPER ---
def get_kinematics_at_frame(traj_df, frame_idx):
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
        if current_idx in target_indices:
            ret, frame = cap.read()
            if not ret: break
            frames[current_idx] = frame
        else:
            ret = cap.grab()
            if not ret: break
            
        if current_idx >= max_idx:
            break
            
        current_idx += 1
            
    cap.release()
    return frames

def numpy_to_b64(img_nyp):
    # img_np is BGR
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    b64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return "data:image/jpeg;base64," + b64_str

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
    cache_feat_path = files['cache_features']
    cache_meta_path = files['cache_meta']
    labels_list = cfg['common']['labels']
    colors_cfg = cfg['common']['colors']
    # Convert colors to hex for plotly
    colors_hex = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in colors_cfg]
    
    device = cfg['common']['device'] if torch.cuda.is_available() else "cpu"

    print(f"Mode: {data_mode}, Device: {device}")
    
    extractor = FeatureExtractor(device)
    
    terminal_features = []
    terminal_labels = []
    terminal_images = [] # RGB
    terminal_images_tac0 = [] # Tactile 0
    terminal_images_tac1 = [] # Tactile 1
    terminal_demo_ids = []
    terminal_frame_ids = []
    
    # --- CACHING LOGIC ---
    loaded_from_cache = False
    
    if os.path.exists(cache_feat_path) and os.path.exists(cache_meta_path):
        print("Loading cached features and metadata...")
        try:
            data = np.load(cache_feat_path)
            X = data['X']
            Y = data['Y']
            
            with open(cache_meta_path, 'rb') as f:
                meta = pickle.load(f)
                terminal_demo_ids = meta['demo_ids']
                terminal_frame_ids = meta['frame_ids']
                terminal_images = meta['images']
                terminal_images_tac0 = meta.get('images_tac0', []) # Handle backward compatibility
                terminal_images_tac1 = meta.get('images_tac1', [])
                
            if not terminal_images_tac0:
                raise ValueError("Cache missing tactile images. Re-running.")

            print(f"Loaded {len(X)} terminal states from cache.")
            loaded_from_cache = True
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-running extraction.")
            loaded_from_cache = False

    if not loaded_from_cache:
        # Reset lists
        terminal_features = []
        terminal_labels = []
        terminal_images = [] 
        terminal_images_tac0 = []
        terminal_images_tac1 = []
        terminal_demo_ids = []
        terminal_frame_ids = []

        episodes = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        print(f"Found {len(episodes)} episodes.")
        
        for ep in tqdm(episodes):
            ep_dir = os.path.join(base_dir, ep)
            lbl_path = os.path.join(ep_dir, f"{lbl_prefix}_{data_mode}.csv")
            
            if not os.path.exists(lbl_path):
                continue
                
            df_lbl = pd.read_csv(lbl_path)
            labels_indices = df_lbl['label_idx'].values
            
            if 'frame_idx' in df_lbl.columns:
                frame_mapping = df_lbl['frame_idx'].values
            else:
                frame_mapping = np.arange(len(labels_indices))

            transitions = np.where(labels_indices[:-1] != labels_indices[1:])[0]
            terminal_row_indices = np.append(transitions, len(labels_indices) - 1)
            terminal_frame_indices = frame_mapping[terminal_row_indices]
            
            traj_df = None
            if use_traj:
                traj_path = os.path.join(ep_dir, traj_name)
                if os.path.exists(traj_path):
                    traj_df = pd.read_csv(traj_path)
            
            vid_path = os.path.join(ep_dir, vid_name)
            tac0_path = os.path.join(ep_dir, tac0_name)
            tac1_path = os.path.join(ep_dir, tac1_name)
            
            # Always extract all frames for visualization
            rgb_frames = extract_frames_from_video(vid_path, terminal_frame_indices)
            tac0_frames = extract_frames_from_video(tac0_path, terminal_frame_indices)
            tac1_frames = extract_frames_from_video(tac1_path, terminal_frame_indices)
                
            for i, row_idx in enumerate(terminal_row_indices):
                frame_idx = terminal_frame_indices[i]
                label_idx = labels_indices[row_idx]
                
                features_list = []
                
                # 1. RGB Features
                rgb_frame = rgb_frames.get(frame_idx)
                if rgb_frame is None: continue # Skip if frame missing
                
                # Resize for feature extraction AND visualization
                rgb_frame_resized = cv2.resize(rgb_frame, (224, 224))
                
                if data_mode == "rgb" or data_mode == "all":
                    features_list.append(extractor(rgb_frame_resized))

                # 2. Tactile Features
                f0 = tac0_frames.get(frame_idx)
                f1 = tac1_frames.get(frame_idx)
                
                # Resize for viz
                f0_resized = cv2.resize(f0, (224, 224)) if f0 is not None else np.zeros((224, 224, 3), dtype=np.uint8)
                f1_resized = cv2.resize(f1, (224, 224)) if f1 is not None else np.zeros((224, 224, 3), dtype=np.uint8)

                if data_mode == "tactile" or data_mode == "all":
                    if f0 is not None: features_list.append(extractor(f0_resized))
                    else: features_list.append(np.zeros(512, dtype=np.float32))
                        
                    if f1 is not None: features_list.append(extractor(f1_resized))
                    else: features_list.append(np.zeros(512, dtype=np.float32))

                # 3. Kinematics
                if use_traj and traj_df is not None:
                    kin = get_kinematics_at_frame(traj_df, frame_idx)
                    features_list.append(kin)
                elif use_traj:
                    features_list.append(np.zeros(6, dtype=np.float32))
                    
                if features_list:
                    feat_vec = np.hstack(features_list)
                    terminal_features.append(feat_vec)
                    terminal_labels.append(label_idx)
                    terminal_demo_ids.append(ep)
                    terminal_frame_ids.append(frame_idx)
                    # Store images for viz
                    terminal_images.append(numpy_to_b64(rgb_frame_resized))
                    terminal_images_tac0.append(numpy_to_b64(f0_resized))
                    terminal_images_tac1.append(numpy_to_b64(f1_resized))

        if not terminal_features:
            print("No features collected.")
            return 

        X = np.array(terminal_features)
        Y = np.array(terminal_labels)
        
        print("Saving cache...")
        np.savez(cache_feat_path, X=X, Y=Y)
        with open(cache_meta_path, 'wb') as f:
            pickle.dump({
                'demo_ids': terminal_demo_ids,
                'frame_ids': terminal_frame_ids,
                'images': terminal_images,
                'images_tac0': terminal_images_tac0,
                'images_tac1': terminal_images_tac1,
                'labels_list': labels_list
            }, f)
    
    print(f"Collected {len(X)} terminal states.")
    print(f"Feature dimension: {X.shape[1]}")
    print("Running t-SNE...")
    
    n_samples = len(X)
    perp = min(30, n_samples - 1) if n_samples > 1 else 1
    
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    # --- DASH APP ---
    print("Launching Dash App...")
    
    # Prepare DataFrame for Plotly
    df_viz = pd.DataFrame({
        'x': X_embedded[:, 0],
        'y': X_embedded[:, 1],
        'label_idx': Y,
        'label_name': [labels_list[i] if i < len(labels_list) else str(i) for i in Y],
        'demo_id': terminal_demo_ids,
        'frame_idx': terminal_frame_ids,
        'image': terminal_images,
        'image_tac0': terminal_images_tac0,
        'image_tac1': terminal_images_tac1
    })
    df_viz['index'] = df_viz.index # Add explicit index for customdata
    
    # Map colors
    color_map = {name: colors_hex[i] if i < len(colors_hex) else "#000000" 
                 for i, name in enumerate(labels_list)}
    
    app = dash.Dash(__name__)
    
    fig = px.scatter(
        df_viz, x='x', y='y', color='label_name',
        hover_data=['demo_id', 'frame_idx'],
        custom_data=['index'], # Pass index to frontend
        color_discrete_map=color_map,
        title=f"t-SNE Terminal States (Mode: {data_mode})",
        template='plotly_dark', # Dark theme for plot
        width=1000, height=800
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='white')), selector=dict(mode='markers'))
    fig.update_layout(
        paper_bgcolor='#1e1e1e', # Dark background for plot area
        plot_bgcolor='#1e1e1e',
        font_color='white'
    )
    
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("Terminal State Analysis", style={'color': 'white', 'textAlign': 'center', 'padding': '20px', 'fontFamily': 'sans-serif'}),
        ], style={'backgroundColor': '#2c2c2c', 'marginBottom': '20px'}),

        html.Div([
            # Left Column: Plot
            html.Div([
                dcc.Graph(id='tsne-plot', figure=fig, clear_on_unhover=True),
            ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
            
            # Right Column: Details
            html.Div([
                html.Div([
                    html.H3("Selected State Details", style={'borderBottom': '1px solid #555', 'paddingBottom': '10px', 'marginBottom': '20px'}),
                    html.Div(id='info-container', style={'marginBottom': '20px', 'fontSize': '16px'}),
                    
                    html.Div([
                        html.H4("RGB View", style={'color': '#aaa', 'fontSize': '14px'}),
                        html.Img(id='img-rgb', style={'width': '100%', 'borderRadius': '8px', 'marginBottom': '15px', 'border': '1px solid #444'}),
                        
                        html.H4("Tactile 0", style={'color': '#aaa', 'fontSize': '14px'}),
                        html.Img(id='img-tac0', style={'width': '100%', 'borderRadius': '8px', 'marginBottom': '15px', 'border': '1px solid #444'}),
                        
                        html.H4("Tactile 1", style={'color': '#aaa', 'fontSize': '14px'}),
                        html.Img(id='img-tac1', style={'width': '100%', 'borderRadius': '8px', 'marginBottom': '15px', 'border': '1px solid #444'}),
                    ], id='images-wrapper', style={'display': 'none'}) # Hidden initially
                    
                ], style={'backgroundColor': '#2c2c2c', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.3)', 'color': 'white', 'fontFamily': 'sans-serif'})
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'height': '800px', 'overflowY': 'auto'})
        ], style={'display': 'flex', 'justifyContent': 'center'})
        
    ], style={'backgroundColor': '#121212', 'minHeight': '100vh', 'margin': '0'})
    
    @app.callback(
        [Output('img-rgb', 'src'),
         Output('img-tac0', 'src'),
         Output('img-tac1', 'src'),
         Output('info-container', 'children'),
         Output('images-wrapper', 'style')],
        [Input('tsne-plot', 'clickData')] # Changed to clickData
    )
    def display_click_data(clickData):
        if clickData is None:
            return "", "", "", "Click on a point to view details.", {'display': 'none'}
        
        pt = clickData['points'][0]
        idx = pt['customdata'][0] # Use customdata for correct index
        
        row = df_viz.iloc[idx]
        
        info_text = html.Div([
            html.P([html.Strong("Demo ID: "), row['demo_id']]),
            html.P([html.Strong("Frame: "), str(row['frame_idx'])]),
            html.P([html.Strong("Label: "), row['label_name']], style={'color': color_map.get(row['label_name'], 'white')})
        ])
        
        return row['image'], row['image_tac0'], row['image_tac1'], info_text, {'display': 'block'}

    app.run(debug=True, port=8050)

if __name__ == "__main__":
    main()
