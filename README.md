# Action Segmentation Pipeline

This repository implements a robust, end-to-end pipeline for **Action Segmentation** using multi-modal data (RGB, Tactile, and Robot Kinematics). It utilizes a **Multi-Stage Temporal Convolutional Network (MS-TCN)** to classify and segment complex manipulation tasks from continuous streams.

The pipeline integrates the entire workflow:
- **Data Utilities**: Manual labeling GUI and robust feature extraction.
- **Modeling**: MS-TCN training with multi-modal sensor fusion.
- **Analysis**: Advanced visualization tools including Saliency Maps, t-SNE embedding analysis, and segmentation gantt charts.

## Features

- **Multi-Modal Support**: Can train on RGB only, Tactile only, or combined modalities.
- **Robust Pipeline**: End-to-end support from raw data to visualized results.
- **Configurable**: usage of central `config.yaml` for all parameters and file paths.
- **Visualization**:
    - **Saliency Maps**: Visualize which parts of the image contribute to the model's decision (Grad-CAM based).
    - **t-SNE**: Visualize the embedding space, specifically focusing on terminal states of actions.
    - **Label Comparison**: Compare predicted labels across all demos in a single plot.

## 1. Installation

### Prerequisites
- Linux
- Python 3.9+
- CUDA-enabled GPU (recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd MS_TCN_action_segmentation
   ```

2. Create a conda environment and install dependencies:
   ```bash
   conda env create -f environment.yaml
   conda activate action_seg
   # Packages are already installed via environment.yaml
   ```
   Or manually:
   ```bash
   conda create -n action_seg python=3.9
   conda activate action_seg
   pip install -r requirements.txt
   ```

## 2. Data Structure

The pipeline expects data to be organized in a `data/` directory. Each "episode" or demonstration should have its own folder.

```
Action_Segmentation/
├── config.yaml
├── data/
│   ├── episode_1/
│   │   ├── raw_video.mp4           # Main RGB view
│   │   ├── tac_0.mp4               # Tactile view 0 (optional)
│   │   ├── tac_1.mp4               # Tactile view 1 (optional)
│   │   ├── camera_trajectory.csv   # Trajectory data (optional)
│   │   └── frame_labels.csv        # Ground truth labels (optional)
│   └── episode_2/
│       └── ...
```

You can customize the required filenames in `config.yaml` under `data.files`.

## 3. Configuration

All settings are managed in `config.yaml`.

- **Data Paths**: Set `dataset_root` (default `./data`) and filenames.
- **Mode**: Choose input modalities: `rgb`, `tactile`, or `all`.
- **Training**: Set `epochs`, `batch_size`, `learning_rate`.
    - **Validation**: Control split size via `val_size` (ratio `0.2` or absolute count `5`), enable `random_split`, and set `seed`.
- **Model**: Tune TCN parameters (`num_layers`, `num_f_maps`).

## 4. Usage Pipeline

### Step 1: Labeling (Optional)
If you need to create ground-truth labels for your data:
```bash
python label_actions.py
```
This launches a GUI to manually annotate video segments.

### Step 2: Preprocessing
Extract features from raw video/data. This is required before training.
```bash
python preprocess.py
```
This generates a `training_data_[mode]` directory with compressed `.npz` feature files.

### Step 3: Training
Train the action segmentation model.
```bash
python train.py
```
- The best model will be saved as `best_multimodal_v3_[mode].pth` (or similar prefix defined in config).
- Adjust `config.yaml` -> `training` -> `epochs` as needed.

### Step 4: Inference
Run the trained model on all episodes in your `data/` folder.
```bash
python inference.py
```
This will:
1. Load the trained model.
2. Predict action labels for every frame.
3. Save predictions to `pred_labels_[mode].csv` in each episode folder.
4. Generate a visualization video `viz_[episode_name]_[mode].mp4`.

### Step 5: Visualization & Analysis

#### Saliency Maps
Visualize model attention (Grad-CAM style):
```bash
python visualize_saliency.py
```
This will generate `saliency_[mode]_[episode].mp4` showing the heatmap overlay.

#### t-SNE Analysis
Analyze the latent space, specifically action transitions (terminal states):
```bash
python plot_tsne_terminal_states.py
```
Or use the interactive Plotly UI:
```bash
python plot_tsne_terminal_state_UI.py
```
This creates a `tsne_terminal_states.png` (or interactive plot) showing how different actions cluster in the feature space.

#### Label Comparison
To see a Gantt-chart style visualization of labels for all episodes in `data/`:
```bash
python visualize_all_labels.py
```
This produces `all_demos_labels_[mode].png`, useful for spotting inconsistencies or overall segmentation quality.

## 5. File Overview

- **`config.yaml`**: Central configuration file.
- **`model.py`**: Neural network definitions (KinematicEncoder, SingleStageTCN).
- **`utils.py`**: Shared utilities (Feature extraction, data loading).
- **`preprocess.py`**: Feature extraction script.
- **`train.py`**: Training loop.
- **`inference.py`**: Inference and video generation.
- **`visualize_saliency.py`**: Generates attention maps.
- **`visualize_all_labels.py`**: Visualizes all predicted labels side-by-side.
- **`plot_tsne_terminal_states.py`**: Generates static t-SNE plot.
- **`plot_tsne_terminal_state_UI.py`**: Interactive dashboard for t-SNE analysis.
- **`label_actions.py`**: Labeling GUI.

## 6. Model Architecture Explained

### Why is it "Temporal"?
Unlike standard image classifiers that predict action based on a single static frame (e.g., "this image contains a cup"), the **MS-TCN (Multi-Stage Temporal Convolutional Network)** makes predictions based on a **long history of frames**.

1.  **Input Sequence**: Instead of one image, the model takes a timeline of feature vectors representing the entire video (e.g., 500+ frames).
2.  **Dilated Convolutions**: The network uses layers with exponentially increasing dilation factors ($2^0, 2^1, \dots, 2^{10}$). 
    - Layer 1 looks at immediate neighbors ($t \pm 1$).
    - Layer 10 looks at frames hundreds of steps away ($t \pm 512$).
3.  **Receptive Field**: This architecture gives the model a massive "Receptive Field". When predicting the action for **Frame $t$**, the model effectively "sees" the context from seconds or minutes before and after that moment. 

This allows the model to:
- **Smooth Predictions**: Ignore short flickers or visual noise.
- **Learn Action Dependencies**: Understand that "Grasping" typically precedes "Lifting".