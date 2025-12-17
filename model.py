import torch
import torch.nn as nn
import yaml

# Load config once to set defaults?? 
# Better: Pass config params to classes.

class KinematicEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_dim) 
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, out_dim, 1),
            nn.GELU()
        )

    def forward(self, x):
        return self.mlp(self.norm(x))

class SingleStageTCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super().__init__()
        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_f_maps, num_f_maps, 3, dilation=2**i, padding=2**i),
                nn.ReLU(inplace=True), nn.Dropout(0.5)
            ) for i in range(num_layers)
        ])
        self.conv_1x1_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1_in(x)
        for layer in self.layers: out = out + layer(out)
        return self.conv_1x1_out(out)

class SegmentationModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        feat_dim = cfg['model']['feature_dim']
        data_mode = cfg['data']['mode']
        use_traj = cfg['data'].get('use_trajectory', True)
        
        # Calculate Dimensions
        if data_mode == "all": self.vis_dim = feat_dim * 3
        elif data_mode == "tactile": self.vis_dim = feat_dim * 2
        elif data_mode == "rgb": self.vis_dim = feat_dim
        else: raise ValueError(f"Invalid DATA_MODE in config: {data_mode}")

        self.kin_dim_raw = cfg['model']['kin_dim_raw'] if use_traj else 0
        self.kin_dim_embed = cfg['model']['kin_dim_embed'] if use_traj else 0
        
        self.use_traj = use_traj
        if use_traj:
            self.kin_encoder = KinematicEncoder(self.kin_dim_raw, 64, self.kin_dim_embed)
        
        total_dim = self.vis_dim + self.kin_dim_embed
        
        self.stage1 = SingleStageTCN(cfg['model']['num_layers'], cfg['model']['num_f_maps'], total_dim, cfg['model']['num_classes'])
        self.stage2 = SingleStageTCN(cfg['model']['num_layers'], cfg['model']['num_f_maps'], cfg['model']['num_classes'], cfg['model']['num_classes'])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Always slice from the end for kinematics
        if self.use_traj:
            k_dim = self.kin_dim_raw
            x_vis = x[:, :-k_dim, :]
            x_kin = x[:, -k_dim:, :]
            
            x_kin_emb = self.kin_encoder(x_kin)
            x_fused = torch.cat([x_vis, x_kin_emb], dim=1)
        else:
            x_fused = x
        
        out1 = self.stage1(x_fused)
        out2 = self.stage2(self.softmax(out1))
        return out1, out2
