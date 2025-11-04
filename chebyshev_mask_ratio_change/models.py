#!/usr/bin/env python3
"""
共通モデル定義モジュール
学習スクリプトと分析スクリプトで共有されるネットワーク定義
"""

import torch
import torch.nn as nn
import numpy as np

class BranchNetwork(nn.Module):
    """Branch Network for DeepONet"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh', dropout=0.1):
        super(BranchNetwork, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 最後の層以外
                if activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                
                # Dropoutを追加（学習時のみ有効）
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """重み初期化"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

class TrunkNetwork(nn.Module):
    """Trunk Network for DeepONet"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh', dropout=0.1):
        super(TrunkNetwork, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 最後の層以外
                if activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                
                # Dropoutを追加（学習時のみ有効）
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """重み初期化"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

class DeepONet(nn.Module):
    """Deep Operator Network"""
    
    def __init__(self, branch_input_dim, trunk_input_dim, 
                 branch_hidden_dims, trunk_hidden_dims, 
                 latent_dim, activation='tanh', dropout=0.1):
        super(DeepONet, self).__init__()
        
        self.branch_net = BranchNetwork(
            branch_input_dim, branch_hidden_dims, latent_dim, 
            activation, dropout
        )
        
        self.trunk_net = TrunkNetwork(
            trunk_input_dim, trunk_hidden_dims, latent_dim, 
            activation, dropout
        )
        
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        output = torch.sum(branch_output * trunk_output, dim=1, keepdim=True) + self.bias
        return output

class DeepONetDataset(torch.utils.data.Dataset):
    """DeepONet用のデータセット"""
    
    def __init__(self, branch_data, trunk_data, target_data):
        self.branch_data = torch.FloatTensor(branch_data)
        self.trunk_data = torch.FloatTensor(trunk_data)
        self.target_data = torch.FloatTensor(target_data)
    
    def __len__(self):
        return len(self.branch_data)
    
    def __getitem__(self, idx):
        return self.branch_data[idx], self.trunk_data[idx], self.target_data[idx]

def create_deeponet_model(branch_input_dim, trunk_input_dim=1, 
                         latent_dim=128, hidden_layers=3, hidden_dim=128,
                         activation='tanh', dropout=0.1, device=None):
    """
    標準的なDeepONetモデルを作成するファクトリ関数
    
    Args:
        branch_input_dim (int): Branch networkの入力次元
        trunk_input_dim (int): Trunk networkの入力次元（通常1）
        latent_dim (int): 潜在空間の次元数
        hidden_layers (int): 隠れ層の数
        hidden_dim (int): 隠れ層の次元数
        activation (str): 活性化関数（'tanh', 'relu', 'gelu'）
        dropout (float): Dropout率（学習時のみ使用）
        device: PyTorchデバイス
    
    Returns:
        DeepONet: 作成されたモデル
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hidden_dims = [hidden_dim] * hidden_layers
    
    model = DeepONet(
        branch_input_dim=branch_input_dim,
        trunk_input_dim=trunk_input_dim,
        branch_hidden_dims=hidden_dims,
        trunk_hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        activation=activation,
        dropout=dropout
    ).to(device)
    
    return model

def load_deeponet_model(model_path, branch_input_dim, trunk_input_dim=1,
                       latent_dim=128, hidden_layers=3, hidden_dim=128,
                       activation='tanh', device=None):
    """
    保存されたDeepONetモデルを読み込む
    
    Args:
        model_path (str): モデルファイルのパス
        その他: create_deeponet_modelと同じ
    
    Returns:
        DeepONet: 読み込まれたモデル
    """
    model = create_deeponet_model(
        branch_input_dim=branch_input_dim,
        trunk_input_dim=trunk_input_dim,
        latent_dim=latent_dim,
        hidden_layers=hidden_layers,
        hidden_dim=hidden_dim,
        activation=activation,
        dropout=0.0,  # 評価時はDropout無効
        device=device
    )
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # 評価モード
    
    return model

def calculate_l2_relative_error(true_values, pred_values, epsilon=1e-10):
    """
    L2相対誤差を計算する共通関数
    
    Args:
        true_values (np.ndarray): 真値
        pred_values (np.ndarray): 予測値
        epsilon (float): ゼロ除算回避用の小さな値
    
    Returns:
        float: L2相対誤差
    """
    return np.linalg.norm(true_values - pred_values) / (np.linalg.norm(true_values) + epsilon)

def calculate_statistics(errors):
    """
    エラー配列から統計値を計算
    
    Args:
        errors (np.ndarray): エラー配列
    
    Returns:
        dict: 統計値の辞書
    """
    errors = np.array(errors)
    n_samples = len(errors)
    
    return {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'stderr': np.std(errors) / np.sqrt(n_samples),  # 標準誤差
        'min': np.min(errors),
        'max': np.max(errors),
        'median': np.median(errors),
        'n_samples': n_samples
    }