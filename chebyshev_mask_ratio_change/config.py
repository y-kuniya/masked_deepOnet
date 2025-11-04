#!/usr/bin/env python3
"""
共通設定 - シンプル版
"""

import torch
import os

# デバイス設定
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデル設定
MODEL_CONFIG = {
    'latent_dim': 128,
    'hidden_layers': 3,
    'hidden_dim': 128,
    'activation': 'tanh',
    'dropout': 0.0  # 正則化なし
}

# 学習設定  
TRAINING_CONFIG = {
    'epochs': 1000,
    'batch_size': 1000,
    'learning_rate': 1e-3,
    'weight_decay': 0.0  # 正則化なし
}

def get_paths(order):
    """ディレクトリパスを取得"""
    base_data_dir = f'./data_order_{order}'
    result_dir = f'./result_order_{order}_mask_experiment'
    os.makedirs(result_dir, exist_ok=True)
    return base_data_dir, result_dir