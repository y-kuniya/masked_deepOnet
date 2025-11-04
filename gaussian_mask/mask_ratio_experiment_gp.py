#!/usr/bin/env python3
"""
ガウス過程データ用マスク率実験スクリプト
使い方: python mask_ratio_experiment_gp.py --smoothness 5 --mask_ratios 0.2 0.3 --epochs 500
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time
import os
import argparse
import datetime
import pandas as pd
from typing import List, Tuple, Any, Dict

# 共通モジュールをインポート
from models import DeepONetDataset, create_deeponet_model, calculate_l2_relative_error, calculate_statistics
from config import DEVICE, MODEL_CONFIG, TRAINING_CONFIG

def get_gp_paths(smoothness):
    """ガウス過程データ用のディレクトリパスを取得"""
    base_data_dir = f'./data_gp_smooth_{smoothness}'
    result_dir = f'./result_gp_smooth_{smoothness}_mask_experiment'
    os.makedirs(result_dir, exist_ok=True)
    return base_data_dir, result_dir

class GaussianProcessMaskExperiment:
    """ガウス過程データ用マスク率実験クラス"""
    
    def __init__(self, smoothness: int):
        self.smoothness = smoothness
        self.base_data_dir, self.result_dir = get_gp_paths(smoothness)
        
        # データファイル存在確認
        branch_file = f"{self.base_data_dir}/deeponet_branch.npy"
        if not os.path.exists(branch_file):
            raise FileNotFoundError(
                f"ガウス過程データが見つかりません: {branch_file}\n"
                f"先に以下を実行してください:\n"
                f"python gp_data_generator.py --smoothness {smoothness} --samples 500 --nx 100"
            )
        
        # 元データを一度だけ読み込み
        self.branch_data = np.load(f"{self.base_data_dir}/deeponet_branch.npy")
        self.trunk_coords = np.load(f"{self.base_data_dir}/deeponet_trunk.npy") 
        self.target_data = np.load(f"{self.base_data_dir}/deeponet_target.npy")
        
        self.num_samples, self.Nx = self.branch_data.shape
        
        # テスト用インデックスを固定（一貫した評価のため）
        sample_indices = np.arange(self.num_samples)
        self.train_sample_idx, self.test_sample_idx = train_test_split(
            sample_indices, test_size=0.2, random_state=42
        )
        
        print(f"ガウス過程データ読み込み完了: {self.branch_data.shape}")
        print(f"Smoothness Level: {smoothness}")
        print(f"訓練サンプル: {len(self.train_sample_idx)}, テストサンプル: {len(self.test_sample_idx)}")
        
        # データの特性を表示
        self.analyze_data_characteristics()
    
    def analyze_data_characteristics(self):
        """データの特性を分析表示"""
        # 滑らかさメトリクス計算
        gradients = []
        curvatures = []
        for i in range(min(50, self.num_samples)):
            u = self.branch_data[i]
            grad = np.gradient(u, self.trunk_coords)
            curvature = np.gradient(grad, self.trunk_coords)
            gradients.append(np.std(grad))
            curvatures.append(np.std(curvature))
        
        print(f"データ特性分析:")
        print(f"  - データ範囲: [{np.min(self.branch_data):.3f}, {np.max(self.branch_data):.3f}]")
        print(f"  - 勾配標準偏差: {np.mean(gradients):.3f}")
        print(f"  - 曲率標準偏差: {np.mean(curvatures):.3f}")
        print(f"  - 最大隣接差分: {np.mean([np.max(np.abs(np.diff(self.branch_data[i]))) for i in range(min(50, self.num_samples))]):.3f}")
    
    def create_mask_indices(self, mask_ratio: float) -> Tuple[np.ndarray, int, int]:
        """マスク用のインデックスを作成"""
        mask_points = int(self.Nx * mask_ratio)
        unmask_start = mask_points
        unmask_end = self.Nx - mask_points
        eval_indices = np.arange(unmask_start, unmask_end)
        return eval_indices, unmask_start, unmask_end
    
    def prepare_data(self, mask_ratio: float, case: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """指定されたマスク率とケースでデータを準備"""
        eval_indices, unmask_start, unmask_end = self.create_mask_indices(mask_ratio)
        
        # マスク設定の表示
        mask_points = int(self.Nx * mask_ratio)
        print(f"マスク設定:")
        print(f"  全体グリッド点数: {self.Nx}")
        print(f"  マスク点数（各端）: {mask_points}")
        print(f"  評価領域: [{unmask_start}, {unmask_end}) ({len(eval_indices)}点)")
        
        if case == "case1":
            print(f"=== ケース1: 初期値全域 + 予測中央領域 ===")
            # ケース1: 初期値全域, 予測中央
            branch_input = []
            trunk_input = []
            target_output = []
            
            for i in range(self.num_samples):
                for j in eval_indices:
                    branch_input.append(self.branch_data[i])  # 全域
                    trunk_input.append([self.trunk_coords[j]])
                    target_output.append([self.target_data[i, j]])
                    
            branch_dim = self.Nx
            
        elif case == "case2":
            print(f"=== ケース2: 初期値中央部分 + 予測中央領域 ===")
            # ケース2: 初期値マスク, 予測中央
            branch_data_masked = self.branch_data[:, eval_indices]
            branch_input = []
            trunk_input = []
            target_output = []
            
            for i in range(self.num_samples):
                for j in eval_indices:
                    branch_input.append(branch_data_masked[i])  # マスク後
                    trunk_input.append([self.trunk_coords[j]])
                    target_output.append([self.target_data[i, j]])
            
            branch_dim = len(eval_indices)
        
        else:
            raise ValueError("case must be 'case1' or 'case2'")
        
        branch_input = np.array(branch_input, dtype=np.float32)
        trunk_input = np.array(trunk_input, dtype=np.float32)  
        target_output = np.array(target_output, dtype=np.float32)
        
        # データ形状の表示
        if case == "case1":
            print(f"ケース1データ形状:")
            print(f"  Branch input: {branch_input.shape} (全域初期条件)")
        else:
            print(f"ケース2データ形状:")
            print(f"  Branch input: {branch_input.shape} (マスク初期条件)")
        print(f"  Trunk input: {trunk_input.shape}")
        print(f"  Target output: {target_output.shape}")
        
        return branch_input, trunk_input, target_output, branch_dim
    
    def train_model(self, mask_ratio: float, case: str, num_epochs: int = None) -> Dict[str, Any]:
        """指定されたマスク率とケースでモデルを学習"""
        
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG['epochs']
        
        # データ準備
        branch_input, trunk_input, target_output, branch_dim = self.prepare_data(mask_ratio, case)
        
        # データローダー作成（固定されたtrain/test分割を使用）
        train_data_indices = []
        test_data_indices = []
        eval_indices, _, _ = self.create_mask_indices(mask_ratio)
        
        for i, sample_idx in enumerate(self.train_sample_idx):
            start_idx = sample_idx * len(eval_indices)
            end_idx = (sample_idx + 1) * len(eval_indices)
            train_data_indices.extend(range(start_idx, end_idx))
        
        for i, sample_idx in enumerate(self.test_sample_idx):
            start_idx = sample_idx * len(eval_indices) 
            end_idx = (sample_idx + 1) * len(eval_indices)
            test_data_indices.extend(range(start_idx, end_idx))
        
        train_dataset = DeepONetDataset(
            branch_input[train_data_indices], 
            trunk_input[train_data_indices], 
            target_output[train_data_indices]
        )
        test_dataset = DeepONetDataset(
            branch_input[test_data_indices],
            trunk_input[test_data_indices], 
            target_output[test_data_indices]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)
        
        print(f"データ分割: 訓練 {len(train_dataset)}, テスト {len(test_dataset)}")
        print(f"  {case} 学習開始...")
        
        # モデル作成（共通モジュール使用）
        model = create_deeponet_model(
            branch_input_dim=branch_dim,
            **MODEL_CONFIG,
            device=DEVICE
        )
        
        # 学習設定
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # 訓練
            model.train()
            train_loss = 0.0
            
            for batch_branch, batch_trunk, batch_target in train_loader:
                batch_branch = batch_branch.to(DEVICE)
                batch_trunk = batch_trunk.to(DEVICE)
                batch_target = batch_target.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(batch_branch, batch_trunk)
                loss = criterion(outputs, batch_target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # テスト
            model.eval()
            test_loss = 0.0
            
            with torch.no_grad():
                for batch_branch, batch_trunk, batch_target in test_loader:
                    batch_branch = batch_branch.to(DEVICE)
                    batch_trunk = batch_trunk.to(DEVICE)
                    batch_target = batch_target.to(DEVICE)
                    
                    outputs = model(batch_branch, batch_trunk)
                    loss = criterion(outputs, batch_target)
                    test_loss += loss.item()
            
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            
            scheduler.step()
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                model_path = f'{self.result_dir}/best_model_{case}_mask{int(mask_ratio*100)}.pth'
                torch.save(model.state_dict(), model_path)
            
            if (epoch + 1) % 100 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f'    Epoch [{epoch+1:4d}/{num_epochs}] | '
                      f'Train: {train_loss:.6f} | Test: {test_loss:.6f} | Time: {elapsed:.1f}s')
        
        print(f"  {case} 学習完了! 最良テスト損失: {best_test_loss:.6f}")
        
        return {
            'model': model,
            'train_losses': train_losses,
            'test_losses': test_losses, 
            'best_test_loss': best_test_loss,
            'branch_dim': branch_dim,
            'eval_indices': eval_indices
        }
    
    def evaluate_model(self, mask_ratio: float, case: str, model_info: Dict[str, Any]) -> np.ndarray:
        """モデルを評価してエラーを計算（共通関数使用）"""
        eval_indices = model_info['eval_indices']
        model = model_info['model']
        model.eval()
        
        errors = []
        
        if case == "case1":
            branch_data_input = self.branch_data
        else:  # case2
            branch_data_input = self.branch_data[:, eval_indices]
        
        with torch.no_grad():
            for sample_idx in self.test_sample_idx:
                # 真の値（評価領域）
                true_values = self.target_data[sample_idx, eval_indices]
                
                # 予測
                branch_input = torch.FloatTensor(branch_data_input[sample_idx]).unsqueeze(0).repeat(len(eval_indices), 1).to(DEVICE)
                trunk_input = torch.FloatTensor(self.trunk_coords[eval_indices]).unsqueeze(1).to(DEVICE)
                pred_values = model(branch_input, trunk_input).cpu().numpy().flatten()
                
                # 相対L2誤差（共通関数使用）
                l2_error = calculate_l2_relative_error(true_values, pred_values)
                errors.append(l2_error)
        
        return np.array(errors)
    
    def run_experiment(self, mask_ratios: List[float], num_epochs: int = None) -> pd.DataFrame:
        """複数のマスク率で実験を実行"""
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG['epochs']
            
        print(f"=== ガウス過程 Smoothness {self.smoothness} マスク率実験開始 ===")
        print(f"マスク率: {[r*100 for r in mask_ratios]}%")
        print(f"エポック数: {num_epochs}")
        print(f"結果保存先: {self.result_dir}")
        
        results = []
        
        for mask_ratio in mask_ratios:
            print(f"\n{'='*50}")
            print(f"マスク率 {mask_ratio*100}% 実験")
            print(f"{'='*50}")
            
            # Case1とCase2を学習
            case1_info = self.train_model(mask_ratio, "case1", num_epochs)
            case2_info = self.train_model(mask_ratio, "case2", num_epochs)
            
            # 評価
            case1_errors = self.evaluate_model(mask_ratio, "case1", case1_info)
            case2_errors = self.evaluate_model(mask_ratio, "case2", case2_info)
            
            # 統計計算（共通関数使用）
            case1_stats = calculate_statistics(case1_errors)
            case2_stats = calculate_statistics(case2_errors)
            
            # 改善率計算
            improvement = (case1_stats['mean'] - case2_stats['mean']) / case1_stats['mean'] * 100
            
            result = {
                'Smoothness': self.smoothness,
                'Mask_Ratio': mask_ratio,
                'Eval_Region_Pct': (1 - 2*mask_ratio) * 100,
                'Eval_Points': len(case1_info['eval_indices']),
                'Case1_Loss': case1_info['best_test_loss'],
                'Case2_Loss': case2_info['best_test_loss'],
                'Loss_Ratio': case2_info['best_test_loss'] / case1_info['best_test_loss'],
                'Case1_Mean_Error': case1_stats['mean'],
                'Case1_Std_Error': case1_stats['std'],
                'Case2_Mean_Error': case2_stats['mean'],
                'Case2_Std_Error': case2_stats['std'],
                'Improvement_Rate': improvement,
                'Case1_Input_Dim': case1_info['branch_dim'],
                'Case2_Input_Dim': case2_info['branch_dim']
            }
            
            results.append(result)
            
            print(f"\nマスク率 {mask_ratio*100}% 結果:")
            print(f"  Case1 誤差: {case1_stats['mean']:.6f} ± {case1_stats['stderr']:.6f} (SE)")
            print(f"  Case2 誤差: {case2_stats['mean']:.6f} ± {case2_stats['stderr']:.6f} (SE)")
            print(f"  相対変化率: {improvement:.2f}%")
            
            # 学習成功/失敗の判定
            if case2_stats['mean'] > 1.0:  # 相対誤差100%以上
                print(f"  ⚠️  Case2学習困難: 相対誤差 {case2_stats['mean']:.3f}")
            elif case2_stats['mean'] > 0.5:  # 相対誤差50%以上
                print(f"  🟡 Case2学習部分的: 相対誤差 {case2_stats['mean']:.3f}")
            else:
                print(f"  ✅ Case2学習成功: 相対誤差 {case2_stats['mean']:.3f}")
        
        return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Gaussian Process Mask Ratio Experiment')
    parser.add_argument('--smoothness', type=int, default=5, help='Smoothness level (0:smooth → 10:sharp)')
    parser.add_argument('--epochs', type=int, default=None, help=f'Number of epochs (default: {TRAINING_CONFIG["epochs"]})')
    parser.add_argument('--mask_ratios', nargs='+', type=float, 
                       default=[0.2, 0.3], 
                       help='List of mask ratios to test')
    args = parser.parse_args()
    
    if not (0 <= args.smoothness <= 10):
        print("Warning: smoothness level should be between 0-10")
        print("  0-3: RBF kernel (infinitely smooth)")
        print("  4-6: Matérn 5/2 kernel (moderately smooth)")  
        print("  7-8: Matérn 3/2 kernel (less smooth)")
        print("  9-10: Matérn 1/2 kernel (sharp)")
    
    print("="*60)
    print(f"ガウス過程マスク率実験")
    print("="*60)
    print(f"Smoothness Level: {args.smoothness}")
    print(f"マスク率: {[r*100 for r in args.mask_ratios]}%")
    print(f"エポック数: {args.epochs or TRAINING_CONFIG['epochs']}")
    print(f"デバイス: {DEVICE}")
    
    try:
        # 実験実行
        experiment = GaussianProcessMaskExperiment(args.smoothness)
        results_df = experiment.run_experiment(args.mask_ratios, args.epochs)
        
        # 結果保存
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f'{experiment.result_dir}/gp_mask_results_{timestamp}.csv'
        results_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # 簡潔なサマリー表示
        print(f"\n{'='*60}")
        print("実験完了サマリー")
        print(f"{'='*60}")
        print(results_df.to_string(index=False, float_format='%.4f'))
        print(f"\n結果ファイル: {csv_path}")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print(f"\n解決方法:")
        print(f"python gp_data_generator.py --smoothness {args.smoothness} --samples 500 --nx 100")

if __name__ == "__main__":
    main()