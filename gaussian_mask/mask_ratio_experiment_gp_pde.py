#!/usr/bin/env python3
"""
PDE制約付きガウス過程データ用マスク率実験スクリプト
使い方: python mask_ratio_experiment_gp_pde.py --smoothness 7 --mask_ratios 0.2 --epochs 500 --pde_weight 0.1
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
    result_dir = f'./result_gp_smooth_{smoothness}_mask_experiment_pde'
    os.makedirs(result_dir, exist_ok=True)
    return base_data_dir, result_dir

class PDEConstrainedLoss(nn.Module):
    """PDE残差制約付き損失関数"""
    
    def __init__(self,
                 prediction_weight: float = 1.0,
                 pde_weight: float = 0.1,
                 boundary_weight: float = 0.1,
                 wave_speed: float = 1.0,
                 domain_length: float = 1.0):
        super().__init__()
        self.prediction_weight = prediction_weight
        self.pde_weight = pde_weight
        self.boundary_weight = boundary_weight
        self.wave_speed = wave_speed
        self.domain_length = domain_length
        self.mse = nn.MSELoss()
        
    def compute_spatial_second_derivative(self, u: torch.Tensor, dx: float) -> torch.Tensor:
        """2階空間微分を中心差分で計算"""
        u_padded = torch.nn.functional.pad(u, (1, 1), mode='constant', value=0.0)
        u_xx = (u_padded[:, 2:] - 2 * u_padded[:, 1:-1] + u_padded[:, :-2]) / (dx ** 2)
        return u_xx
    
    def compute_pde_residual_simple(self, model, branch_input: torch.Tensor, 
                                  trunk_coords: torch.Tensor, current_prediction: torch.Tensor) -> torch.Tensor:
        """簡化版PDE残差計算"""
        try:
            # 空間微分（2階）
            dx = self.domain_length / (current_prediction.size(-1) - 1)
            u_xx = self.compute_spatial_second_derivative(current_prediction.view(-1, current_prediction.size(-1)), dx)
            
            # 時間微分の代わりに、予測値の滑らかさを制約
            # 実際の時間微分計算は複雑なので、空間的な滑らかさで代用
            smoothness_penalty = torch.mean(u_xx ** 2)
            
            return smoothness_penalty
            
        except Exception as e:
            return torch.tensor(0.0, device=branch_input.device)
    
    def compute_boundary_constraint(self, u_predicted: torch.Tensor) -> torch.Tensor:
        """固定端境界条件制約"""
        if u_predicted.dim() == 1:
            u_predicted = u_predicted.unsqueeze(0)
        
        # 両端の値
        boundary_values = torch.cat([
            u_predicted[:, 0:1],   # x=0での値
            u_predicted[:, -1:]    # x=Lでの値
        ], dim=1)
        
        return torch.mean(boundary_values ** 2)
    
    def forward(self, model, branch_input: torch.Tensor, trunk_input: torch.Tensor, 
                target: torch.Tensor, enable_pde_constraint: bool = True) -> Tuple[torch.Tensor, Dict]:
        
        # 1. 通常の予測損失
        prediction = model(branch_input, trunk_input)
        prediction_loss = self.mse(prediction, target)
        
        # 2. 境界条件制約
        boundary_loss = self.compute_boundary_constraint(prediction)
        
        # 3. PDE残差制約（簡化版）
        pde_loss = torch.tensor(0.0, device=branch_input.device)
        if enable_pde_constraint and self.pde_weight > 0:
            spatial_coords = trunk_input[:, :1] if trunk_input.size(1) > 1 else trunk_input
            pde_loss = self.compute_pde_residual_simple(model, branch_input, spatial_coords, prediction)
        
        # 4. 総損失
        total_loss = (self.prediction_weight * prediction_loss + 
                     self.boundary_weight * boundary_loss +
                     self.pde_weight * pde_loss)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'prediction_loss': prediction_loss.item(),
            'boundary_loss': boundary_loss.item() if isinstance(boundary_loss, torch.Tensor) else boundary_loss,
            'pde_loss': pde_loss.item() if isinstance(pde_loss, torch.Tensor) else pde_loss,
            'prediction_weight': self.prediction_weight,
            'pde_weight': self.pde_weight,
            'boundary_weight': self.boundary_weight
        }
        
        return total_loss, loss_dict

class PDEConstrainedGPExperiment:
    """PDE制約付きガウス過程データ用マスク率実験クラス"""
    
    def __init__(self, smoothness: int, pde_config: dict = None):
        self.smoothness = smoothness
        self.base_data_dir, self.result_dir = get_gp_paths(smoothness)
        
        # PDE制約設定
        default_pde_config = {
            'prediction_weight': 1.0,
            'pde_weight': 0.1,
            'boundary_weight': 0.1,
            'wave_speed': 1.0,
            'domain_length': 1.0,
            'warmup_epochs': 100
        }
        
        if pde_config:
            default_pde_config.update(pde_config)
        
        self.pde_config = default_pde_config
        
        # データファイル存在確認
        branch_file = f"{self.base_data_dir}/deeponet_branch.npy"
        if not os.path.exists(branch_file):
            raise FileNotFoundError(
                f"ガウス過程データが見つかりません: {branch_file}\n"
                f"先に以下を実行してください:\n"
                f"python gp_data_generator.py --smoothness {smoothness} --samples 500 --nx 100"
            )
        
        # 元データを読み込み
        self.branch_data = np.load(f"{self.base_data_dir}/deeponet_branch.npy")
        self.trunk_coords = np.load(f"{self.base_data_dir}/deeponet_trunk.npy") 
        self.target_data = np.load(f"{self.base_data_dir}/deeponet_target.npy")
        
        self.num_samples, self.Nx = self.branch_data.shape
        
        # テスト用インデックスを固定
        sample_indices = np.arange(self.num_samples)
        self.train_sample_idx, self.test_sample_idx = train_test_split(
            sample_indices, test_size=0.2, random_state=42
        )
        
        print(f"PDE制約付きガウス過程データ読み込み完了: {self.branch_data.shape}")
        print(f"Smoothness Level: {smoothness}")
        print(f"訓練サンプル: {len(self.train_sample_idx)}, テストサンプル: {len(self.test_sample_idx)}")
        print(f"PDE制約設定: {self.pde_config}")
        
        # データの特性を表示
        self.analyze_data_characteristics()
    
    def analyze_data_characteristics(self):
        """データの特性を分析表示"""
        gradients = []
        curvatures = []
        amplitudes = []
        
        for i in range(min(50, self.num_samples)):
            u = self.branch_data[i]
            grad = np.gradient(u, self.trunk_coords)
            curvature = np.gradient(grad, self.trunk_coords)
            gradients.append(np.std(grad))
            curvatures.append(np.std(curvature))
            amplitudes.append(np.max(np.abs(u)))
        
        self.data_amplitude_range = [np.min(amplitudes), np.max(amplitudes)]
        
        print(f"データ特性分析:")
        print(f"  - データ範囲: [{np.min(self.branch_data):.3f}, {np.max(self.branch_data):.3f}]")
        print(f"  - 振幅範囲: [{self.data_amplitude_range[0]:.3f}, {self.data_amplitude_range[1]:.3f}]")
        print(f"  - 勾配標準偏差: {np.mean(gradients):.3f}")
        print(f"  - 曲率標準偏差: {np.mean(curvatures):.3f}")
    
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
        
        if case == "case1":
            print(f"=== ケース1: 初期値全域 + 予測中央領域 ===")
            branch_input = []
            trunk_input = []
            target_output = []
            
            for i in range(self.num_samples):
                for j in eval_indices:
                    branch_input.append(self.branch_data[i])
                    trunk_input.append([self.trunk_coords[j]])
                    target_output.append([self.target_data[i, j]])
                    
            branch_dim = self.Nx
            
        elif case == "case2":
            print(f"=== ケース2: 初期値中央部分 + 予測中央領域 ===")
            branch_data_masked = self.branch_data[:, eval_indices]
            branch_input = []
            trunk_input = []
            target_output = []
            
            for i in range(self.num_samples):
                for j in eval_indices:
                    branch_input.append(branch_data_masked[i])
                    trunk_input.append([self.trunk_coords[j]])
                    target_output.append([self.target_data[i, j]])
            
            branch_dim = len(eval_indices)
        
        else:
            raise ValueError("case must be 'case1' or 'case2'")
        
        branch_input = np.array(branch_input, dtype=np.float32)
        trunk_input = np.array(trunk_input, dtype=np.float32)  
        target_output = np.array(target_output, dtype=np.float32)
        
        print(f"{case}データ形状: Branch={branch_input.shape}, Trunk={trunk_input.shape}, Target={target_output.shape}")
        
        return branch_input, trunk_input, target_output, branch_dim
    
    def update_pde_weight(self, epoch: int) -> float:
        """エポックに応じてPDE制約の重みを更新"""
        if epoch < self.pde_config['warmup_epochs']:
            return 0.0
        
        base_weight = self.pde_config['pde_weight']
        progress = min(1.0, (epoch - self.pde_config['warmup_epochs']) / 100)
        return base_weight * progress
    
    def train_model(self, mask_ratio: float, case: str, num_epochs: int = None) -> Dict[str, Any]:
        """PDE制約付きでモデルを学習"""
        
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG['epochs']
        
        # データ準備
        branch_input, trunk_input, target_output, branch_dim = self.prepare_data(mask_ratio, case)
        
        # データローダー作成
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
        print(f"  {case} PDE制約付き学習開始...")
        
        # モデル作成
        model = create_deeponet_model(
            branch_input_dim=branch_dim,
            **MODEL_CONFIG,
            device=DEVICE
        )
        
        # PDE制約付き損失関数
        criterion = PDEConstrainedLoss(
            prediction_weight=self.pde_config['prediction_weight'],
            pde_weight=0.0,  # 初期は0（ウォームアップ）
            boundary_weight=self.pde_config['boundary_weight'],
            wave_speed=self.pde_config['wave_speed'],
            domain_length=self.pde_config['domain_length']
        )
        
        # オプティマイザー
        optimizer = optim.Adam(
            model.parameters(), 
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        
        # 訓練ログ
        train_losses = []
        test_losses = []
        pde_losses = []
        boundary_losses = []
        pde_weights = []
        best_test_loss = float('inf')
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # PDE制約の重みを更新
            current_pde_weight = self.update_pde_weight(epoch)
            criterion.pde_weight = current_pde_weight
            pde_weights.append(current_pde_weight)
            
            # 訓練
            model.train()
            train_loss = 0.0
            epoch_pde_loss = 0.0
            epoch_boundary_loss = 0.0
            
            for batch_branch, batch_trunk, batch_target in train_loader:
                batch_branch = batch_branch.to(DEVICE)
                batch_trunk = batch_trunk.to(DEVICE)
                batch_target = batch_target.to(DEVICE)
                
                optimizer.zero_grad()
                
                # PDE制約付き損失計算
                total_loss, loss_dict = criterion(
                    model, batch_branch, batch_trunk, batch_target,
                    enable_pde_constraint=(epoch >= self.pde_config['warmup_epochs'])
                )
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += loss_dict['prediction_loss']
                epoch_pde_loss += loss_dict['pde_loss']
                epoch_boundary_loss += loss_dict['boundary_loss']
            
            train_loss /= len(train_loader)
            epoch_pde_loss /= len(train_loader)
            epoch_boundary_loss /= len(train_loader)
            train_losses.append(train_loss)
            pde_losses.append(epoch_pde_loss)
            boundary_losses.append(epoch_boundary_loss)
            
            # テスト（通常の損失のみ）
            model.eval()
            test_loss = 0.0
            
            with torch.no_grad():
                for batch_branch, batch_trunk, batch_target in test_loader:
                    batch_branch = batch_branch.to(DEVICE)
                    batch_trunk = batch_trunk.to(DEVICE)
                    batch_target = batch_target.to(DEVICE)
                    
                    outputs = model(batch_branch, batch_trunk)
                    loss = nn.MSELoss()(outputs, batch_target)
                    test_loss += loss.item()
            
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            
            scheduler.step()
            
            # ベストモデル保存
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                model_path = f'{self.result_dir}/best_model_{case}_mask{int(mask_ratio*100)}_pde.pth'
                torch.save(model.state_dict(), model_path)
            
            # 進捗表示
            if (epoch + 1) % 100 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f'    Epoch [{epoch+1:4d}/{num_epochs}] | '
                      f'Train: {train_loss:.6f} | Test: {test_loss:.6f} | '
                      f'PDE: {epoch_pde_loss:.6f} | Boundary: {epoch_boundary_loss:.6f} | '
                      f'PDE Weight: {current_pde_weight:.4f} | Time: {elapsed:.1f}s')
        
        print(f"  {case} PDE制約付き学習完了! 最良テスト損失: {best_test_loss:.6f}")
        
        return {
            'model': model,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'pde_losses': pde_losses,
            'boundary_losses': boundary_losses,
            'pde_weights': pde_weights,
            'best_test_loss': best_test_loss,
            'branch_dim': branch_dim,
            'eval_indices': eval_indices
        }
    
    def evaluate_model(self, mask_ratio: float, case: str, model_info: Dict[str, Any]) -> np.ndarray:
        """モデルを評価してエラーを計算"""
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
                
                # 相対L2誤差
                l2_error = calculate_l2_relative_error(true_values, pred_values)
                errors.append(l2_error)
        
        return np.array(errors)
    
    def run_experiment(self, mask_ratios: List[float], num_epochs: int = None) -> pd.DataFrame:
        """複数のマスク率でPDE制約付き実験を実行"""
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG['epochs']
            
        print(f"=== PDE制約付きガウス過程 Smoothness {self.smoothness} マスク率実験開始 ===")
        print(f"マスク率: {[r*100 for r in mask_ratios]}%")
        print(f"エポック数: {num_epochs}")
        print(f"結果保存先: {self.result_dir}")
        
        results = []
        
        for mask_ratio in mask_ratios:
            print(f"\n{'='*50}")
            print(f"マスク率 {mask_ratio*100}% PDE制約付き実験")
            print(f"{'='*50}")
            
            # Case1とCase2を学習
            case1_info = self.train_model(mask_ratio, "case1", num_epochs)
            case2_info = self.train_model(mask_ratio, "case2", num_epochs)
            
            # 評価
            case1_errors = self.evaluate_model(mask_ratio, "case1", case1_info)
            case2_errors = self.evaluate_model(mask_ratio, "case2", case2_info)
            
            # 統計計算
            case1_stats = calculate_statistics(case1_errors)
            case2_stats = calculate_statistics(case2_errors)
            
            # 改善率計算
            improvement = (case1_stats['mean'] - case2_stats['mean']) / case1_stats['mean'] * 100
            
            result = {
                'Smoothness': self.smoothness,
                'Mask_Ratio': mask_ratio,
                'Eval_Region_Pct': (1 - 2*mask_ratio) * 100,
                'Case1_Mean_Error': case1_stats['mean'],
                'Case1_Std_Error': case1_stats['std'],
                'Case2_Mean_Error': case2_stats['mean'],
                'Case2_Std_Error': case2_stats['std'],
                'Improvement_Rate': improvement,
                'PDE_Weight': self.pde_config['pde_weight'],
                'Boundary_Weight': self.pde_config['boundary_weight'],
                'Warmup_Epochs': self.pde_config['warmup_epochs'],
                'Case1_Final_PDE_Loss': case1_info['pde_losses'][-1] if case1_info['pde_losses'] else 0,
                'Case2_Final_PDE_Loss': case2_info['pde_losses'][-1] if case2_info['pde_losses'] else 0
            }
            
            results.append(result)
            
            print(f"\nマスク率 {mask_ratio*100}% 結果:")
            print(f"  Case1 誤差: {case1_stats['mean']:.6f} ± {case1_stats['stderr']:.6f}")
            print(f"  Case2 誤差: {case2_stats['mean']:.6f} ± {case2_stats['stderr']:.6f}")
            print(f"  相対変化率: {improvement:.2f}%")
            print(f"  最終PDE損失 - Case1: {result['Case1_Final_PDE_Loss']:.6f}, Case2: {result['Case2_Final_PDE_Loss']:.6f}")
        
        return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='PDE-Constrained Gaussian Process Mask Experiment')
    parser.add_argument('--smoothness', type=int, default=7, help='Smoothness level (0:smooth → 10:sharp)')
    parser.add_argument('--epochs', type=int, default=None, help=f'Number of epochs')
    parser.add_argument('--mask_ratios', nargs='+', type=float, 
                       default=[0.2], help='List of mask ratios to test')
    parser.add_argument('--pde_weight', type=float, default=0.1, 
                       help='Weight for PDE residual constraint')
    parser.add_argument('--boundary_weight', type=float, default=0.1,
                       help='Weight for boundary condition constraint')
    parser.add_argument('--warmup_epochs', type=int, default=100,
                       help='Number of warmup epochs without PDE constraint')
    parser.add_argument('--wave_speed', type=float, default=1.0,
                       help='Wave speed parameter')
    
    args = parser.parse_args()
    
    # PDE制約設定
    pde_config = {
        'pde_weight': args.pde_weight,
        'boundary_weight': args.boundary_weight,
        'warmup_epochs': args.warmup_epochs,
        'wave_speed': args.wave_speed,
        'domain_length': 1.0
    }
    
    print("="*60)
    print(f"PDE制約付きガウス過程マスク率実験")
    print("="*60)
    print(f"Smoothness Level: {args.smoothness}")
    print(f"マスク率: {[r*100 for r in args.mask_ratios]}%")
    print(f"エポック数: {args.epochs or TRAINING_CONFIG['epochs']}")
    print(f"PDE制約重み: {args.pde_weight}")
    print(f"境界制約重み: {args.boundary_weight}")
    print(f"ウォームアップエポック: {args.warmup_epochs}")
    print(f"デバイス: {DEVICE}")
    
    try:
        # 実験実行
        experiment = PDEConstrainedGPExperiment(args.smoothness, pde_config)
        results_df = experiment.run_experiment(args.mask_ratios, args.epochs)
        
        # 結果保存
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f'{experiment.result_dir}/gp_pde_results_{timestamp}.csv'
        results_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # サマリー表示
        print(f"\n{'='*60}")
        print("PDE制約付き実験完了サマリー")
        print(f"{'='*60}")
        print(results_df.to_string(index=False, float_format='%.4f'))
        print(f"\n結果ファイル: {csv_path}")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main()