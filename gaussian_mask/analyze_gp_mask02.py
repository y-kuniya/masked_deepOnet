#!/usr/bin/env python3
"""
ガウス過程データ用マスク率0.2専用分析スクリプト（スケール変換テスト付き）
使用方法: python analyze_gp_mask02_enhanced.py --smoothness 5
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import argparse
import torch
from sklearn.model_selection import train_test_split

# 共通モジュールをインポート
from models import load_deeponet_model, calculate_l2_relative_error, calculate_statistics
from config import DEVICE, MODEL_CONFIG

def get_gp_paths(smoothness):
    """ガウス過程データ用のディレクトリパスを取得"""
    base_data_dir = f'./data_gp_smooth_{smoothness}'
    result_dir = f'./result_gp_smooth_{smoothness}_mask_experiment'
    return base_data_dir, result_dir

class EnhancedGPMaskRatio02Analyzer:
    """スケール変換テスト付きガウス過程データ用マスク率0.2専用分析クラス"""

    def __init__(self, smoothness):
        self.smoothness = smoothness
        self.mask_ratio = 0.2  # 固定
        self.base_data_dir, self.result_dir = get_gp_paths(smoothness)

        # データ読み込み
        try:
            self.branch_data = np.load(f"{self.base_data_dir}/deeponet_branch.npy")
            self.trunk_coords = np.load(f"{self.base_data_dir}/deeponet_trunk.npy")
            self.target_data = np.load(f"{self.base_data_dir}/deeponet_target.npy")
            self.num_samples, self.Nx = self.branch_data.shape
        except FileNotFoundError:
            raise FileNotFoundError(
                f"ガウス過程データが見つかりません: {self.base_data_dir}\n"
                f"先に以下を実行してください:\n"
                f"python gp_data_generator.py --smoothness {smoothness} --samples 500 --nx 100"
            )

        # 学習と同じtrain/test分割を再現
        sample_indices = np.arange(self.num_samples)
        self.train_sample_idx, self.test_sample_idx = train_test_split(
            sample_indices, test_size=0.2, random_state=42
        )

        # マスク設定
        self.eval_indices, self.unmask_start, self.unmask_end = self.create_mask_indices()

        print(f"ガウス過程データ読み込み完了:")
        print(f"  Smoothness Level: {self.smoothness}")
        print(f"  全サンプル数: {self.num_samples}")
        print(f"  テストサンプル数: {len(self.test_sample_idx)}")
        print(f"  グリッド点数: {self.Nx}")
        print(f"  マスク率: {self.mask_ratio} (中央部 {(1-2*self.mask_ratio)*100:.0f}%)")
        print(f"  評価領域: [{self.unmask_start}, {self.unmask_end}) ({len(self.eval_indices)}点)")

        # データ特性分析
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
        
        print(f"  データ特性:")
        print(f"    - データ範囲: [{np.min(self.branch_data):.3f}, {np.max(self.branch_data):.3f}]")
        print(f"    - 振幅範囲: [{self.data_amplitude_range[0]:.3f}, {self.data_amplitude_range[1]:.3f}]")
        print(f"    - 勾配標準偏差: {np.mean(gradients):.3f}")
        print(f"    - 曲率標準偏差: {np.mean(curvatures):.3f}")
        print(f"    - 最大隣接差分: {np.mean([np.max(np.abs(np.diff(self.branch_data[i]))) for i in range(min(50, self.num_samples))]):.3f}")

    def create_mask_indices(self):
        """マスク率0.2でのインデックス作成"""
        mask_points = int(self.Nx * self.mask_ratio)
        unmask_start = mask_points
        unmask_end = self.Nx - mask_points
        eval_indices = np.arange(unmask_start, unmask_end)
        return eval_indices, unmask_start, unmask_end

    def load_models(self):
        """Case1とCase2のモデルを読み込み"""
        try:
            # Case1モデル（全域初期条件）
            case1_model_path = f'{self.result_dir}/best_model_case1_mask20.pth'
            self.case1_model = load_deeponet_model(
                model_path=case1_model_path,
                branch_input_dim=self.Nx,  # 全域
                latent_dim=MODEL_CONFIG['latent_dim'],
                hidden_layers=MODEL_CONFIG['hidden_layers'],
                hidden_dim=MODEL_CONFIG['hidden_dim'],
                activation=MODEL_CONFIG['activation'],
                device=DEVICE
            )
            print(f"Case1モデル読み込み完了: {case1_model_path}")

            # Case2モデル（マスク初期条件）
            case2_model_path = f'{self.result_dir}/best_model_case2_mask20.pth'
            self.case2_model = load_deeponet_model(
                model_path=case2_model_path,
                branch_input_dim=len(self.eval_indices),  # マスク後
                latent_dim=MODEL_CONFIG['latent_dim'],
                hidden_layers=MODEL_CONFIG['hidden_layers'],
                hidden_dim=MODEL_CONFIG['hidden_dim'],
                activation=MODEL_CONFIG['activation'],
                device=DEVICE
            )
            print(f"Case2モデル読み込み完了: {case2_model_path}")

            return True

        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            print(f"先に以下を実行してください:")
            print(f"python mask_ratio_experiment_gp.py --smoothness {self.smoothness} --mask_ratios 0.2 --epochs 300")
            return False

    def scale_transform_test(self, scale_factors=[0.5, 2.0], n_test_samples=20):
        """スケール変換テストを実行"""
        print(f"\n{'='*60}")
        print(f"スケール変換テスト実行中...")
        print(f"スケール係数: {scale_factors}")
        print(f"テストサンプル数: {n_test_samples}")
        print(f"{'='*60}")
        
        # テストサンプルを選択（最初のn_test_samples個）
        test_samples = self.test_sample_idx[:n_test_samples]
        
        scale_results = {}
        
        for scale_factor in scale_factors:
            print(f"\nスケール係数 {scale_factor} での実行中...")
            
            case1_original_errors = []
            case1_scaled_errors = []
            case1_linearity_errors = []
            
            case2_original_errors = []
            case2_scaled_errors = []
            case2_linearity_errors = []
            
            with torch.no_grad():
                for i, sample_idx in enumerate(test_samples):
                    if (i + 1) % 5 == 0:
                        print(f"  進捗: {i + 1}/{len(test_samples)}")
                    
                    # 元の初期条件
                    u0_original = self.branch_data[sample_idx]
                    true_values = self.target_data[sample_idx, self.eval_indices]
                    
                    # スケール変換した初期条件
                    u0_scaled = scale_factor * u0_original
                    
                    # 真の解もスケール変換（線形性により）
                    true_values_scaled = scale_factor * true_values
                    
                    # Case1: 元の初期条件での予測
                    branch_input_case1_orig = torch.FloatTensor(u0_original).unsqueeze(0).repeat(len(self.eval_indices), 1).to(DEVICE)
                    trunk_input = torch.FloatTensor(self.trunk_coords[self.eval_indices]).unsqueeze(1).to(DEVICE)
                    pred_case1_orig = self.case1_model(branch_input_case1_orig, trunk_input).cpu().numpy().flatten()
                    
                    # Case1: スケール変換した初期条件での予測
                    branch_input_case1_scaled = torch.FloatTensor(u0_scaled).unsqueeze(0).repeat(len(self.eval_indices), 1).to(DEVICE)
                    pred_case1_scaled = self.case1_model(branch_input_case1_scaled, trunk_input).cpu().numpy().flatten()
                    
                    # Case1: 線形性チェック
                    pred_case1_expected = scale_factor * pred_case1_orig
                    
                    # Case2: 元の初期条件（マスク）での予測
                    branch_input_case2_orig = torch.FloatTensor(u0_original[self.eval_indices]).unsqueeze(0).repeat(len(self.eval_indices), 1).to(DEVICE)
                    pred_case2_orig = self.case2_model(branch_input_case2_orig, trunk_input).cpu().numpy().flatten()
                    
                    # Case2: スケール変換した初期条件（マスク）での予測
                    branch_input_case2_scaled = torch.FloatTensor(u0_scaled[self.eval_indices]).unsqueeze(0).repeat(len(self.eval_indices), 1).to(DEVICE)
                    pred_case2_scaled = self.case2_model(branch_input_case2_scaled, trunk_input).cpu().numpy().flatten()
                    
                    # Case2: 線形性チェック
                    pred_case2_expected = scale_factor * pred_case2_orig
                    
                    # 誤差計算
                    # Case1
                    case1_orig_error = calculate_l2_relative_error(true_values, pred_case1_orig)
                    case1_scaled_error = calculate_l2_relative_error(true_values_scaled, pred_case1_scaled)
                    case1_linearity_error = calculate_l2_relative_error(pred_case1_scaled, pred_case1_expected)
                    
                    case1_original_errors.append(case1_orig_error)
                    case1_scaled_errors.append(case1_scaled_error)
                    case1_linearity_errors.append(case1_linearity_error)
                    
                    # Case2
                    case2_orig_error = calculate_l2_relative_error(true_values, pred_case2_orig)
                    case2_scaled_error = calculate_l2_relative_error(true_values_scaled, pred_case2_scaled)
                    case2_linearity_error = calculate_l2_relative_error(pred_case2_scaled, pred_case2_expected)
                    
                    case2_original_errors.append(case2_orig_error)
                    case2_scaled_errors.append(case2_scaled_error)
                    case2_linearity_errors.append(case2_linearity_error)
            
            # 結果保存
            scale_results[scale_factor] = {
                'case1_original_errors': np.array(case1_original_errors),
                'case1_scaled_errors': np.array(case1_scaled_errors),
                'case1_linearity_errors': np.array(case1_linearity_errors),
                'case2_original_errors': np.array(case2_original_errors),
                'case2_scaled_errors': np.array(case2_scaled_errors),
                'case2_linearity_errors': np.array(case2_linearity_errors)
            }
            
            print(f"  完了: スケール係数 {scale_factor}")
        
        self.scale_results = scale_results
        return scale_results
    
    def analyze_scale_results(self):
        """スケール変換テスト結果を分析"""
        print(f"\n{'='*60}")
        print(f"スケール変換テスト結果分析")
        print(f"{'='*60}")
        
        for scale_factor, results in self.scale_results.items():
            print(f"\n【スケール係数 {scale_factor}】")
            
            # Case1分析
            case1_orig_stats = calculate_statistics(results['case1_original_errors'])
            case1_scaled_stats = calculate_statistics(results['case1_scaled_errors'])
            case1_linearity_stats = calculate_statistics(results['case1_linearity_errors'])
            
            print(f"  Case1 (全域初期条件):")
            print(f"    元の誤差: {case1_orig_stats['mean']:.4f} ± {case1_orig_stats['stderr']:.4f}")
            print(f"    スケール後誤差: {case1_scaled_stats['mean']:.4f} ± {case1_scaled_stats['stderr']:.4f}")
            print(f"    線形性誤差: {case1_linearity_stats['mean']:.4f} ± {case1_linearity_stats['stderr']:.4f}")
            
            # Case2分析
            case2_orig_stats = calculate_statistics(results['case2_original_errors'])
            case2_scaled_stats = calculate_statistics(results['case2_scaled_errors'])
            case2_linearity_stats = calculate_statistics(results['case2_linearity_errors'])
            
            print(f"  Case2 (マスク初期条件):")
            print(f"    元の誤差: {case2_orig_stats['mean']:.4f} ± {case2_orig_stats['stderr']:.4f}")
            print(f"    スケール後誤差: {case2_scaled_stats['mean']:.4f} ± {case2_scaled_stats['stderr']:.4f}")
            print(f"    線形性誤差: {case2_linearity_stats['mean']:.4f} ± {case2_linearity_stats['stderr']:.4f}")
            
            # 線形性判定
            linearity_threshold = 0.1  # 10%以下なら線形とみなす
            
            case1_is_linear = case1_linearity_stats['mean'] < linearity_threshold
            case2_is_linear = case2_linearity_stats['mean'] < linearity_threshold
            
            print(f"  線形性判定:")
            print(f"    Case1: {'✅ 線形' if case1_is_linear else '❌ 非線形'} (誤差 {case1_linearity_stats['mean']*100:.1f}%)")
            print(f"    Case2: {'✅ 線形' if case2_is_linear else '❌ 非線形'} (誤差 {case2_linearity_stats['mean']*100:.1f}%)")
    
    def create_scale_analysis_plots(self):
        """スケール変換テスト結果をプロット"""
        if not hasattr(self, 'scale_results'):
            print("スケール変換テストが実行されていません。")
            return
        
        n_scales = len(self.scale_results)
        fig, axes = plt.subplots(2, n_scales, figsize=(6*n_scales, 12))
        
        if n_scales == 1:
            axes = axes.reshape(2, 1)
        
        scale_factors = list(self.scale_results.keys())
        
        for i, scale_factor in enumerate(scale_factors):
            results = self.scale_results[scale_factor]
            
            # Case1線形性誤差分布
            ax1 = axes[0, i]
            ax1.hist(results['case1_linearity_errors'], bins=20, alpha=0.7, color='blue', 
                    density=True, edgecolor='black', linewidth=0.5)
            
            mean_err = np.mean(results['case1_linearity_errors'])
            ax1.axvline(mean_err, color='red', linestyle='--', linewidth=2)
            ax1.axvline(0.1, color='green', linestyle=':', linewidth=2, label='Linearity threshold (10%)')
            
            ax1.set_xlabel('Linearity Error')
            ax1.set_ylabel('Density')
            ax1.set_title(f'Case1 Linearity (Scale {scale_factor})\nMean: {mean_err:.4f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Case2線形性誤差分布
            ax2 = axes[1, i]
            ax2.hist(results['case2_linearity_errors'], bins=20, alpha=0.7, color='orange',
                    density=True, edgecolor='black', linewidth=0.5)
            
            mean_err = np.mean(results['case2_linearity_errors'])
            ax2.axvline(mean_err, color='red', linestyle='--', linewidth=2)
            ax2.axvline(0.1, color='green', linestyle=':', linewidth=2, label='Linearity threshold (10%)')
            
            ax2.set_xlabel('Linearity Error')
            ax2.set_ylabel('Density')
            ax2.set_title(f'Case2 Linearity (Scale {scale_factor})\nMean: {mean_err:.4f}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        plot_path = f'{self.result_dir}/scale_analysis_gp_smooth_{self.smoothness}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"スケール変換分析結果保存: {plot_path}")

    def calculate_all_errors(self):
        """全テストサンプルの誤差を計算"""
        case1_errors = []
        case2_errors = []

        print("通常誤差計算中...")

        with torch.no_grad():
            for i, sample_idx in enumerate(self.test_sample_idx):
                if (i + 1) % 10 == 0:
                    print(f"  進捗: {i + 1}/{len(self.test_sample_idx)}")

                # 真の値（評価領域）
                true_values = self.target_data[sample_idx, self.eval_indices]

                # Case1予測（全域初期条件）
                branch_input_case1 = torch.FloatTensor(self.branch_data[sample_idx]).unsqueeze(0).repeat(len(self.eval_indices), 1).to(DEVICE)
                trunk_input = torch.FloatTensor(self.trunk_coords[self.eval_indices]).unsqueeze(1).to(DEVICE)
                pred_case1 = self.case1_model(branch_input_case1, trunk_input).cpu().numpy().flatten()

                # Case2予測（マスク初期条件）
                branch_input_case2 = torch.FloatTensor(self.branch_data[sample_idx, self.eval_indices]).unsqueeze(0).repeat(len(self.eval_indices), 1).to(DEVICE)
                pred_case2 = self.case2_model(branch_input_case2, trunk_input).cpu().numpy().flatten()

                # 相対L2誤差計算
                l2_error_case1 = calculate_l2_relative_error(true_values, pred_case1)
                l2_error_case2 = calculate_l2_relative_error(true_values, pred_case2)

                case1_errors.append(l2_error_case1)
                case2_errors.append(l2_error_case2)

        self.case1_errors = np.array(case1_errors)
        self.case2_errors = np.array(case2_errors)

        print("通常誤差計算完了!")
        return self.case1_errors, self.case2_errors

    def calculate_relative_change(self):
        """(C2-C1)/C1を計算"""
        self.relative_changes = (self.case2_errors - self.case1_errors) / self.case1_errors
        return self.relative_changes

    def get_sample_predictions(self, sample_indices=[0, 1, 2]):
        """指定サンプルの予測結果を取得"""
        predictions = {}

        with torch.no_grad():
            for i, idx in enumerate(sample_indices):
                if idx >= len(self.test_sample_idx):
                    continue

                sample_idx = self.test_sample_idx[idx]

                # 真の値
                true_values = self.target_data[sample_idx, self.eval_indices]
                initial_values = self.branch_data[sample_idx]

                # Case1予測
                branch_input_case1 = torch.FloatTensor(initial_values).unsqueeze(0).repeat(len(self.eval_indices), 1).to(DEVICE)
                trunk_input = torch.FloatTensor(self.trunk_coords[self.eval_indices]).unsqueeze(1).to(DEVICE)
                pred_case1 = self.case1_model(branch_input_case1, trunk_input).cpu().numpy().flatten()

                # Case2予測
                branch_input_case2 = torch.FloatTensor(initial_values[self.eval_indices]).unsqueeze(0).repeat(len(self.eval_indices), 1).to(DEVICE)
                pred_case2 = self.case2_model(branch_input_case2, trunk_input).cpu().numpy().flatten()

                # 誤差計算
                error_case1 = calculate_l2_relative_error(true_values, pred_case1)
                error_case2 = calculate_l2_relative_error(true_values, pred_case2)
                relative_change = (error_case2 - error_case1) / error_case1

                predictions[f'sample_{idx}'] = {
                    'sample_idx': sample_idx,
                    'x_eval': self.trunk_coords[self.eval_indices],
                    'x_all': self.trunk_coords,
                    'initial_full': initial_values,
                    'true_eval': true_values,
                    'pred_case1': pred_case1,
                    'pred_case2': pred_case2,
                    'error_case1': error_case1,
                    'error_case2': error_case2,
                    'relative_change': relative_change
                }

        return predictions

    def create_analysis_plots(self):
        """分析プロットを作成"""
        fig = plt.figure(figsize=(20, 12))

        # 統計計算
        case1_stats = calculate_statistics(self.case1_errors)
        case2_stats = calculate_statistics(self.case2_errors)
        relative_change_stats = calculate_statistics(self.relative_changes)

        # プロット作成（元のコードと同じ）
        # ... [省略：元のプロット作成コードをそのまま使用]

        plt.tight_layout()

        # 保存
        plot_path = f'{self.result_dir}/enhanced_analysis_gp_mask02_smooth_{self.smoothness}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n分析結果保存: {plot_path}")

        return {
            'case1_stats': case1_stats,
            'case2_stats': case2_stats,
            'relative_change_stats': relative_change_stats,
            'sample_predictions': self.get_sample_predictions([0, 1, 2])
        }

    def print_detailed_summary(self, analysis_results):
        """詳細サマリーを出力（スケール変換テスト結果含む）"""
        # 元のサマリー出力
        # ... [省略：元のprint_detailed_summaryと同じ]
        
        # スケール変換テスト結果追加
        if hasattr(self, 'scale_results'):
            print(f"\n【スケール変換テスト結果】")
            for scale_factor, results in self.scale_results.items():
                case1_linearity = np.mean(results['case1_linearity_errors'])
                case2_linearity = np.mean(results['case2_linearity_errors'])
                
                print(f"  スケール係数 {scale_factor}:")
                print(f"    Case1線形性誤差: {case1_linearity:.4f} ({'✅' if case1_linearity < 0.1 else '❌'})")
                print(f"    Case2線形性誤差: {case2_linearity:.4f} ({'✅' if case2_linearity < 0.1 else '❌'})")
            
            print(f"\n【物理 vs 統計パターン判定】")
            avg_case1_linearity = np.mean([np.mean(results['case1_linearity_errors']) for results in self.scale_results.values()])
            avg_case2_linearity = np.mean([np.mean(results['case2_linearity_errors']) for results in self.scale_results.values()])
            
            if avg_case1_linearity < 0.1 and avg_case2_linearity < 0.1:
                print(f"  ✅ 両ケースとも線形性を保持 → 物理法則学習の可能性高")
            elif avg_case1_linearity < 0.1:
                print(f"  🟡 Case1のみ線形性保持 → 全域情報で物理法則学習")
            elif avg_case2_linearity < 0.1:
                print(f"  🟡 Case2のみ線形性保持 → マスク情報で物理法則学習")
            else:
                print(f"  ❌ 両ケースとも非線形 → 統計的パターンマッチングの可能性")

def main():
    parser = argparse.ArgumentParser(description='ガウス過程マスク率0.2専用分析（スケール変換テスト付き）')
    parser.add_argument('--smoothness', type=int, default=5, help='Gaussian Process smoothness level (0-10)')
    parser.add_argument('--scale_test', action='store_true', help='Run scale transformation test')
    parser.add_argument('--scale_factors', nargs='+', type=float, default=[0.5, 2.0], help='Scale factors for testing')
    parser.add_argument('--n_scale_samples', type=int, default=20, help='Number of samples for scale test')
    args = parser.parse_args()

    if not (0 <= args.smoothness <= 10):
        print("Warning: smoothness level should be between 0-10")

    print("="*60)
    print(f"ガウス過程マスク率0.2専用分析（拡張版）")
    print("="*60)
    print(f"Smoothness Level: {args.smoothness}")
    print(f"スケール変換テスト: {'有効' if args.scale_test else '無効'}")
    print(f"デバイス: {DEVICE}")

    try:
        # 分析器作成
        analyzer = EnhancedGPMaskRatio02Analyzer(args.smoothness)

        # モデル読み込み
        if not analyzer.load_models():
            return

        # 通常の誤差計算
        analyzer.calculate_all_errors()
        analyzer.calculate_relative_change()

        # スケール変換テスト
        if args.scale_test:
            analyzer.scale_transform_test(scale_factors=args.scale_factors, n_test_samples=args.n_scale_samples)
            analyzer.analyze_scale_results()
            analyzer.create_scale_analysis_plots()

        # 分析実行
        analysis_results = analyzer.create_analysis_plots()
        analyzer.print_detailed_summary(analysis_results)

        print("\n" + "="*60)
        print("ガウス過程分析完了!")
        print("="*60)

    except FileNotFoundError as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main()