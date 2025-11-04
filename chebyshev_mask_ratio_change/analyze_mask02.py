#!/usr/bin/env python3
"""
マスク率0.2専用分析スクリプト
使用方法: python analyze_mask02.py --order 5
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
from config import DEVICE, MODEL_CONFIG, get_paths

class MaskRatio02Analyzer:
    """マスク率0.2専用分析クラス"""

    def __init__(self, order):
        self.order = order
        self.mask_ratio = 0.2  # 固定
        self.base_data_dir, self.result_dir = get_paths(order)

        # データ読み込み
        self.branch_data = np.load(f"{self.base_data_dir}/deeponet_branch.npy")
        self.trunk_coords = np.load(f"{self.base_data_dir}/deeponet_trunk.npy")
        self.target_data = np.load(f"{self.base_data_dir}/deeponet_target.npy")
        self.num_samples, self.Nx = self.branch_data.shape

        # 学習と同じtrain/test分割を再現
        sample_indices = np.arange(self.num_samples)
        self.train_sample_idx, self.test_sample_idx = train_test_split(
            sample_indices, test_size=0.2, random_state=42
        )

        # マスク設定
        self.eval_indices, self.unmask_start, self.unmask_end = self.create_mask_indices()

        print(f"データ読み込み完了:")
        print(f"  全サンプル数: {self.num_samples}")
        print(f"  テストサンプル数: {len(self.test_sample_idx)}")
        print(f"  グリッド点数: {self.Nx}")
        print(f"  マスク率: {self.mask_ratio} (中央部 {(1-2*self.mask_ratio)*100:.0f}%)")
        print(f"  評価領域: [{self.unmask_start}, {self.unmask_end}) ({len(self.eval_indices)}点)")

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
            return False

    def calculate_all_errors(self):
        """全テストサンプルの誤差を計算"""
        case1_errors = []
        case2_errors = []

        print("誤差計算中...")

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

        print("誤差計算完了!")
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

        # 1. 誤差分布比較（左上）
        ax1 = plt.subplot(2, 4, 1)
        bins = np.linspace(0, max(np.max(self.case1_errors), np.max(self.case2_errors)), 50)

        ax1.hist(self.case1_errors, bins=bins, alpha=0.7, label='Case1 (Full Initial)',
                color='blue', density=True, edgecolor='black', linewidth=0.5)
        ax1.hist(self.case2_errors, bins=bins, alpha=0.7, label='Case2 (Masked Initial)',
                color='orange', density=True, edgecolor='black', linewidth=0.5)

        ax1.axvline(case1_stats['mean'], color='blue', linestyle='--', linewidth=2, alpha=0.8)
        ax1.axvline(case2_stats['mean'], color='orange', linestyle='--', linewidth=2, alpha=0.8)

        ax1.set_xlabel('L2 Relative Error')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Error Distribution (Mask Ratio 0.2)\nCase1: {case1_stats["mean"]:.4f}±{case1_stats["stderr"]:.4f}\nCase2: {case2_stats["mean"]:.4f}±{case2_stats["stderr"]:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 相対変化率分布（右上）
        ax2 = plt.subplot(2, 4, 2)
        ax2.hist(self.relative_changes * 100, bins=30, alpha=0.7, color='green',
                density=True, edgecolor='black', linewidth=0.5)
        ax2.axvline(relative_change_stats['mean'] * 100, color='red', linestyle='--', linewidth=2)
        ax2.axvline(0, color='black', linestyle='-', linewidth=1)

        ax2.set_xlabel('(C2-C1)/C1 (%)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Relative Change Distribution\nMean: {relative_change_stats["mean"]*100:.2f}±{relative_change_stats["stderr"]*100:.2f}%')
        ax2.grid(True, alpha=0.3)

        # 3. 散布図（Case1 vs Case2）
        ax3 = plt.subplot(2, 4, 3)
        ax3.scatter(self.case1_errors, self.case2_errors, alpha=0.6, s=20, color='purple')

        min_err = min(np.min(self.case1_errors), np.min(self.case2_errors))
        max_err = max(np.max(self.case1_errors), np.max(self.case2_errors))
        ax3.plot([min_err, max_err], [min_err, max_err], 'r--', alpha=0.7, linewidth=2, label='y=x')

        ax3.set_xlabel('Case1 Error')
        ax3.set_ylabel('Case2 Error')
        ax3.set_title('Error Scatter Plot')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. ボックスプロット
        ax4 = plt.subplot(2, 4, 4)
        box_data = [self.case1_errors, self.case2_errors]
        box_labels = ['Case1\n(Full Initial)', 'Case2\n(Masked Initial)']

        bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax4.set_ylabel('L2 Relative Error')
        ax4.set_title('Error Distribution Comparison')
        ax4.grid(True, alpha=0.3)

        # 5-7. 3つのサンプル予測比較
        sample_predictions = self.get_sample_predictions([0, 1, 2])

        for i, (sample_key, pred_data) in enumerate(sample_predictions.items()):
            ax = plt.subplot(2, 4, 5 + i)

            x_all = pred_data['x_all']
            x_eval = pred_data['x_eval']
            initial_full = pred_data['initial_full']
            true_eval = pred_data['true_eval']
            pred_case1 = pred_data['pred_case1']
            pred_case2 = pred_data['pred_case2']

            # マスク領域を背景色で表示
            ax.axvspan(x_all[0], x_all[self.unmask_start], alpha=0.2, color='lightgray', label='Masked region')
            ax.axvspan(x_all[self.unmask_end], x_all[-1], alpha=0.2, color='lightgray')

            # 初期条件（全域）
            ax.plot(x_all, initial_full, ':', color='gray', linewidth=1, alpha=0.7, label='Initial condition')

            # 真の解（評価領域のみ）
            ax.plot(x_eval, true_eval, '-', color='black', linewidth=3, label='True solution', alpha=0.9)

            # 予測結果
            ax.plot(x_eval, pred_case1, '--', color='blue', linewidth=2, alpha=0.8,
                   label=f'Case1 (Err: {pred_data["error_case1"]:.4f})')
            ax.plot(x_eval, pred_case2, '--', color='orange', linewidth=2, alpha=0.8,
                   label=f'Case2 (Err: {pred_data["error_case2"]:.4f})')

            # マスク境界線
            ax.axvline(x_all[self.unmask_start], color='red', linestyle=':', alpha=0.7, linewidth=1)
            ax.axvline(x_all[self.unmask_end], color='red', linestyle=':', alpha=0.7, linewidth=1)

            ax.set_xlabel('Spatial Position x')
            ax.set_ylabel('u(x,T)')
            ax.set_title(f'Sample {i+1}\nΔ={(pred_data["relative_change"]*100):+.1f}%')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 8. 統計サマリー
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('off')

        summary_text = f"""
Mask Ratio 0.2 Analysis Results

[Basic Statistics]
Test samples: {len(self.test_sample_idx)}
Eval region: Central {(1-2*self.mask_ratio)*100:.0f}% ({len(self.eval_indices)} points)

[Case1 (Full Initial Conditions)]
Mean error: {case1_stats['mean']:.6f}
Std dev: {case1_stats['std']:.6f}
Std error: {case1_stats['stderr']:.6f}

[Case2 (Masked Initial Conditions)]
Mean error: {case2_stats['mean']:.6f}
Std dev: {case2_stats['std']:.6f}
Std error: {case2_stats['stderr']:.6f}

[Relative Change (C2-C1)/C1]
Mean: {relative_change_stats['mean']*100:+.2f}%
Std dev: {relative_change_stats['std']*100:.2f}%
Std error: {relative_change_stats['stderr']*100:.2f}%

[Conclusion]
{"Case2 is better" if relative_change_stats['mean'] < -0.05 else
 "Case1 is better" if relative_change_stats['mean'] > 0.05 else
 "Both are similar"}
(Mean diff: {abs(relative_change_stats['mean']*100):.1f}%)
        """

        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()

        # 保存
        plot_path = f'{self.result_dir}/analysis_mask02_order_{self.order}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n分析結果保存: {plot_path}")

        return {
            'case1_stats': case1_stats,
            'case2_stats': case2_stats,
            'relative_change_stats': relative_change_stats,
            'sample_predictions': sample_predictions
        }

    def print_detailed_summary(self, analysis_results):
        """詳細サマリーを出力"""
        case1_stats = analysis_results['case1_stats']
        case2_stats = analysis_results['case2_stats']
        relative_change_stats = analysis_results['relative_change_stats']
        sample_predictions = analysis_results['sample_predictions']

        print(f"\n{'='*60}")
        print(f"マスク率0.2分析結果詳細 (Order {self.order})")
        print(f"{'='*60}")

        print(f"\n【実験設定】")
        print(f"  マスク率: {self.mask_ratio} (両端各{self.mask_ratio*100:.0f}%をマスク)")
        print(f"  評価領域: 中央{(1-2*self.mask_ratio)*100:.0f}% ({len(self.eval_indices)}点)")
        print(f"  テストサンプル数: {len(self.test_sample_idx)}")

        print(f"\n【相対L2誤差統計】")
        print(f"  Case1 (全域初期条件):")
        print(f"    平均: {case1_stats['mean']:.6f} ± {case1_stats['stderr']:.6f} (SE)")
        print(f"    範囲: [{case1_stats['min']:.6f}, {case1_stats['max']:.6f}]")
        print(f"  Case2 (マスク初期条件):")
        print(f"    平均: {case2_stats['mean']:.6f} ± {case2_stats['stderr']:.6f} (SE)")
        print(f"    範囲: [{case2_stats['min']:.6f}, {case2_stats['max']:.6f}]")

        print(f"\n【相対変化率 (C2-C1)/C1】")
        print(f"  平均: {relative_change_stats['mean']*100:+.2f}% ± {relative_change_stats['stderr']*100:.2f}% (SE)")
        print(f"  中央値: {relative_change_stats['median']*100:+.2f}%")
        print(f"  範囲: [{relative_change_stats['min']*100:+.2f}%, {relative_change_stats['max']*100:+.2f}%]")

        # 統計的有意性の簡易判定
        mean_change = relative_change_stats['mean']
        stderr_change = relative_change_stats['stderr']
        z_score = abs(mean_change) / stderr_change if stderr_change > 0 else 0

        print(f"\n【統計的判定】")
        print(f"  Z-score: {z_score:.2f}")
        if z_score > 2:
            print(f"  → 有意な差が存在 (95%信頼度)")
        elif z_score > 1.65:
            print(f"  → 比較的有意な差 (90%信頼度)")
        else:
            print(f"  → 有意な差は認められない")

        print(f"\n【個別サンプル分析】")
        for sample_key, pred_data in sample_predictions.items():
            sample_num = sample_key.split('_')[1]
            print(f"  サンプル{int(sample_num)+1}: "
                  f"Case1={pred_data['error_case1']:.4f}, "
                  f"Case2={pred_data['error_case2']:.4f}, "
                  f"変化={pred_data['relative_change']*100:+.1f}%")

        print(f"\n【結論】")
        if mean_change < -0.05:
            print(f"  ✓ Case2（マスク初期条件）が Case1 より平均{abs(mean_change)*100:.1f}% 良い")
            print(f"  → 境界情報を除去することで汎化性能が向上")
        elif mean_change > 0.05:
            print(f"  ✗ Case1（全域初期条件）が Case2 より平均{mean_change*100:.1f}% 良い")
            print(f"  → 境界情報が予測精度向上に寄与")
        else:
            print(f"  ≈ 両手法の性能は同程度（平均差{abs(mean_change)*100:.1f}%）")
            print(f"  → マスキングの効果は限定的")

def main():
    parser = argparse.ArgumentParser(description='マスク率0.2専用分析')
    parser.add_argument('--order', type=int, default=5, help='Chebyshev order')
    args = parser.parse_args()

    print("="*60)
    print(f"マスク率0.2専用分析")
    print("="*60)
    print(f"Order: {args.order}")
    print(f"デバイス: {DEVICE}")

    # 分析器作成
    analyzer = MaskRatio02Analyzer(args.order)

    # モデル読み込み
    if not analyzer.load_models():
        print("モデル読み込みに失敗しました。先に実験を実行してください。")
        return

    # 誤差計算
    analyzer.calculate_all_errors()
    analyzer.calculate_relative_change()

    # 分析実行
    analysis_results = analyzer.create_analysis_plots()
    analyzer.print_detailed_summary(analysis_results)

    print("\n" + "="*60)
    print("分析完了!")
    print("="*60)

if __name__ == "__main__":
    main()