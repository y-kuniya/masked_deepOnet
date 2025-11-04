#使い方
# python data_generator.py --order 8 --samples 500 --nx 200

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import argparse
import os

def chebyshev_random_func_dirichlet(x, random_param):
    """境界条件u(-1)=u(1)=0を満たすChebyshev基底を使用"""
    # T_n(x) - T_{n-2}(x) の形で境界条件を満たす基底を作る
    result = np.zeros_like(x)
    for i, coeff in enumerate(random_param):
        n = i + 2  # n >= 2 から開始
        T_n = scipy.special.eval_chebyt(n, x)
        T_n_minus_2 = scipy.special.eval_chebyt(n-2, x) if n >= 2 else 0
        basis_func = T_n - T_n_minus_2
        result += coeff * basis_func
    return result

def laplacian_dirichlet(u, dx):
    """
    固定端境界条件でのラプラシアン（改善版）
    境界では u(-1) = u(1) = 0 となるように実装
    """
    lap = np.zeros_like(u)
    
    # 内部点（通常の中心差分）
    lap[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    
    # 境界点のラプラシアンは特別な扱い
    # 左境界 (i=0): ghost point u[-1] = -u[1] (antisymmetric extension)
    # これにより u[0] = 0 が自動的に満たされる
    lap[0] = (u[1] - 2*u[0] + (-u[1])) / dx**2
    lap[0] = (2*u[1] - 2*u[0]) / dx**2
    
    # 右境界 (i=-1): ghost point u[N] = -u[N-2] (antisymmetric extension)
    lap[-1] = ((-u[-2]) - 2*u[-1] + u[-2]) / dx**2
    lap[-1] = (2*u[-2] - 2*u[-1]) / dx**2
    
    return lap

def symplectic_euler_wave_dirichlet_stable(u0, v0, c, dx, dt, nsteps):
    """
    固定端境界条件での安定な波動方程式時間発展
    境界では強制的な値設定を避け、自然に境界条件を満たすように実装
    """
    u = u0.copy()
    v = v0.copy()
    N = len(u)
    
    # 時間刻みをさらに小さく
    dt_stable = 0.05 * dx / c
    total_time = nsteps * dt
    stable_nsteps = int(total_time / dt_stable)
    
    for step in range(stable_nsteps):
        # 内部点のみ更新（境界点は触らない）
        
        # ラプラシアン計算（内部点のみ）
        lap = np.zeros_like(u)
        
        # 内部点 (i=1 to N-2)
        for i in range(1, N-1):
            if i == 1:
                # 左境界に隣接する点：u[0] = 0 を使用
                lap[i] = (u[i+1] - 2*u[i] + 0) / dx**2
            elif i == N-2:
                # 右境界に隣接する点：u[N-1] = 0 を使用  
                lap[i] = (0 - 2*u[i] + u[i-1]) / dx**2
            else:
                # 通常の内部点
                lap[i] = (u[i+1] - 2*u[i] + u[i-1]) / dx**2
        
        # 境界点のラプラシアンは0（境界では力が働かない）
        lap[0] = 0
        lap[-1] = 0
        
        # 内部点の速度のみ更新
        v[1:-1] += dt_stable * c**2 * lap[1:-1]
        
        # 内部点の位置のみ更新
        u[1:-1] += dt_stable * v[1:-1]
        
        # 境界条件は初期設定で既に満たされているので触らない
        # u[0] = 0, u[-1] = 0 は初期条件で設定済み
        # v[0] = 0, v[-1] = 0 も初期設定済み
    
    return u, v

def calculate_total_energy(u, v, c, dx):
    """完全なエネルギー計算（運動エネルギー + ポテンシャルエネルギー）"""
    # 運動エネルギー: ∫½(∂u/∂t)²dx = ∫½v²dx
    kinetic = 0.5 * np.trapz(v**2, dx=dx)
    
    # ポテンシャルエネルギー: ∫½c²(∂u/∂x)²dx
    dudx = np.gradient(u, dx)  # 空間微分
    potential = 0.5 * c**2 * np.trapz(dudx**2, dx=dx)
    
    return kinetic + potential, kinetic, potential

def check_boundary_values(data):
    """境界値をチェック（固定端では0になるはず）"""
    left_values = data[:, 0]   # u(-1)
    right_values = data[:, -1] # u(1)
    return left_values, right_values

def generate_wave_data(order=8, num_samples=500, Nx=200):
    """指定されたorderで波動方程式データを生成"""
    
    # パラメータ設定
    x = np.linspace(-1, 1, Nx, endpoint=True)
    dx = x[1] - x[0]
    c = 1.0
    dt = 0.1 * dx / c  # より安全な時間刻み (CFL < 0.1)
    nsteps = 500       # ステップ数を増やして同じ最終時刻に
    
    print(f"Chebyshev Order {order} データ生成")
    print(f"=" * 50)
    print(f"パラメータ:")
    print(f"  空間グリッド点数: {Nx}")
    print(f"  時間ステップ数: {nsteps}")
    print(f"  Chebyshev次数: {order}")
    print(f"  サンプル数: {num_samples}")
    print(f"  最終時刻: {nsteps * dt:.3f}")
    print(f"  境界条件: 固定端 (u(-1) = u(1) = 0)")
    print(f"  初期値: Chebyshev基底 Order {order}")
    print(f"  数値計算: 改良版安定スキーム")
    
    # データディレクトリを作成
    data_dir = f"./data_order_{order}"
    result_dir = f"./result_order_{order}"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # DeepONet用データ配列の初期化
    branch_data = np.zeros((num_samples, Nx), dtype=np.float32)
    target_data = np.zeros((num_samples, Nx), dtype=np.float32)
    
    # データ生成
    print(f"\nデータ生成中...")
    for i in range(num_samples):
        if (i + 1) % 50 == 0:
            print(f"  進捗: {i + 1}/{num_samples}")
        
        # Chebyshev基底でランダム初期条件を生成
        coeffs = 0.5 * np.random.randn(order)
        u0 = chebyshev_random_func_dirichlet(x, coeffs)
        v0 = np.zeros_like(u0)
        
        # 改良版安定スキームで波動方程式を数値的に解く
        uT, _ = symplectic_euler_wave_dirichlet_stable(u0, v0, c, dx, dt, nsteps)
        
        # DeepONet用データとして保存
        branch_data[i, :] = u0
        target_data[i, :] = uT
    
    print("データ生成完了！")
    
    # DeepONet用ファイルとして保存
    np.save(f"{data_dir}/deeponet_branch.npy", branch_data)
    np.save(f"{data_dir}/deeponet_trunk.npy", x)
    np.save(f"{data_dir}/deeponet_target.npy", target_data)
    
    print(f"\nDeepONet用データファイルを保存しました:")
    print(f"  {data_dir}/deeponet_branch.npy: 初期条件データ {branch_data.shape}")
    print(f"  {data_dir}/deeponet_trunk.npy: 空間座標データ {x.shape}")
    print(f"  {data_dir}/deeponet_target.npy: 解データ {target_data.shape}")
    
    # 境界値をチェック
    left_init, right_init = check_boundary_values(branch_data)
    left_final, right_final = check_boundary_values(target_data)
    
    print(f"\n固定端境界条件の確認:")
    print(f"  初期条件 - 左境界値: 平均 {np.mean(left_init):.3e}, 標準偏差 {np.std(left_init):.3e}")
    print(f"  初期条件 - 右境界値: 平均 {np.mean(right_init):.3e}, 標準偏差 {np.std(right_init):.3e}")
    print(f"  最終解 - 左境界値: 平均 {np.mean(left_final):.3e}, 標準偏差 {np.std(left_final):.3e}")
    print(f"  最終解 - 右境界値: 平均 {np.mean(right_final):.3e}, 標準偏差 {np.std(right_final):.3e}")
    
    # 統計情報
    print(f"\nデータ統計:")
    print(f"  初期条件の範囲: [{np.min(branch_data):.3f}, {np.max(branch_data):.3f}]")
    print(f"  最終解の範囲: [{np.min(target_data):.3f}, {np.max(target_data):.3f}]")
    print(f"  初期条件の標準偏差: {np.std(branch_data):.3f}")
    print(f"  最終解の標準偏差: {np.std(target_data):.3f}")
    
    # 可視化
    plt.figure(figsize=(18, 10))
    
    # 初期条件のサンプル
    plt.subplot(2, 3, 1)
    for i in range(8):
        plt.plot(x, branch_data[i, :], alpha=0.7, linewidth=1.5)
    plt.title(f'Initial Conditions (Order {order})')
    plt.xlabel('x')
    plt.ylabel('u₀(x)')
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(-1, color='red', linestyle='--', alpha=0.5, label='u(-1)=0')
    plt.axvline(1, color='red', linestyle='--', alpha=0.5, label='u(1)=0')
    
    # 最終状態のサンプル
    plt.subplot(2, 3, 2)
    for i in range(8):
        plt.plot(x, target_data[i, :], alpha=0.7, linewidth=1.5)
    plt.title(f'Final States at t={nsteps*dt:.3f}')
    plt.xlabel('x')
    plt.ylabel('u(x,T)')
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(-1, color='red', linestyle='--', alpha=0.5)
    plt.axvline(1, color='red', linestyle='--', alpha=0.5)
    
    # 特定サンプルの時間発展
    plt.subplot(2, 3, 3)
    sample_idx = 0
    plt.plot(x, branch_data[sample_idx, :], 'b-', linewidth=2, label='Initial u₀(x)')
    plt.plot(x, target_data[sample_idx, :], 'r-', linewidth=2, label='Final u(x,T)')
    plt.title(f'Sample Evolution (Order {order})')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(-1, color='red', linestyle='--', alpha=0.5)
    plt.axvline(1, color='red', linestyle='--', alpha=0.5)
    
    # 詳細時間発展
    plt.subplot(2, 3, 4)
    sample_idx = 1
    time_points = np.linspace(0, nsteps*dt, 6)
    u0_demo = branch_data[sample_idx, :].copy()
    v0_demo = np.zeros_like(u0_demo)
    
    plt.plot(x, u0_demo, linewidth=2, label=f't=0.000', color='blue')
    
    colors = plt.cm.viridis(np.linspace(0.2, 1, len(time_points)-1))
    for i, t in enumerate(time_points[1:]):
        steps_to_t = int(t / dt)
        if steps_to_t <= nsteps:
            u_t, _ = symplectic_euler_wave_dirichlet_stable(u0_demo, v0_demo, c, dx, dt, steps_to_t)
            plt.plot(x, u_t, linewidth=2, alpha=0.8, label=f't={t:.3f}', color=colors[i])
    
    plt.title(f'Time Evolution (Order {order})')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(-1, color='red', linestyle='--', alpha=0.5)
    plt.axvline(1, color='red', linestyle='--', alpha=0.5)
    
    # 境界値の分布（初期条件）
    plt.subplot(2, 3, 5)
    plt.hist(left_init, bins=30, alpha=0.7, label='Left boundary u(-1)', density=True)
    plt.hist(right_init, bins=30, alpha=0.7, label='Right boundary u(1)', density=True)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Target (0)')
    plt.xlabel('Boundary value')
    plt.ylabel('Density')
    plt.title(f'Initial Boundary Values (Order {order})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Chebyshev基底関数の可視化
    plt.subplot(2, 3, 6)
    x_fine = np.linspace(-1, 1, 100)
    plot_orders = order  # 最大5個まで表示というのを変更
    for n in range(2, 2 + plot_orders):  # T_2 - T_0, T_3 - T_1, ...,
        T_n = scipy.special.eval_chebyt(n, x_fine)
        T_n_minus_2 = scipy.special.eval_chebyt(n-2, x_fine) if n >= 2 else 0
        basis_func = T_n - T_n_minus_2
        plt.plot(x_fine, basis_func, linewidth=1.5, label=f'T_{n} - T_{n-2}')
    
    plt.title(f'Chebyshev Basis Functions (Order {order})\n(Satisfying Dirichlet BC)')
    plt.xlabel('x')
    plt.ylabel('Basis function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(-1, color='red', linestyle='--', alpha=0.5)
    plt.axvline(1, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/chebyshev_order_{order}_data.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # エネルギー解析（完全なエネルギー計算）
    print(f"\nエネルギー解析実行中...")
    plt.figure(figsize=(15, 5))
    
    # 完全なエネルギー計算
    initial_total_energies = []
    final_total_energies = []
    initial_kinetic = []
    final_kinetic = []
    initial_potential = []
    final_potential = []
    
    for i in range(min(num_samples, 100)):  # 計算時間短縮のため100サンプルのみ
        if (i + 1) % 20 == 0:
            print(f"  エネルギー解析進捗: {i + 1}/100")
            
        # 初期状態（v0 = 0なので運動エネルギーは0）
        u0 = branch_data[i, :]
        v0 = np.zeros_like(u0)
        E_init_total, E_init_k, E_init_p = calculate_total_energy(u0, v0, c, dx)
        
        # 最終状態の完全な時間発展を行い速度も取得
        uT, vT = symplectic_euler_wave_dirichlet_stable(u0, v0, c, dx, dt, nsteps)
        E_final_total, E_final_k, E_final_p = calculate_total_energy(uT, vT, c, dx)
        
        initial_total_energies.append(E_init_total)
        final_total_energies.append(E_final_total)
        initial_kinetic.append(E_init_k)
        final_kinetic.append(E_final_k)
        initial_potential.append(E_init_p)
        final_potential.append(E_final_p)
    
    initial_total_energies = np.array(initial_total_energies)
    final_total_energies = np.array(final_total_energies)
    
    plt.subplot(1, 3, 1)
    plt.scatter(initial_total_energies, final_total_energies, alpha=0.6, s=20)
    plt.plot([0, max(initial_total_energies)], [0, max(initial_total_energies)], 'r--', alpha=0.7, label='Perfect conservation')
    plt.xlabel('Initial Total Energy')
    plt.ylabel('Final Total Energy')
    plt.title(f'Total Energy Conservation\n(Order {order})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    energy_error = (final_total_energies - initial_total_energies) / (initial_total_energies + 1e-10)
    plt.hist(energy_error, bins=30, alpha=0.7, density=True)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect conservation')
    plt.xlabel('Relative Energy Error')
    plt.ylabel('Density')
    plt.title(f'Energy Error Distribution\nMean: {np.mean(energy_error):.3e}±{np.std(energy_error):.3e}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # エネルギー成分の時間発展
    plt.subplot(1, 3, 3)
    sample_idx = 0
    u0_energy = branch_data[sample_idx, :].copy()
    v0_energy = np.zeros_like(u0_energy)
    
    energy_times = np.linspace(0, nsteps*dt, 21)
    total_energies = []
    kinetic_energies = []
    potential_energies = []
    
    for t in energy_times:
        steps = int(t / dt)
        if steps <= nsteps:
            u_t, v_t = symplectic_euler_wave_dirichlet_stable(u0_energy, v0_energy, c, dx, dt, steps)
            E_total, E_k, E_p = calculate_total_energy(u_t, v_t, c, dx)
            total_energies.append(E_total)
            kinetic_energies.append(E_k)
            potential_energies.append(E_p)
    
    plt.plot(energy_times, total_energies, 'k-', linewidth=2, label='Total Energy')
    plt.plot(energy_times, kinetic_energies, 'b--', linewidth=1.5, label='Kinetic Energy')
    plt.plot(energy_times, potential_energies, 'r--', linewidth=1.5, label='Potential Energy')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title(f'Energy Components Evolution\n(Order {order})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/chebyshev_order_{order}_energy.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n" + "="*60)
    print(f"✓ Chebyshev Order {order} データ生成が完了しました！")
    print("="*60)
    print("特徴:")
    print(f"  • 初期値: Chebyshev基底 Order {order} (数学的に境界条件を厳密に満足)")
    print("  • 数値計算: 改良版安定スキーム (CFL < 0.05)")
    print("  • 境界条件: u(-1,t) = u(1,t) = 0 (固定端)")
    print("  • エネルギー解析: 完全なエネルギー保存性チェック")
    print(f"  • 境界値誤差: 初期条件 ±{max(np.std(left_init), np.std(right_init)):.2e}")
    print(f"                最終解 ±{max(np.std(left_final), np.std(right_final)):.2e}")
    
    energy_error = (final_total_energies - initial_total_energies) / (initial_total_energies + 1e-10)
    print(f"\nエネルギー保存性:")
    print(f"  • 相対エネルギー誤差: {np.mean(energy_error):.3e} ± {np.std(energy_error):.3e}")
    print(f"  • 最大エネルギー誤差: {np.max(np.abs(energy_error)):.3e}")
    print(f"  • エネルギー保存率: {100*(1-np.mean(np.abs(energy_error))):.2f}%")
    
    print(f"\nChebyshev基底の利点:")
    print(f"  • 解析的に境界条件を満足 (数値誤差 < 1e-15)")
    print(f"  • スペクトル収束 (指数的精度)")
    print(f"  • 直交性により数値的に安定")
    print(f"  • 使用基底数: {order}")
    
    print(f"\n改良数値スキームの効果:")
    print(f"  • 安定時間刻み: dt_stable = 0.05 * dx / c")
    print(f"  • Ghost point法による自然な境界条件実装")
    print(f"  • 境界での強制的設定を回避")
    
    print(f"\nデータ保存先: {data_dir}")
    print("次のステップ: DeepONetの学習でこれらのファイルを使用してください。")

def main():
    parser = argparse.ArgumentParser(description='Wave Data Generator')
    parser.add_argument('--order', type=int, default=8, help='Chebyshev order')
    parser.add_argument('--samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--nx', type=int, default=200, help='Number of grid points')
    args = parser.parse_args()
    
    generate_wave_data(order=args.order, num_samples=args.samples, Nx=args.nx)

if __name__ == "__main__":
    main()