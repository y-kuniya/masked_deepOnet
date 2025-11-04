#使い方
# python gp_data_generator.py --smoothness 0 --samples 500 --nx 200
# python gp_data_generator.py --smoothness 5 --samples 500 --nx 200
# python gp_data_generator.py --smoothness 10 --samples 500 --nx 200

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.linalg import cholesky
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv

def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """RBFカーネル（ガウシアンカーネル）"""
    distances = cdist(x1.reshape(-1, 1), x2.reshape(-1, 1), 'euclidean')
    return variance * np.exp(-0.5 * (distances / length_scale) ** 2)

def matern_kernel(x1, x2, length_scale=1.0, nu=1.5, variance=1.0):
    """Matérnカーネル（滑らかさ制御）"""
    distances = cdist(x1.reshape(-1, 1), x2.reshape(-1, 1), 'euclidean')
    distances = np.maximum(distances, 1e-10)
    
    if nu == 0.5:
        K = variance * np.exp(-distances / length_scale)
    elif nu == 1.5:
        sqrt3_d_l = np.sqrt(3) * distances / length_scale
        K = variance * (1 + sqrt3_d_l) * np.exp(-sqrt3_d_l)
    elif nu == 2.5:
        sqrt5_d_l = np.sqrt(5) * distances / length_scale
        K = variance * (1 + sqrt5_d_l + (5 * distances**2) / (3 * length_scale**2)) * np.exp(-sqrt5_d_l)
    else:
        sqrt2nu_d_l = np.sqrt(2 * nu) * distances / length_scale
        sqrt2nu_d_l[sqrt2nu_d_l == 0] = 1e-10
        K = variance * (2 ** (1 - nu)) / gamma(nu) * \
            (sqrt2nu_d_l ** nu) * kv(nu, sqrt2nu_d_l)
        K[distances == 0] = variance
    
    return K

def sample_gp_with_boundary_constraints(x, kernel_func, boundary_value=0.0, n_samples=1, random_state=None):
    """境界条件u(-1) = u(1) = boundary_valueを満たすガウス過程サンプル"""
    if random_state is not None:
        np.random.seed(random_state)
    
    N = len(x)
    K = kernel_func(x, x)
    
    boundary_indices = [0, N-1]
    interior_indices = list(range(1, N-1))
    
    K_bb = K[np.ix_(boundary_indices, boundary_indices)]
    K_bi = K[np.ix_(boundary_indices, interior_indices)]
    K_ib = K[np.ix_(interior_indices, boundary_indices)]
    K_ii = K[np.ix_(interior_indices, interior_indices)]
    
    boundary_values = np.full(len(boundary_indices), boundary_value)
    
    try:
        K_bb_inv = np.linalg.inv(K_bb + 1e-6 * np.eye(len(boundary_indices)))
    except np.linalg.LinAlgError:
        K_bb_inv = np.linalg.pinv(K_bb + 1e-6 * np.eye(len(boundary_indices)))
    
    conditional_mean = K_ib @ K_bb_inv @ boundary_values
    conditional_cov = K_ii - K_ib @ K_bb_inv @ K_bi
    conditional_cov += 1e-6 * np.eye(len(interior_indices))
    
    samples = np.zeros((n_samples, N))
    
    for i in range(n_samples):
        try:
            L = cholesky(conditional_cov, lower=True)
            interior_sample = conditional_mean + L @ np.random.normal(0, 1, len(interior_indices))
        except np.linalg.LinAlgError:
            eigenvals, eigenvecs = np.linalg.eigh(conditional_cov)
            eigenvals = np.maximum(eigenvals, 1e-6)
            L = eigenvecs @ np.diag(np.sqrt(eigenvals))
            interior_sample = conditional_mean + L @ np.random.normal(0, 1, len(interior_indices))
        
        samples[i, boundary_indices] = boundary_values
        samples[i, interior_indices] = interior_sample
    
    return samples

def gp_random_func_dirichlet(x, smoothness_level, random_state=None):
    """ガウス過程による滑らかさ制御初期条件"""
    # length_scaleをsmoothness_levelで制御
    length_scale = 1.0 * np.exp(-smoothness_level * 0.23)
    variance = 1.0
    
    # カーネルタイプの選択
    if smoothness_level <= 3:
        kernel_func = lambda x1, x2: rbf_kernel(x1, x2, length_scale, variance)
        kernel_name = "RBF"
    elif smoothness_level <= 6:
        kernel_func = lambda x1, x2: matern_kernel(x1, x2, length_scale, nu=2.5, variance=variance)
        kernel_name = "Matérn 5/2"
    elif smoothness_level <= 8:
        kernel_func = lambda x1, x2: matern_kernel(x1, x2, length_scale, nu=1.5, variance=variance)
        kernel_name = "Matérn 3/2"
    else:
        kernel_func = lambda x1, x2: matern_kernel(x1, x2, length_scale, nu=0.5, variance=variance)
        kernel_name = "Matérn 1/2"
    
    # サンプリング
    samples = sample_gp_with_boundary_constraints(
        x, kernel_func, boundary_value=0.0, n_samples=1, random_state=random_state
    )
    
    return samples[0], kernel_name, length_scale

def laplacian_dirichlet(u, dx):
    """固定端境界条件でのラプラシアン（元のコードと同じ）"""
    lap = np.zeros_like(u)
    lap[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    lap[0] = (2*u[1] - 2*u[0]) / dx**2
    lap[-1] = (2*u[-2] - 2*u[-1]) / dx**2
    return lap

def symplectic_euler_wave_dirichlet_stable(u0, v0, c, dx, dt, nsteps):
    """固定端境界条件での安定な波動方程式時間発展"""
    u = u0.copy()
    v = v0.copy()
    N = len(u)
    
    dt_stable = 0.05 * dx / c
    total_time = nsteps * dt
    stable_nsteps = int(total_time / dt_stable)
    
    for step in range(stable_nsteps):
        lap = np.zeros_like(u)
        
        for i in range(1, N-1):
            if i == 1:
                lap[i] = (u[i+1] - 2*u[i] + 0) / dx**2  # u[0] = 0
            elif i == N-2:
                lap[i] = (0 - 2*u[i] + u[i-1]) / dx**2  # u[N-1] = 0
            else:
                lap[i] = (u[i+1] - 2*u[i] + u[i-1]) / dx**2
        
        lap[0] = 0
        lap[-1] = 0
        
        v[1:-1] += dt_stable * c**2 * lap[1:-1]
        u[1:-1] += dt_stable * v[1:-1]
        
        # 固定端境界条件を各ステップで強制
        u[0] = 0.0
        u[-1] = 0.0
        v[0] = 0.0
        v[-1] = 0.0
    
    return u, v

def calculate_total_energy(u, v, c, dx):
    """完全なエネルギー計算（元のコードと同じ）"""
    kinetic = 0.5 * np.trapz(v**2, dx=dx)
    dudx = np.gradient(u, dx)
    potential = 0.5 * c**2 * np.trapz(dudx**2, dx=dx)
    return kinetic + potential, kinetic, potential

def check_boundary_values(data):
    """境界値をチェック（固定端では0になるはず）"""
    left_values = data[:, 0]
    right_values = data[:, -1]
    return left_values, right_values

def generate_wave_data_gp(smoothness_level=5, num_samples=500, Nx=200):
    """ガウス過程を使った波動方程式データ生成"""
    
    # パラメータ設定
    x = np.linspace(-1, 1, Nx, endpoint=True)
    dx = x[1] - x[0]
    c = 1.0
    dt = 0.1 * dx / c
    nsteps = 400
    
    print(f"ガウス過程 Smoothness Level {smoothness_level} データ生成")
    print(f"=" * 60)
    print(f"パラメータ:")
    print(f"  空間グリッド点数: {Nx}")
    print(f"  時間ステップ数: {nsteps}")
    print(f"  滑らかさレベル: {smoothness_level} (0:滑らか → 10:急峻)")
    print(f"  サンプル数: {num_samples}")
    print(f"  最終時刻: {nsteps * dt:.3f}")
    print(f"  境界条件: 固定端 (u(-1) = u(1) = 0)")
    print(f"  初期値: ガウス過程 (滑らかさ制御)")
    
    # データディレクトリを作成
    data_dir = f"./data_gp_smooth_{smoothness_level}"
    result_dir = f"./result_gp_smooth_{smoothness_level}"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # DeepONet用データ配列の初期化
    branch_data = np.zeros((num_samples, Nx), dtype=np.float32)
    target_data = np.zeros((num_samples, Nx), dtype=np.float32)
    
    # データ生成
    print(f"\nデータ生成中...")
    kernel_names = []
    length_scales = []
    
    for i in range(num_samples):
        if (i + 1) % 50 == 0:
            print(f"  進捗: {i + 1}/{num_samples}")
        
        # ガウス過程でランダム初期条件を生成
        u0, kernel_name, length_scale = gp_random_func_dirichlet(x, smoothness_level, random_state=42+i)
        v0 = np.zeros_like(u0)
        
        # 記録（最初のサンプルのみ）
        if i == 0:
            kernel_names.append(kernel_name)
            length_scales.append(length_scale)
        
        # 波動方程式を数値的に解く
        uT, _ = symplectic_euler_wave_dirichlet_stable(u0, v0, c, dx, dt, nsteps)
        
        # DeepONet用データとして保存
        branch_data[i, :] = u0
        target_data[i, :] = uT
    
    print("データ生成完了！")
    
    # カーネル情報表示
    if kernel_names:
        print(f"\n使用カーネル情報:")
        print(f"  カーネルタイプ: {kernel_names[0]}")
        print(f"  Length Scale: {length_scales[0]:.4f}")
    
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
    print(f"  初期条件 - 左境界値: 最大|u| {np.max(np.abs(left_init)):.3e}")
    print(f"  初期条件 - 右境界値: 最大|u| {np.max(np.abs(right_init)):.3e}")
    print(f"  最終解 - 左境界値: 最大|u| {np.max(np.abs(left_final)):.3e}")
    print(f"  最終解 - 右境界値: 最大|u| {np.max(np.abs(right_final)):.3e}")
    
    # 統計情報
    print(f"\nデータ統計:")
    print(f"  初期条件の範囲: [{np.min(branch_data):.3f}, {np.max(branch_data):.3f}]")
    print(f"  最終解の範囲: [{np.min(target_data):.3f}, {np.max(target_data):.3f}]")
    print(f"  初期条件の標準偏差: {np.std(branch_data):.3f}")
    print(f"  最終解の標準偏差: {np.std(target_data):.3f}")
    
    # 滑らかさメトリクス
    gradients = []
    curvatures = []
    for i in range(min(100, num_samples)):  # 100サンプルで分析
        u = branch_data[i]
        grad = np.gradient(u, x)
        curvature = np.gradient(grad, x)
        gradients.append(np.std(grad))
        curvatures.append(np.std(curvature))
    
    print(f"\n滑らかさメトリクス (100サンプル平均):")
    print(f"  勾配の標準偏差: {np.mean(gradients):.3f}")
    print(f"  曲率の標準偏差: {np.mean(curvatures):.3f}")
    print(f"  最大隣接差分: {np.mean([np.max(np.abs(np.diff(branch_data[i]))) for i in range(min(100, num_samples))]):.3f}")
    
    # 可視化
    plt.figure(figsize=(18, 10))
    
    # 初期条件のサンプル
    plt.subplot(2, 3, 1)
    for i in range(8):
        plt.plot(x, branch_data[i, :], alpha=0.7, linewidth=1.5)
    plt.title(f'Initial Conditions (GP Smoothness {smoothness_level})')
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
    plt.title(f'Sample Evolution (GP Smoothness {smoothness_level})')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(-1, color='red', linestyle='--', alpha=0.5)
    plt.axvline(1, color='red', linestyle='--', alpha=0.5)
    
    # 勾配の分析
    plt.subplot(2, 3, 4)
    sample_gradients = [np.gradient(branch_data[i], x) for i in range(min(5, num_samples))]
    for i, grad in enumerate(sample_gradients):
        plt.plot(x, grad, alpha=0.7, label=f'Sample {i+1}')
    plt.title('Gradients of Initial Conditions')
    plt.xlabel('x')
    plt.ylabel('du/dx')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 滑らかさの分布
    plt.subplot(2, 3, 5)
    plt.hist(gradients, bins=20, alpha=0.7, density=True)
    plt.axvline(np.mean(gradients), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(gradients):.3f}')
    plt.xlabel('Gradient Standard Deviation')
    plt.ylabel('Density')
    plt.title(f'Gradient Smoothness Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # カーネル情報表示
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.7, f'Smoothness Level: {smoothness_level}', fontsize=12, transform=plt.gca().transAxes)
    if kernel_names:
        plt.text(0.1, 0.6, f'Kernel Type: {kernel_names[0]}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.5, f'Length Scale: {length_scales[0]:.4f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'Data Range: [{np.min(branch_data):.3f}, {np.max(branch_data):.3f}]', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f'Gradient Std: {np.mean(gradients):.3f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, f'Curvature Std: {np.mean(curvatures):.3f}', fontsize=12, transform=plt.gca().transAxes)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Generation Summary')
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/gp_smoothness_{smoothness_level}_data.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # エネルギー解析
    print(f"\nエネルギー解析実行中...")
    plt.figure(figsize=(15, 5))
    
    initial_total_energies = []
    final_total_energies = []
    
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
    
    initial_total_energies = np.array(initial_total_energies)
    final_total_energies = np.array(final_total_energies)
    
    plt.subplot(1, 3, 1)
    plt.scatter(initial_total_energies, final_total_energies, alpha=0.6, s=20)
    plt.plot([0, max(initial_total_energies)], [0, max(initial_total_energies)], 'r--', alpha=0.7, label='Perfect conservation')
    plt.xlabel('Initial Total Energy')
    plt.ylabel('Final Total Energy')
    plt.title(f'Total Energy Conservation\n(GP Smoothness {smoothness_level})')
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
    plt.title(f'Energy Components Evolution\n(GP Smoothness {smoothness_level})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/gp_smoothness_{smoothness_level}_energy.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # エネルギー統計の追加
    energy_error = (final_total_energies - initial_total_energies) / (initial_total_energies + 1e-10)
    print(f"\nエネルギー保存性:")
    print(f"  • 相対エネルギー誤差: {np.mean(energy_error):.3e} ± {np.std(energy_error):.3e}")
    print(f"  • 最大エネルギー誤差: {np.max(np.abs(energy_error)):.3e}")
    print(f"  • エネルギー保存率: {100*(1-np.mean(np.abs(energy_error))):.2f}%")
    
    print(f"\n" + "="*60)
    print(f"✓ ガウス過程 Smoothness Level {smoothness_level} データ生成が完了しました！")
    print("="*60)
    print("特徴:")
    print(f"  • 滑らかさレベル: {smoothness_level}")
    if kernel_names:
        print(f"  • カーネルタイプ: {kernel_names[0]}")
        print(f"  • Length Scale: {length_scales[0]:.4f}")
    print(f"  • 境界条件: u(-1,t) = u(1,t) = 0 (固定端)")
    print(f"  • 境界値誤差: < {max(np.max(np.abs(left_init)), np.max(np.abs(right_init))):.2e}")
    
    print(f"\nガウス過程の特徴:")
    print(f"  • 数学的に制御された滑らかさ")
    print(f"  • 境界条件を厳密に満足")
    print(f"  • 多様で自然な関数生成")
    
    print(f"\nデータ保存先: {data_dir}")
    print("次のステップ: このデータでDeepONetのマスク実験を実行してください。")

def main():
    parser = argparse.ArgumentParser(description='Gaussian Process Wave Data Generator with Smoothness Control')
    parser.add_argument('--smoothness', type=int, default=5, help='Smoothness level (0:smooth → 10:sharp)')
    parser.add_argument('--samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--nx', type=int, default=200, help='Number of grid points')
    args = parser.parse_args()
    
    if not (0 <= args.smoothness <= 10):
        print("Warning: smoothness level should be between 0-10")
        print("  0-3: RBF kernel (infinitely differentiable)")
        print("  4-6: Matérn 5/2 (2 times differentiable)")
        print("  7-8: Matérn 3/2 (1 time differentiable)")  
        print("  9-10: Matérn 1/2 (continuous but not differentiable)")
    
    generate_wave_data_gp(smoothness_level=args.smoothness, num_samples=args.samples, Nx=args.nx)

if __name__ == "__main__":
    main()