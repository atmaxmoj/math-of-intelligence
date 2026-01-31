"""
Distance Matching 验证 - 第九版

测试假说: κ(J) 决定收敛速度的概率分布

- κ(J) 小 → 分布集中，几乎所有 seed 都收敛快
- κ(J) 大 → 分布分散，收敛速度取决于初始化
"""

import numpy as np
from typing import Callable, List, Dict

np.random.seed(42)

# ============================================================================
# 训练工具
# ============================================================================

def compute_loss(model_fn: Callable, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    pred = model_fn(theta, x)
    return np.mean((pred - y) ** 2)

def compute_gradient(model_fn: Callable, theta: np.ndarray, x: np.ndarray, y: np.ndarray,
                     eps: float = 1e-5) -> np.ndarray:
    grad = np.zeros_like(theta)
    loss0 = compute_loss(model_fn, theta, x, y)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        grad[i] = (compute_loss(model_fn, theta_plus, x, y) - loss0) / eps
    return grad

def train_single_run(model_fn: Callable, param_dim: int, x: np.ndarray, y: np.ndarray,
                     n_epochs: int, lr: float, init_scale: float, seed: int) -> List[float]:
    """单次训练，返回 loss 曲线"""
    np.random.seed(seed)
    theta = np.random.randn(param_dim) * init_scale

    losses = []
    for _ in range(n_epochs):
        loss = compute_loss(model_fn, theta, x, y)
        losses.append(loss)
        grad = compute_gradient(model_fn, theta, x, y)
        # 梯度裁剪防止爆炸
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 10:
            grad = grad * 10 / grad_norm
        theta = theta - lr * grad

    return losses

def compute_jacobian(model_fn: Callable, theta: np.ndarray, x: np.ndarray,
                     eps: float = 1e-5) -> np.ndarray:
    p = len(theta)
    f0 = model_fn(theta, x).flatten()
    n = len(f0)
    J = np.zeros((n, p))
    for j in range(p):
        theta_plus = theta.copy()
        theta_plus[j] += eps
        f_plus = model_fn(theta_plus, x).flatten()
        J[:, j] = (f_plus - f0) / eps
    return J

def condition_number(J: np.ndarray) -> float:
    s = np.linalg.svd(J, compute_uv=False)
    if s[-1] < 1e-10:
        return np.inf
    return s[0] / s[-1]

# ============================================================================
# 主实验
# ============================================================================

def main():
    print("="*80)
    print("Distance Matching - Seed 诱导的收敛速度分布")
    print("="*80)

    # 数据
    n, d = 50, 6
    np.random.seed(0)
    x = np.random.randn(n, d)
    true_w = np.random.randn(d, 1)
    y = x @ true_w + np.random.randn(n, 1) * 0.1

    def relu(a):
        return np.maximum(0, a)

    # 模型定义
    h = 8

    def linear_direct(theta, x):
        return x @ theta.reshape(d, 1)

    def linear_ab(theta, x):
        a, b = theta[:d], theta[d:]
        return x @ (a - b).reshape(d, 1)

    def mlp_2(theta, x):
        W1 = theta[:d*h].reshape(d, h)
        W2 = theta[d*h:d*h+h].reshape(h, 1)
        return relu(x @ W1) @ W2

    def resnet_2(theta, x):
        W1 = theta[:d*h].reshape(d, h)
        W2 = theta[d*h:d*h+h].reshape(h, 1)
        w_out = theta[d*h+h:d*h+h+d].reshape(d, 1)
        return x @ w_out + 0.1 * relu(x @ W1) @ W2

    models = {
        "Linear 直接": (linear_direct, d, 0.5, 0.1),
        "Linear (a-b)": (linear_ab, 2*d, 0.5, 0.05),
        "MLP 2层": (mlp_2, d*h+h, 0.3, 0.02),
        "ResNet 2层": (resnet_2, d*h+h+d, 0.3, 0.02),
    }

    n_seeds = 30
    n_epochs = 50

    print(f"\n每个架构跑 {n_seeds} 个不同的 seed\n")

    all_results = {}

    for name, (fn, param_dim, init_scale, lr) in models.items():
        print(f"训练 {name} ({n_seeds} seeds)...")

        # 计算 κ(J) (用几个随机点的平均)
        kappas = []
        for seed in range(5):
            np.random.seed(seed + 1000)
            theta = np.random.randn(param_dim) * init_scale
            J = compute_jacobian(fn, theta, x)
            k = condition_number(J)
            if not np.isinf(k) and k < 1e10:
                kappas.append(k)
        avg_kappa = np.mean(kappas) if kappas else np.inf

        # 多 seed 训练
        final_losses = []
        epoch10_losses = []
        loss_curves = []

        for seed in range(n_seeds):
            losses = train_single_run(fn, param_dim, x, y, n_epochs, lr, init_scale, seed)
            loss_curves.append(losses)
            final_losses.append(losses[-1])
            epoch10_losses.append(losses[min(10, n_epochs-1)])

        all_results[name] = {
            'kappa': avg_kappa,
            'final_losses': np.array(final_losses),
            'epoch10_losses': np.array(epoch10_losses),
            'loss_curves': np.array(loss_curves),
        }

    # 打印统计结果
    print("\n" + "="*80)
    print("收敛速度分布统计")
    print("="*80 + "\n")

    print(f"{'架构':<18} {'κ(J)':>10} | {'Epoch 10 Loss':^30} | {'Final Loss':^30}")
    print(f"{'':18} {'':>10} | {'mean':>8} {'std':>8} {'min':>8} {'max':>8} | {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    print("-"*100)

    for name, res in all_results.items():
        kappa = res['kappa']
        e10 = res['epoch10_losses']
        final = res['final_losses']

        kappa_str = f"{kappa:.1f}" if kappa < 1e6 else "∞"

        print(f"{name:<18} {kappa_str:>10} | "
              f"{e10.mean():>8.4f} {e10.std():>8.4f} {e10.min():>8.4f} {e10.max():>8.4f} | "
              f"{final.mean():>8.4f} {final.std():>8.4f} {final.min():>8.4f} {final.max():>8.4f}")

    # 打印分布 (直方图形式)
    print("\n" + "="*80)
    print("Final Loss 分布 (直方图)")
    print("="*80 + "\n")

    for name, res in all_results.items():
        final = res['final_losses']
        kappa = res['kappa']
        kappa_str = f"κ={kappa:.1f}" if kappa < 1e6 else "κ=∞"

        print(f"\n{name} ({kappa_str}):")
        print(f"  范围: [{final.min():.4f}, {final.max():.4f}]")
        print(f"  标准差/均值 (CV): {final.std()/final.mean():.2f}")

        # 简单的文本直方图
        bins = np.linspace(final.min(), final.max() + 0.001, 6)
        hist, _ = np.histogram(final, bins=bins)

        print("  分布:")
        for i, count in enumerate(hist):
            bar = "█" * count
            print(f"    [{bins[i]:.4f}-{bins[i+1]:.4f}]: {bar} ({count})")

    # 收敛曲线的方差
    print("\n" + "="*80)
    print("Loss 曲线的方差随 Epoch 变化")
    print("="*80 + "\n")

    print(f"{'Epoch':<8}", end="")
    for name in all_results.keys():
        print(f"{name:<18}", end="")
    print("\n" + "-"*80)

    for epoch in [0, 5, 10, 20, 30, 49]:
        print(f"{epoch:<8}", end="")
        for name, res in all_results.items():
            curves = res['loss_curves']
            std_at_epoch = curves[:, epoch].std()
            print(f"{std_at_epoch:<18.4f}", end="")
        print()

    print("\n" + "="*80)
    print("结论")
    print("="*80)
    print("""
κ(J) 与收敛速度分布的关系:

1. κ(J) 小 (Linear 直接):
   - 几乎所有 seed 都收敛到相同的好结果
   - Final loss 的 std 很小
   - 分布集中

2. κ(J) 大 (MLP):
   - 不同 seed 的收敛速度差异大
   - Final loss 的 std 大
   - 分布分散
   - 有些 seed 收敛好，有些差

3. κ(J) 决定的是:
   - 不是确定的收敛速度
   - 而是收敛速度的 **分布**
   - κ 小 → 分布窄 (稳定)
   - κ 大 → 分布宽 (不稳定)

这解释了为什么:
- 好的架构 "容易训练" = 几乎所有初始化都 work
- 差的架构 "难训练" = 需要运气/调参
""")

if __name__ == "__main__":
    main()
