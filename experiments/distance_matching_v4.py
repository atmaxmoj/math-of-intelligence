"""
Distance Matching 验证 - 第四版

更精细的实验:
1. 修正 CNN vs FC (真正测试平移不变性)
2. Attention 机制
3. Batch Normalization 效果
4. 稀疏 vs 稠密
5. 线性变换的不同参数化
"""

import numpy as np
from typing import Callable, Tuple

np.random.seed(42)

# ============================================================================
# 工具函数
# ============================================================================

def param_distance(theta1: np.ndarray, theta2: np.ndarray) -> float:
    return np.linalg.norm(theta1 - theta2)

def func_distance(f: Callable, theta1: np.ndarray, theta2: np.ndarray,
                  x: np.ndarray) -> float:
    y1 = f(theta1, x)
    y2 = f(theta2, x)
    return np.linalg.norm(y1 - y2)

def experiment(model_fn: Callable, param_dim: int, x: np.ndarray,
               n_pairs: int = 300, scale: float = 1.0) -> Tuple[float, float]:
    d_params = []
    d_funcs = []

    for _ in range(n_pairs):
        theta1 = np.random.randn(param_dim) * scale
        theta2 = np.random.randn(param_dim) * scale

        d_p = param_distance(theta1, theta2)
        d_f = func_distance(model_fn, theta1, theta2, x)

        if d_p > 1e-8:
            d_params.append(d_p)
            d_funcs.append(d_f)

    d_params = np.array(d_params)
    d_funcs = np.array(d_funcs)

    corr = np.corrcoef(d_params, d_funcs)[0, 1]
    ratios = d_funcs / (d_params + 1e-8)
    cv = np.std(ratios) / (np.mean(ratios) + 1e-8)

    return corr, cv

def print_result(name: str, corr: float, cv: float):
    if corr > 0.9 and cv < 0.2:
        grade = "★★★"
    elif corr > 0.7 and cv < 0.4:
        grade = "★★"
    elif corr > 0.5:
        grade = "★"
    else:
        grade = "✗"
    print(f"  {name:<40} corr={corr:.3f}  cv={cv:.3f}  {grade}")

# ============================================================================
# 实验 1: 修正 CNN vs FC (用真正平移不变的目标)
# ============================================================================

def exp1_cnn_vs_fc_fixed():
    print("\n" + "="*70)
    print("实验 1: CNN vs FC (修正版 - 测试真正的平移不变结构)")
    print("="*70)
    print("关键: 让输入本身具有平移结构，CNN 应该能利用这个结构")

    n, seq_len = 100, 32
    kernel_size = 5

    # 生成具有局部相关性的输入 (模拟图像的局部结构)
    x = np.zeros((n, seq_len))
    for i in range(n):
        # 每个样本有几个局部特征
        for _ in range(3):
            pos = np.random.randint(0, seq_len - kernel_size)
            pattern = np.random.randn(kernel_size)
            x[i, pos:pos+kernel_size] += pattern

    # CNN: kernel 直接参数化
    def cnn(theta, x):
        k = theta[:kernel_size]
        out_len = seq_len - kernel_size + 1
        output = np.zeros((n, out_len))
        for i in range(out_len):
            output[:, i] = np.sum(x[:, i:i+kernel_size] * k, axis=1)
        return output.mean(axis=1, keepdims=True)

    corr, cv = experiment(cnn, kernel_size, x, scale=1.0)
    print_result("CNN (5 params)", corr, cv)

    # FC: 参数量相同，但不利用平移结构
    def fc_same_params(theta, x):
        # 只用 kernel_size 个参数，但放在固定位置
        W = np.zeros(seq_len)
        W[:kernel_size] = theta[:kernel_size]
        return (x @ W).reshape(-1, 1)

    corr, cv = experiment(fc_same_params, kernel_size, x, scale=1.0)
    print_result("FC-same-params (5 params, 固定位置)", corr, cv)

    # FC: 完整参数
    def fc_full(theta, x):
        W = theta[:seq_len]
        return (x @ W).reshape(-1, 1)

    corr, cv = experiment(fc_full, seq_len, x, scale=1.0)
    print_result("FC-full (32 params)", corr, cv)

    # 关键测试: Toeplitz 参数化 (强制平移不变)
    def toeplitz_conv(theta, x):
        """用 Toeplitz 矩阵实现卷积，和 CNN 等价"""
        k = theta[:kernel_size]
        out_len = seq_len - kernel_size + 1
        # 构建 Toeplitz 矩阵
        W = np.zeros((seq_len, out_len))
        for i in range(out_len):
            W[i:i+kernel_size, i] = k
        out = x @ W
        return out.mean(axis=1, keepdims=True)

    corr, cv = experiment(toeplitz_conv, kernel_size, x, scale=1.0)
    print_result("Toeplitz-conv (5 params, 等价 CNN)", corr, cv)

# ============================================================================
# 实验 2: Attention 的 distance matching
# ============================================================================

def exp2_attention():
    print("\n" + "="*70)
    print("实验 2: Attention 机制")
    print("="*70)

    n, seq_len, d_model = 20, 8, 4

    # 随机输入
    x = np.random.randn(n, seq_len, d_model)
    x_flat = x.reshape(n, -1)  # for comparison

    def softmax(x, axis=-1):
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    # 简单 self-attention (单头)
    def self_attention(theta, x_input):
        # 解包 Q, K, V 矩阵
        Wq = theta[:d_model*d_model].reshape(d_model, d_model)
        Wk = theta[d_model*d_model:2*d_model*d_model].reshape(d_model, d_model)
        Wv = theta[2*d_model*d_model:3*d_model*d_model].reshape(d_model, d_model)

        results = []
        for sample in x_input.reshape(n, seq_len, d_model):
            Q = sample @ Wq
            K = sample @ Wk
            V = sample @ Wv

            scores = Q @ K.T / np.sqrt(d_model)
            attn = softmax(scores, axis=-1)
            out = attn @ V
            results.append(out.mean())

        return np.array(results).reshape(-1, 1)

    param_dim = 3 * d_model * d_model
    corr, cv = experiment(lambda t, _: self_attention(t, x), param_dim, x_flat, scale=0.3)
    print_result("Self-Attention (QKV)", corr, cv)

    # 对比: 简单线性
    def linear_pool(theta, x_input):
        W = theta[:d_model].reshape(d_model, 1)
        results = []
        for sample in x_input.reshape(n, seq_len, d_model):
            out = sample @ W
            results.append(out.mean())
        return np.array(results).reshape(-1, 1)

    corr, cv = experiment(lambda t, _: linear_pool(t, x), d_model, x_flat, scale=1.0)
    print_result("Linear-pool", corr, cv)

    # 对比: 双线性 (类似 attention 但无 softmax)
    def bilinear(theta, x_input):
        W = theta[:d_model*d_model].reshape(d_model, d_model)
        results = []
        for sample in x_input.reshape(n, seq_len, d_model):
            # x^T W x 的平均
            out = np.sum(sample @ W * sample)
            results.append(out)
        return np.array(results).reshape(-1, 1)

    corr, cv = experiment(lambda t, _: bilinear(t, x), d_model*d_model, x_flat, scale=0.3)
    print_result("Bilinear (无 softmax)", corr, cv)

# ============================================================================
# 实验 3: Normalization 的影响
# ============================================================================

def exp3_normalization():
    print("\n" + "="*70)
    print("实验 3: Normalization 的影响")
    print("="*70)

    n, d = 100, 10
    x = np.random.randn(n, d)
    hidden = 16

    def relu(x):
        return np.maximum(0, x)

    # 无 normalization
    def mlp_plain(theta, x):
        w1 = theta[:d*hidden].reshape(d, hidden)
        w2 = theta[d*hidden:d*hidden+hidden].reshape(hidden, 1)
        h = relu(x @ w1)
        return h @ w2

    param_dim = d*hidden + hidden
    corr, cv = experiment(mlp_plain, param_dim, x, scale=0.3)
    print_result("MLP (无 norm)", corr, cv)

    # 带 Layer Norm (简化版)
    def mlp_layernorm(theta, x):
        w1 = theta[:d*hidden].reshape(d, hidden)
        w2 = theta[d*hidden:d*hidden+hidden].reshape(hidden, 1)

        h = x @ w1
        # Layer norm
        h = (h - h.mean(axis=1, keepdims=True)) / (h.std(axis=1, keepdims=True) + 1e-5)
        h = relu(h)
        return h @ w2

    corr, cv = experiment(mlp_layernorm, param_dim, x, scale=0.3)
    print_result("MLP + LayerNorm", corr, cv)

    # 带权重归一化 (weight normalization)
    def mlp_weightnorm(theta, x):
        v1 = theta[:d*hidden].reshape(d, hidden)
        g1 = theta[d*hidden:d*hidden+hidden]  # scales
        w2 = theta[d*hidden+hidden:d*hidden+2*hidden].reshape(hidden, 1)

        # Weight normalization: w = g * v / ||v||
        v_norm = np.linalg.norm(v1, axis=0, keepdims=True) + 1e-5
        w1 = g1 * v1 / v_norm

        h = relu(x @ w1)
        return h @ w2

    param_dim = d*hidden + 2*hidden
    corr, cv = experiment(mlp_weightnorm, param_dim, x, scale=0.3)
    print_result("MLP + WeightNorm", corr, cv)

# ============================================================================
# 实验 4: 稀疏 vs 稠密
# ============================================================================

def exp4_sparse_vs_dense():
    print("\n" + "="*70)
    print("实验 4: 稀疏 vs 稠密")
    print("="*70)

    n, d = 100, 20
    x = np.random.randn(n, d)

    # 稠密线性
    def dense_linear(theta, x):
        return x @ theta

    corr, cv = experiment(dense_linear, d, x, scale=1.0)
    print_result("Dense linear (20 params)", corr, cv)

    # 稀疏线性 (只有部分权重非零)
    def sparse_linear_5(theta, x):
        # 只用 5 个参数，其他固定为 0
        W = np.zeros(d)
        W[::4] = theta[:5]  # 每隔 4 个放一个
        return x @ W

    corr, cv = experiment(sparse_linear_5, 5, x, scale=1.0)
    print_result("Sparse linear (5 params, 固定位置)", corr, cv)

    # 学习稀疏模式的代理 (用 mask 参数化)
    def soft_sparse(theta, x):
        """用 sigmoid 软化稀疏性"""
        W = theta[:d]
        mask_logits = theta[d:2*d]
        mask = 1 / (1 + np.exp(-mask_logits))  # sigmoid
        return x @ (W * mask)

    corr, cv = experiment(soft_sparse, 2*d, x, scale=1.0)
    print_result("Soft-sparse (40 params, W*sigmoid(m))", corr, cv)

# ============================================================================
# 实验 5: 不同矩阵分解
# ============================================================================

def exp5_matrix_factorization():
    print("\n" + "="*70)
    print("实验 5: 不同矩阵分解方式")
    print("="*70)

    n, d = 100, 10
    x = np.random.randn(n, d)

    # 直接线性
    def direct(theta, x):
        return x @ theta

    corr, cv = experiment(direct, d, x, scale=1.0)
    print_result("Direct linear", corr, cv)

    # 低秩分解: W = A @ B^T，其中 A, B 都是 d x k
    for k in [2, 5, 10]:
        def low_rank(theta, x, k=k):
            A = theta[:d*k].reshape(d, k)
            B = theta[d*k:2*d*k].reshape(d, k)
            W = A @ B.T
            # 只取对角线作为权重 (否则输出维度不对)
            w = np.diag(W)
            return x @ w

        param_dim = 2 * d * k
        corr, cv = experiment(low_rank, param_dim, x, scale=0.3)
        print_result(f"Low-rank (k={k}, {param_dim} params)", corr, cv)

    # SVD 参数化: W = U @ diag(s) @ V^T
    def svd_param(theta, x):
        k = 5
        U = theta[:d*k].reshape(d, k)
        s = theta[d*k:d*k+k]
        V = theta[d*k+k:2*d*k+k].reshape(d, k)
        W = U @ np.diag(s) @ V.T
        w = np.diag(W)
        return x @ w

    param_dim = 2*d*5 + 5
    corr, cv = experiment(svd_param, param_dim, x, scale=0.3)
    print_result(f"SVD param (U@diag(s)@V^T)", corr, cv)

# ============================================================================
# 实验 6: 组合实验 - Skip + Norm
# ============================================================================

def exp6_combinations():
    print("\n" + "="*70)
    print("实验 6: 架构组合 (Skip + Norm)")
    print("="*70)

    n, d = 100, 10
    x = np.random.randn(n, d)

    def relu(x):
        return np.maximum(0, x)

    # Plain MLP
    def plain(theta, x):
        w1 = theta[:d*d].reshape(d, d)
        w2 = theta[d*d:d*d+d].reshape(d, 1)
        h = relu(x @ w1)
        return h @ w2

    param_dim = d*d + d
    corr, cv = experiment(plain, param_dim, x, scale=0.3)
    print_result("Plain MLP", corr, cv)

    # Skip only
    def skip_only(theta, x):
        w1 = theta[:d*d].reshape(d, d)
        w2 = theta[d*d:d*d+d].reshape(d, 1)
        h = x + 0.1 * relu(x @ w1)  # skip
        return h @ w2

    corr, cv = experiment(skip_only, param_dim, x, scale=0.3)
    print_result("Skip only (x + 0.1*relu)", corr, cv)

    # LayerNorm only
    def norm_only(theta, x):
        w1 = theta[:d*d].reshape(d, d)
        w2 = theta[d*d:d*d+d].reshape(d, 1)
        h = x @ w1
        h = (h - h.mean(axis=1, keepdims=True)) / (h.std(axis=1, keepdims=True) + 1e-5)
        h = relu(h)
        return h @ w2

    corr, cv = experiment(norm_only, param_dim, x, scale=0.3)
    print_result("LayerNorm only", corr, cv)

    # Skip + LayerNorm
    def skip_and_norm(theta, x):
        w1 = theta[:d*d].reshape(d, d)
        w2 = theta[d*d:d*d+d].reshape(d, 1)
        h = x @ w1
        h = (h - h.mean(axis=1, keepdims=True)) / (h.std(axis=1, keepdims=True) + 1e-5)
        h = x + 0.1 * relu(h)  # skip + norm
        return h @ w2

    corr, cv = experiment(skip_and_norm, param_dim, x, scale=0.3)
    print_result("Skip + LayerNorm", corr, cv)

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*70)
    print("Distance Matching 验证实验 - 第四版")
    print("="*70)
    print("""
更精细的实验，修正之前的问题，测试更多架构变体。

指标解释:
  - Correlation: 参数距离和函数距离的相关性 (越高越好)
  - CV: 比值的变异系数 (越低越好)
  - ★★★ = 优秀, ★★ = 良好, ★ = 一般, ✗ = 差
""")

    exp1_cnn_vs_fc_fixed()
    exp2_attention()
    exp3_normalization()
    exp4_sparse_vs_dense()
    exp5_matrix_factorization()
    exp6_combinations()

    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("""
关键发现:

1. CNN vs FC (修正版)
   - CNN 和 Toeplitz-conv 应该相同 (都是平移不变参数化)
   - FC-full 有更多参数但不一定更好匹配

2. Attention
   - Softmax 引入非线性，可能影响 distance matching
   - QKV 三组参数有冗余?

3. Normalization
   - LayerNorm 可能改变距离关系 (scale 信息丢失)
   - WeightNorm 保持方向信息

4. 稀疏性
   - 固定稀疏模式: 参数更少，可能更好匹配
   - 软稀疏 (sigmoid mask): 引入冗余

5. 矩阵分解
   - 低秩分解: 冗余，距离匹配变差
   - 秩越高，冗余越多

6. 组合
   - Skip connection 可能改善稳定性
   - LayerNorm 可能有不确定影响
   - 组合效果需要具体看
""")

if __name__ == "__main__":
    main()
