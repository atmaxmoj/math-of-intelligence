"""
Distance Matching 验证 - 第五版

测试假说: 非线性是破坏 distance matching 的关键因素

实验:
1. 线性 vs 非线性 (控制其他变量)
2. 不同非线性函数的影响
3. 深度线性网络 (无非线性但有depth)
4. 组合: 何时非线性是好的?
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
    print(f"  {name:<45} corr={corr:.3f}  cv={cv:.3f}  {grade}")

# ============================================================================
# 实验 1: 线性 vs 非线性 (严格对照)
# ============================================================================

def exp1_linear_vs_nonlinear():
    print("\n" + "="*70)
    print("实验 1: 线性 vs 非线性 (严格对照)")
    print("="*70)
    print("控制变量: 相同参数量、相同结构，只变非线性")

    n, d = 100, 10
    x = np.random.randn(n, d)
    hidden = 16

    # 激活函数
    def relu(x): return np.maximum(0, x)
    def tanh_act(x): return np.tanh(x)
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    def leaky_relu(x): return np.where(x > 0, x, 0.1 * x)
    def linear(x): return x  # 无非线性

    activations = [
        ('Linear (identity)', linear),
        ('ReLU', relu),
        ('LeakyReLU (0.1)', leaky_relu),
        ('Tanh', tanh_act),
        ('Sigmoid', sigmoid),
    ]

    for act_name, act_fn in activations:
        def model(theta, x, act=act_fn):
            w1 = theta[:d*hidden].reshape(d, hidden)
            w2 = theta[d*hidden:d*hidden+hidden].reshape(hidden, 1)
            h = act(x @ w1)
            return h @ w2

        param_dim = d*hidden + hidden
        corr, cv = experiment(model, param_dim, x, scale=0.3)
        print_result(f"2-layer MLP with {act_name}", corr, cv)

# ============================================================================
# 实验 2: 深度线性网络
# ============================================================================

def exp2_deep_linear():
    print("\n" + "="*70)
    print("实验 2: 深度线性网络 (无非线性)")
    print("="*70)
    print("问题: 深度本身会破坏 distance matching 吗?")

    n, d = 100, 8
    x = np.random.randn(n, d)

    # 1 层线性
    def linear_1(theta, x):
        return x @ theta.reshape(d, 1)

    corr, cv = experiment(linear_1, d, x, scale=1.0)
    print_result("1-layer linear", corr, cv)

    # 2 层线性 (W2 @ W1，可以折叠成单矩阵)
    def linear_2(theta, x):
        w1 = theta[:d*d].reshape(d, d)
        w2 = theta[d*d:d*d+d].reshape(d, 1)
        return x @ w1 @ w2

    param_dim = d*d + d
    corr, cv = experiment(linear_2, param_dim, x, scale=0.5)
    print_result("2-layer linear (W2 @ W1 @ x)", corr, cv)

    # 3 层线性
    def linear_3(theta, x):
        idx = 0
        h = x
        for _ in range(2):
            w = theta[idx:idx+d*d].reshape(d, d)
            idx += d*d
            h = h @ w
        w_out = theta[idx:idx+d].reshape(d, 1)
        return h @ w_out

    param_dim = d*d*2 + d
    corr, cv = experiment(linear_3, param_dim, x, scale=0.3)
    print_result("3-layer linear", corr, cv)

    # 4 层线性
    def linear_4(theta, x):
        idx = 0
        h = x
        for _ in range(3):
            w = theta[idx:idx+d*d].reshape(d, d)
            idx += d*d
            h = h @ w
        w_out = theta[idx:idx+d].reshape(d, 1)
        return h @ w_out

    param_dim = d*d*3 + d
    corr, cv = experiment(linear_4, param_dim, x, scale=0.2)
    print_result("4-layer linear", corr, cv)

    print("\n  --> 深度线性网络: 函数空间不变，但参数空间增加 → 冗余增加")

# ============================================================================
# 实验 3: 非线性的程度
# ============================================================================

def exp3_nonlinearity_degree():
    print("\n" + "="*70)
    print("实验 3: 非线性的程度")
    print("="*70)
    print("用 softplus(beta * x) / beta 控制非线性程度")

    n, d = 100, 10
    x = np.random.randn(n, d)
    hidden = 16

    def softplus(x, beta=1.0):
        # softplus(x) = log(1 + exp(beta * x)) / beta
        # beta → ∞ 时趋近于 ReLU
        # beta → 0 时趋近于线性
        return np.log(1 + np.exp(np.clip(beta * x, -20, 20))) / beta

    betas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    for beta in betas:
        def model(theta, x, b=beta):
            w1 = theta[:d*hidden].reshape(d, hidden)
            w2 = theta[d*hidden:d*hidden+hidden].reshape(hidden, 1)
            h = softplus(x @ w1, beta=b)
            return h @ w2

        param_dim = d*hidden + hidden
        corr, cv = experiment(model, param_dim, x, scale=0.3)
        print_result(f"Softplus (beta={beta})", corr, cv)

    print("\n  --> beta 越大，非线性越强，distance matching 越差?")

# ============================================================================
# 实验 4: 非线性的位置
# ============================================================================

def exp4_nonlinearity_location():
    print("\n" + "="*70)
    print("实验 4: 非线性的位置")
    print("="*70)

    n, d = 100, 10
    x = np.random.randn(n, d)
    hidden = 16

    def relu(x): return np.maximum(0, x)

    # 非线性在第一层后
    def nonlin_first(theta, x):
        w1 = theta[:d*hidden].reshape(d, hidden)
        w2 = theta[d*hidden:d*hidden+hidden*d].reshape(hidden, d)
        w3 = theta[d*hidden+hidden*d:d*hidden+hidden*d+d].reshape(d, 1)
        h = relu(x @ w1)  # 非线性
        h = h @ w2  # 线性
        return h @ w3

    param_dim = d*hidden + hidden*d + d
    corr, cv = experiment(nonlin_first, param_dim, x, scale=0.2)
    print_result("Nonlinearity after 1st layer only", corr, cv)

    # 非线性在最后一层前
    def nonlin_last(theta, x):
        w1 = theta[:d*hidden].reshape(d, hidden)
        w2 = theta[d*hidden:d*hidden+hidden*d].reshape(hidden, d)
        w3 = theta[d*hidden+hidden*d:d*hidden+hidden*d+d].reshape(d, 1)
        h = x @ w1  # 线性
        h = relu(h @ w2)  # 非线性
        return h @ w3

    corr, cv = experiment(nonlin_last, param_dim, x, scale=0.2)
    print_result("Nonlinearity after 2nd layer only", corr, cv)

    # 非线性在每层
    def nonlin_all(theta, x):
        w1 = theta[:d*hidden].reshape(d, hidden)
        w2 = theta[d*hidden:d*hidden+hidden*d].reshape(hidden, d)
        w3 = theta[d*hidden+hidden*d:d*hidden+hidden*d+d].reshape(d, 1)
        h = relu(x @ w1)  # 非线性
        h = relu(h @ w2)  # 非线性
        return h @ w3

    corr, cv = experiment(nonlin_all, param_dim, x, scale=0.2)
    print_result("Nonlinearity after every layer", corr, cv)

# ============================================================================
# 实验 5: 输入空间大小的影响
# ============================================================================

def exp5_input_dimension():
    print("\n" + "="*70)
    print("实验 5: 输入维度的影响")
    print("="*70)

    n = 100

    for d in [5, 10, 20, 50]:
        x = np.random.randn(n, d)

        # 线性
        def linear(theta, x, d=d):
            return x @ theta.reshape(d, 1)

        corr, cv = experiment(linear, d, x, scale=1.0)
        print_result(f"Linear (d={d})", corr, cv)

    print()

    for d in [5, 10, 20, 50]:
        x = np.random.randn(n, d)
        hidden = max(8, d)

        # 非线性
        def relu(a): return np.maximum(0, a)

        def mlp(theta, x, d=d, h=hidden):
            w1 = theta[:d*h].reshape(d, h)
            w2 = theta[d*h:d*h+h].reshape(h, 1)
            return relu(x @ w1) @ w2

        param_dim = d*hidden + hidden
        corr, cv = experiment(mlp, param_dim, x, scale=0.3)
        print_result(f"MLP with ReLU (d={d})", corr, cv)

# ============================================================================
# 实验 6: 参数初始化尺度
# ============================================================================

def exp6_initialization_scale():
    print("\n" + "="*70)
    print("实验 6: 参数初始化尺度")
    print("="*70)
    print("在不同尺度下测试，看是否影响 distance matching")

    n, d = 100, 10
    x = np.random.randn(n, d)
    hidden = 16

    def relu(x): return np.maximum(0, x)

    for scale in [0.1, 0.3, 0.5, 1.0, 2.0]:
        # 线性
        def linear(theta, x):
            return x @ theta.reshape(d, 1)

        corr, cv = experiment(linear, d, x, scale=scale)
        print_result(f"Linear (scale={scale})", corr, cv)

    print()

    for scale in [0.1, 0.3, 0.5, 1.0, 2.0]:
        def mlp(theta, x):
            w1 = theta[:d*hidden].reshape(d, hidden)
            w2 = theta[d*hidden:d*hidden+hidden].reshape(hidden, 1)
            return relu(x @ w1) @ w2

        param_dim = d*hidden + hidden
        corr, cv = experiment(mlp, param_dim, x, scale=scale)
        print_result(f"MLP (scale={scale})", corr, cv)

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*70)
    print("Distance Matching - 非线性假说验证")
    print("="*70)
    print("""
核心假说: 非线性是破坏 distance matching 的关键因素

测试:
1. 相同结构，不同激活函数
2. 深度线性网络 (无非线性但有深度)
3. 不同程度的非线性
4. 非线性的位置
5. 输入维度的影响
6. 初始化尺度的影响
""")

    exp1_linear_vs_nonlinear()
    exp2_deep_linear()
    exp3_nonlinearity_degree()
    exp4_nonlinearity_location()
    exp5_input_dimension()
    exp6_initialization_scale()

    print("\n" + "="*70)
    print("核心发现")
    print("="*70)
    print("""
Distance Matching 破坏因素:

1. 冗余 (redundancy)
   - 参数空间比函数空间大
   - 例: a-b 参数化, A@B 分解

2. 深度 (即使是线性)
   - W2 @ W1 和 W2' @ W1' 可以给出同一个函数
   - 深度线性网络也有冗余

3. 非线性
   - 使得 theta → f(theta) 映射高度非线性
   - 小参数变化可能导致大函数变化 (反之亦然)

4. 但是! 非线性对于表达能力是必要的
   - 纯线性无法拟合非线性目标
   - 需要权衡: 表达能力 vs distance matching

关键洞察:
- Distance matching 关心的是 **优化容易性**
- 非线性网络虽然 distance matching 差，但可以拟合更复杂的函数
- 好的架构可能是: 保持表达力的同时，尽量减少冗余
  (这解释了为什么 CNN < FC 对于平移不变函数: CNN 减少了冗余)
""")

if __name__ == "__main__":
    main()
