"""
从 Coalgebra 结构推导参数化

问题: 对于 DFA functor H = 2 × (-)^Σ，什么参数化能 match behavioral metric?

实验:
1. 定义 soft DFA 的 behavioral metric (基于语言距离)
2. 尝试不同的参数化方式
3. 测量哪个参数化的 distance matching 最好
"""

import numpy as np
from typing import Callable, Tuple, List

np.random.seed(42)

# ============================================================================
# Soft DFA 定义
# ============================================================================

class SoftDFA:
    """
    Soft DFA with n states, alphabet {0, 1}
    - transition[a] is n×n stochastic matrix for symbol a
    - accept is n-vector of acceptance probabilities
    """
    def __init__(self, transition: List[np.ndarray], accept: np.ndarray):
        self.n_states = len(accept)
        self.transition = transition  # list of stochastic matrices
        self.accept = accept

    def run(self, word: List[int], start_dist: np.ndarray = None) -> float:
        """Run soft DFA on word, return acceptance probability"""
        if start_dist is None:
            # Start from state 0 with probability 1
            start_dist = np.zeros(self.n_states)
            start_dist[0] = 1.0

        dist = start_dist.copy()
        for symbol in word:
            dist = dist @ self.transition[symbol]

        return np.dot(dist, self.accept)

# ============================================================================
# Behavioral Metric
# ============================================================================

def behavioral_distance(dfa1: SoftDFA, dfa2: SoftDFA,
                        max_length: int = 6) -> float:
    """
    Behavioral distance based on language difference.
    d(A, B) = Σ_w 2^(-|w|) |A(w) - B(w)|
    """
    distance = 0.0

    def enumerate_words(length: int):
        if length == 0:
            yield []
        else:
            for w in enumerate_words(length - 1):
                yield w + [0]
                yield w + [1]

    for length in range(max_length + 1):
        weight = 2 ** (-length)
        for word in enumerate_words(length):
            p1 = dfa1.run(word)
            p2 = dfa2.run(word)
            distance += weight * abs(p1 - p2)

    return distance

# ============================================================================
# 参数化方式 1: 直接参数化 (logits → softmax)
# ============================================================================

def params_to_dfa_direct(theta: np.ndarray, n_states: int, n_symbols: int = 2) -> SoftDFA:
    """
    直接参数化:
    - theta[:n_states * n_states * n_symbols] = transition logits
    - theta[...] = accept logits
    """
    trans_size = n_states * n_states * n_symbols

    # Transition matrices via softmax
    trans_logits = theta[:trans_size].reshape(n_symbols, n_states, n_states)
    transitions = []
    for a in range(n_symbols):
        # Softmax over target states
        logits = trans_logits[a]
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        trans = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        transitions.append(trans)

    # Accept via sigmoid
    accept_logits = theta[trans_size:trans_size + n_states]
    accept = 1 / (1 + np.exp(-accept_logits))

    return SoftDFA(transitions, accept)

# ============================================================================
# 参数化方式 2: 分解参数化 (更多冗余)
# ============================================================================

def params_to_dfa_factored(theta: np.ndarray, n_states: int, n_symbols: int = 2) -> SoftDFA:
    """
    分解参数化: transition = softmax(A @ B^T)
    引入低秩结构/冗余
    """
    k = n_states  # 中间维度

    # A and B matrices for each symbol
    transitions = []
    idx = 0
    for a in range(n_symbols):
        A = theta[idx:idx + n_states * k].reshape(n_states, k)
        idx += n_states * k
        B = theta[idx:idx + n_states * k].reshape(n_states, k)
        idx += n_states * k

        logits = A @ B.T
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        trans = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        transitions.append(trans)

    # Accept
    accept_logits = theta[idx:idx + n_states]
    accept = 1 / (1 + np.exp(-accept_logits))

    return SoftDFA(transitions, accept)

# ============================================================================
# 参数化方式 3: 结构化参数化 (利用 functor 结构)
# ============================================================================

def params_to_dfa_structured(theta: np.ndarray, n_states: int, n_symbols: int = 2) -> SoftDFA:
    """
    结构化参数化: 分离 "结构" 和 "权重"

    想法: functor H = 2 × (-)^Σ 告诉我们结构
    - transition 矩阵应该是 "局部" 的
    - 用稀疏/带状结构
    """
    # 简化: 只允许转移到相邻状态 (带状矩阵)
    # 这减少了参数量，可能更好 match

    transitions = []
    idx = 0

    for a in range(n_symbols):
        # 每个状态只能转移到自己或相邻状态 (3个选择)
        # 对于 n 个状态，需要 3n 个参数 (边界特殊处理)
        logits = np.full((n_states, n_states), -100.0)  # 几乎为 0

        for i in range(n_states):
            # 可以转移到 i-1, i, i+1
            targets = [max(0, i-1), i, min(n_states-1, i+1)]
            targets = list(set(targets))  # 去重

            for j, t in enumerate(targets):
                if idx < len(theta):
                    logits[i, t] = theta[idx]
                    idx += 1

        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        trans = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        transitions.append(trans)

    # Accept
    accept_logits = theta[idx:idx + n_states]
    accept = 1 / (1 + np.exp(-accept_logits))

    return SoftDFA(transitions, accept)

# ============================================================================
# 实验: 比较不同参数化的 distance matching
# ============================================================================

def experiment_parameterization(param_fn: Callable, param_dim: int, n_states: int,
                                 n_pairs: int = 200, scale: float = 1.0) -> Tuple[float, float]:
    """比较参数距离和 behavioral 距离"""
    d_params = []
    d_behaviors = []

    for _ in range(n_pairs):
        theta1 = np.random.randn(param_dim) * scale
        theta2 = np.random.randn(param_dim) * scale

        dfa1 = param_fn(theta1, n_states)
        dfa2 = param_fn(theta2, n_states)

        d_p = np.linalg.norm(theta1 - theta2)
        d_b = behavioral_distance(dfa1, dfa2)

        if d_p > 1e-8:
            d_params.append(d_p)
            d_behaviors.append(d_b)

    d_params = np.array(d_params)
    d_behaviors = np.array(d_behaviors)

    corr = np.corrcoef(d_params, d_behaviors)[0, 1]
    ratios = d_behaviors / (d_params + 1e-8)
    cv = np.std(ratios) / (np.mean(ratios) + 1e-8)

    return corr, cv

def compute_kappa(param_fn: Callable, param_dim: int, n_states: int,
                  n_samples: int = 10, scale: float = 1.0) -> float:
    """计算 Jacobian 条件数"""
    kappas = []

    for seed in range(n_samples):
        np.random.seed(seed + 100)
        theta = np.random.randn(param_dim) * scale

        # 数值计算 Jacobian (参数 → behavioral outputs)
        # 用一组测试词
        test_words = [[], [0], [1], [0,0], [0,1], [1,0], [1,1],
                      [0,0,0], [0,0,1], [0,1,0], [0,1,1]]

        eps = 1e-5
        dfa0 = param_fn(theta, n_states)
        f0 = np.array([dfa0.run(w) for w in test_words])

        J = np.zeros((len(f0), param_dim))
        for j in range(param_dim):
            theta_plus = theta.copy()
            theta_plus[j] += eps
            dfa_plus = param_fn(theta_plus, n_states)
            f_plus = np.array([dfa_plus.run(w) for w in test_words])
            J[:, j] = (f_plus - f0) / eps

        s = np.linalg.svd(J, compute_uv=False)
        if s[-1] > 1e-10:
            kappas.append(s[0] / s[-1])

    return np.mean(kappas) if kappas else np.inf

# ============================================================================
# 主实验
# ============================================================================

def main():
    print("="*70)
    print("Coalgebra 参数化实验")
    print("="*70)
    print("""
问题: 对于 DFA functor H = 2 × (-)^Σ，什么参数化能 match behavioral metric?

Behavioral metric: d(A,B) = Σ_w 2^(-|w|) |A(w) - B(w)|
(基于语言差异，自然来自 functor 结构)
""")

    n_states = 4
    n_symbols = 2

    # 参数化方式
    parameterizations = [
        ("直接参数化 (softmax)",
         params_to_dfa_direct,
         n_states * n_states * n_symbols + n_states),

        ("分解参数化 (A@B^T)",
         params_to_dfa_factored,
         2 * n_states * n_states * n_symbols + n_states),

        ("结构化参数化 (稀疏)",
         params_to_dfa_structured,
         3 * n_states * n_symbols + n_states),  # 大约
    ]

    print(f"\n{'参数化方式':<30} {'参数量':>8} {'κ(J)':>10} {'Corr':>8} {'CV':>8}")
    print("-" * 70)

    for name, param_fn, param_dim in parameterizations:
        try:
            kappa = compute_kappa(param_fn, param_dim, n_states, scale=0.5)
            corr, cv = experiment_parameterization(param_fn, param_dim, n_states, scale=0.5)

            kappa_str = f"{kappa:.1f}" if kappa < 1e6 else "∞"
            print(f"{name:<30} {param_dim:>8} {kappa_str:>10} {corr:>8.3f} {cv:>8.3f}")
        except Exception as e:
            print(f"{name:<30} Error: {e}")

    print("\n" + "="*70)
    print("分析")
    print("="*70)
    print("""
假说:
- 直接参数化: 参数直接对应 functor 结构 (转移 + 接受)，应该匹配较好
- 分解参数化: 引入冗余 (A@B^T)，匹配变差
- 结构化参数化: 利用额外结构 (稀疏性)，可能更好或更差

如果 "直接参数化" 最好，说明 functor 结构自然给出了好的参数化!
""")

    # 额外实验: 不同状态数
    print("\n" + "="*70)
    print("不同状态数的影响")
    print("="*70)

    print(f"\n{'状态数':<10} {'参数量':>10} {'κ(J)':>10} {'Corr':>8} {'CV':>8}")
    print("-" * 50)

    for n_states in [2, 3, 4, 5, 6]:
        param_dim = n_states * n_states * n_symbols + n_states
        try:
            kappa = compute_kappa(params_to_dfa_direct, param_dim, n_states, scale=0.5)
            corr, cv = experiment_parameterization(params_to_dfa_direct, param_dim, n_states,
                                                    n_pairs=100, scale=0.5)
            kappa_str = f"{kappa:.1f}" if kappa < 1e6 else "∞"
            print(f"{n_states:<10} {param_dim:>10} {kappa_str:>10} {corr:>8.3f} {cv:>8.3f}")
        except Exception as e:
            print(f"{n_states:<10} Error: {e}")

if __name__ == "__main__":
    main()
