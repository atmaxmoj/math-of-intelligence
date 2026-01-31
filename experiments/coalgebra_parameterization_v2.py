"""
Coalgebra 参数化 - 第二版

新假说: 参数空间的度量应该来自 functor 结构

对于 DFA:
- Transitions 是 stochastic matrices (住在 simplex)
- Simplex 的自然度量是 Fisher-Rao (或 Hellinger)
- 不是 Euclidean!

实验: 用不同的参数空间度量，看哪个 match behavioral metric
"""

import numpy as np
from typing import Callable, Tuple, List

np.random.seed(42)

# ============================================================================
# Soft DFA (复用之前的)
# ============================================================================

class SoftDFA:
    def __init__(self, transition: List[np.ndarray], accept: np.ndarray):
        self.n_states = len(accept)
        self.transition = transition
        self.accept = accept

    def run(self, word: List[int]) -> float:
        dist = np.zeros(self.n_states)
        dist[0] = 1.0
        for symbol in word:
            dist = dist @ self.transition[symbol]
        return np.dot(dist, self.accept)

def behavioral_distance(dfa1: SoftDFA, dfa2: SoftDFA, max_length: int = 5) -> float:
    distance = 0.0
    def enumerate_words(length):
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
# 不同的参数空间度量
# ============================================================================

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """标准 Euclidean 距离"""
    return np.linalg.norm(p1 - p2)

def hellinger_distance_simplex(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Hellinger 距离 for probability vectors
    H(p, q) = sqrt(1 - sum(sqrt(p_i * q_i)))

    这是 simplex 上的自然度量之一
    """
    return np.sqrt(1 - np.sum(np.sqrt(p1 * p2 + 1e-10)))

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence (不对称)"""
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))

def fisher_rao_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Fisher-Rao 距离 (测地线距离 on simplex)
    d_FR(p, q) = 2 * arccos(sum(sqrt(p_i * q_i)))
    """
    inner = np.sum(np.sqrt(p1 * p2 + 1e-10))
    inner = np.clip(inner, -1, 1)
    return 2 * np.arccos(inner)

# ============================================================================
# 直接在概率空间参数化
# ============================================================================

def random_stochastic_matrix(n: int) -> np.ndarray:
    """生成随机 stochastic matrix"""
    M = np.random.dirichlet(np.ones(n), size=n)
    return M

def random_soft_dfa(n_states: int, n_symbols: int = 2) -> Tuple[SoftDFA, np.ndarray]:
    """
    直接在概率空间采样 soft DFA
    返回 (dfa, flattened_params)
    """
    transitions = []
    params = []

    for _ in range(n_symbols):
        T = random_stochastic_matrix(n_states)
        transitions.append(T)
        params.append(T.flatten())

    accept = np.random.rand(n_states)
    params.append(accept)

    return SoftDFA(transitions, accept), np.concatenate(params)

def dfa_param_distance_euclidean(params1: np.ndarray, params2: np.ndarray) -> float:
    """Euclidean 距离 on flattened params"""
    return np.linalg.norm(params1 - params2)

def dfa_param_distance_hellinger(dfa1: SoftDFA, dfa2: SoftDFA) -> float:
    """
    Hellinger 距离: 对每个转移分布求 Hellinger，然后求和
    这是 "结构化" 的度量
    """
    total = 0.0

    # Transition matrices
    for T1, T2 in zip(dfa1.transition, dfa2.transition):
        for i in range(dfa1.n_states):
            total += hellinger_distance_simplex(T1[i], T2[i]) ** 2

    # Accept (treat as Bernoulli)
    for a1, a2 in zip(dfa1.accept, dfa2.accept):
        p1 = np.array([a1, 1-a1])
        p2 = np.array([a2, 1-a2])
        total += hellinger_distance_simplex(p1, p2) ** 2

    return np.sqrt(total)

def dfa_param_distance_fisher(dfa1: SoftDFA, dfa2: SoftDFA) -> float:
    """Fisher-Rao 距离"""
    total = 0.0

    for T1, T2 in zip(dfa1.transition, dfa2.transition):
        for i in range(dfa1.n_states):
            total += fisher_rao_distance(T1[i], T2[i]) ** 2

    for a1, a2 in zip(dfa1.accept, dfa2.accept):
        p1 = np.array([a1 + 1e-10, 1-a1 + 1e-10])
        p2 = np.array([a2 + 1e-10, 1-a2 + 1e-10])
        total += fisher_rao_distance(p1, p2) ** 2

    return np.sqrt(total)

# ============================================================================
# 实验
# ============================================================================

def experiment(n_states: int, n_pairs: int = 300):
    """比较不同度量的 distance matching"""

    d_behavior = []
    d_euclidean = []
    d_hellinger = []
    d_fisher = []

    for _ in range(n_pairs):
        dfa1, params1 = random_soft_dfa(n_states)
        dfa2, params2 = random_soft_dfa(n_states)

        d_b = behavioral_distance(dfa1, dfa2)
        d_e = dfa_param_distance_euclidean(params1, params2)
        d_h = dfa_param_distance_hellinger(dfa1, dfa2)
        d_f = dfa_param_distance_fisher(dfa1, dfa2)

        d_behavior.append(d_b)
        d_euclidean.append(d_e)
        d_hellinger.append(d_h)
        d_fisher.append(d_f)

    d_behavior = np.array(d_behavior)
    d_euclidean = np.array(d_euclidean)
    d_hellinger = np.array(d_hellinger)
    d_fisher = np.array(d_fisher)

    # 计算 correlation
    corr_euclidean = np.corrcoef(d_behavior, d_euclidean)[0, 1]
    corr_hellinger = np.corrcoef(d_behavior, d_hellinger)[0, 1]
    corr_fisher = np.corrcoef(d_behavior, d_fisher)[0, 1]

    # 计算 CV
    def compute_cv(d_param, d_behav):
        ratios = d_behav / (d_param + 1e-8)
        return np.std(ratios) / (np.mean(ratios) + 1e-8)

    cv_euclidean = compute_cv(d_euclidean, d_behavior)
    cv_hellinger = compute_cv(d_hellinger, d_behavior)
    cv_fisher = compute_cv(d_fisher, d_behavior)

    return {
        'euclidean': (corr_euclidean, cv_euclidean),
        'hellinger': (corr_hellinger, cv_hellinger),
        'fisher': (corr_fisher, cv_fisher),
    }

def main():
    print("="*70)
    print("Coalgebra 参数化 - 度量选择实验")
    print("="*70)
    print("""
假说: functor 结构不仅定义了参数空间的"形状"，
      也暗示了参数空间的"度量"。

对于 DFA (H = 2 × (-)^Σ):
- 参数住在 simplex (stochastic matrices)
- Simplex 的自然度量: Hellinger, Fisher-Rao
- 不是 Euclidean!

实验: 哪个参数度量更好地 match behavioral distance?
""")

    print(f"\n{'状态数':<8} {'度量':<15} {'Correlation':>12} {'CV':>10}")
    print("-" * 50)

    for n_states in [3, 4, 5]:
        results = experiment(n_states, n_pairs=300)

        for metric_name, (corr, cv) in results.items():
            print(f"{n_states:<8} {metric_name:<15} {corr:>12.3f} {cv:>10.3f}")
        print()

    print("="*70)
    print("结论")
    print("="*70)
    print("""
如果 Hellinger 或 Fisher-Rao 比 Euclidean 更好地 match behavioral distance，
说明:

1. Functor 结构 → 参数空间的自然度量 (不是 Euclidean)
2. 好的参数化 = 使用这个自然度量
3. 或者: 在 logit 空间用 Euclidean ≈ 在概率空间用 Fisher-Rao

这给出了一个 "自然涌现" 的原则:
从 functor 推导参数空间及其度量 → 自动得到好的 distance matching!
""")

if __name__ == "__main__":
    main()
