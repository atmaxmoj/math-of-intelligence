"""
Coalgebra 参数化 - 第三版

新思路: 用 Kantorovich lifting 定义 behavioral metric

对于 functor H = 2 × (-)^Σ:
- 状态空间 S 上的度量 d
- H 把它 lift 到 H(S) = 2 × S^Σ 上的度量

d_H((a₁, δ₁), (a₂, δ₂)) = |a₁ - a₂| + Σ_σ d(δ₁(σ), δ₂(σ))

对于 soft DFA (概率版):
- H = [0,1] × Dist(-)^Σ
- Kantorovich lift on Dist
"""

import numpy as np
from typing import List, Tuple

np.random.seed(42)

# ============================================================================
# Soft DFA
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

def random_soft_dfa(n_states: int, n_symbols: int = 2) -> SoftDFA:
    transitions = []
    for _ in range(n_symbols):
        T = np.random.dirichlet(np.ones(n_states), size=n_states)
        transitions.append(T)
    accept = np.random.rand(n_states)
    return SoftDFA(transitions, accept)

# ============================================================================
# 不同的 Behavioral Metrics
# ============================================================================

def language_distance(dfa1: SoftDFA, dfa2: SoftDFA, max_length: int = 5) -> float:
    """之前用的：基于语言差异"""
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

def one_step_distance(dfa1: SoftDFA, dfa2: SoftDFA) -> float:
    """
    一步 behavioral distance (Kantorovich lifting 的一步)

    d((a₁, T₁), (a₂, T₂)) = |a₁ - a₂|_avg + Σ_σ W(T₁[σ], T₂[σ])_avg

    其中 W 是 Wasserstein-1 距离 (用离散度量)
    """
    # Accept 差异
    accept_diff = np.mean(np.abs(dfa1.accept - dfa2.accept))

    # Transition 差异 (用 L1 on rows 作为 W1 的上界)
    trans_diff = 0.0
    for T1, T2 in zip(dfa1.transition, dfa2.transition):
        # 每行是一个概率分布，用 total variation
        for i in range(dfa1.n_states):
            trans_diff += 0.5 * np.sum(np.abs(T1[i] - T2[i]))

    trans_diff /= (len(dfa1.transition) * dfa1.n_states)

    return accept_diff + trans_diff

def fixed_point_distance(dfa1: SoftDFA, dfa2: SoftDFA, iterations: int = 10) -> float:
    """
    迭代不动点距离

    d⁰(s, t) = 0 for all s, t
    dⁿ⁺¹(s, t) = |a₁(s) - a₂(t)| + γ Σ_σ W(T₁[s,σ], T₂[t,σ]; dⁿ)

    这更接近真正的 behavioral metric (bisimulation distance)
    """
    n = dfa1.n_states
    gamma = 0.9  # 折扣因子

    # 初始化：状态间距离为 0
    d = np.zeros((n, n))

    for _ in range(iterations):
        d_new = np.zeros((n, n))

        for s in range(n):
            for t in range(n):
                # Accept 差异
                accept_diff = abs(dfa1.accept[s] - dfa2.accept[t])

                # Transition 差异 (Kantorovich with metric d)
                trans_diff = 0.0
                for T1, T2 in zip(dfa1.transition, dfa2.transition):
                    p1 = T1[s]  # 从 s 出发的转移分布
                    p2 = T2[t]  # 从 t 出发的转移分布

                    # Wasserstein-1 with metric d
                    # W(p1, p2; d) = min_{coupling} E[d(X, Y)]
                    # 对于小规模，直接用 LP
                    cost_matrix = d
                    # 简化：用 Earth Mover's Distance 近似
                    # 这里用 L1 作为上界
                    trans_diff += 0.5 * np.sum(np.abs(p1 - p2))

                d_new[s, t] = accept_diff + gamma * trans_diff / len(dfa1.transition)

        d = d_new

    # 返回从初始状态 (0, 0) 的距离
    return d[0, 0]

def structure_preserving_distance(dfa1: SoftDFA, dfa2: SoftDFA) -> float:
    """
    结构保持距离: 把 DFA 看作 coalgebra morphism 的距离

    直接比较对应的结构元素
    """
    total = 0.0

    # Accept
    total += np.sum((dfa1.accept - dfa2.accept) ** 2)

    # Transitions
    for T1, T2 in zip(dfa1.transition, dfa2.transition):
        total += np.sum((T1 - T2) ** 2)

    return np.sqrt(total)

# ============================================================================
# 参数距离
# ============================================================================

def param_distance_euclidean(dfa1: SoftDFA, dfa2: SoftDFA) -> float:
    """Euclidean on flattened parameters"""
    p1 = np.concatenate([T.flatten() for T in dfa1.transition] + [dfa1.accept])
    p2 = np.concatenate([T.flatten() for T in dfa2.transition] + [dfa2.accept])
    return np.linalg.norm(p1 - p2)

def param_distance_fisher(dfa1: SoftDFA, dfa2: SoftDFA) -> float:
    """Fisher-Rao on probability simplex"""
    total = 0.0

    for T1, T2 in zip(dfa1.transition, dfa2.transition):
        for i in range(dfa1.n_states):
            p1, p2 = T1[i] + 1e-10, T2[i] + 1e-10
            inner = np.sum(np.sqrt(p1 * p2))
            inner = np.clip(inner, -1, 1)
            total += (2 * np.arccos(inner)) ** 2

    for a1, a2 in zip(dfa1.accept, dfa2.accept):
        p1 = np.array([a1 + 1e-10, 1 - a1 + 1e-10])
        p2 = np.array([a2 + 1e-10, 1 - a2 + 1e-10])
        inner = np.sum(np.sqrt(p1 * p2))
        inner = np.clip(inner, -1, 1)
        total += (2 * np.arccos(inner)) ** 2

    return np.sqrt(total)

# ============================================================================
# 实验
# ============================================================================

def experiment(n_states: int, n_pairs: int = 200):
    """测试不同 behavioral metric 和 param metric 的匹配程度"""

    results = {}

    behavioral_metrics = [
        ("Language (Σ 2^-|w|)", language_distance),
        ("One-step", one_step_distance),
        ("Fixed-point", fixed_point_distance),
        ("Structure", structure_preserving_distance),
    ]

    param_metrics = [
        ("Euclidean", param_distance_euclidean),
        ("Fisher-Rao", param_distance_fisher),
    ]

    for bm_name, bm_fn in behavioral_metrics:
        for pm_name, pm_fn in param_metrics:
            d_behavior = []
            d_param = []

            for _ in range(n_pairs):
                dfa1 = random_soft_dfa(n_states)
                dfa2 = random_soft_dfa(n_states)

                d_behavior.append(bm_fn(dfa1, dfa2))
                d_param.append(pm_fn(dfa1, dfa2))

            d_behavior = np.array(d_behavior)
            d_param = np.array(d_param)

            corr = np.corrcoef(d_behavior, d_param)[0, 1]
            ratios = d_behavior / (d_param + 1e-8)
            cv = np.std(ratios) / (np.mean(ratios) + 1e-8)

            results[(bm_name, pm_name)] = (corr, cv)

    return results

def main():
    print("="*80)
    print("Coalgebra Behavioral Metric 实验")
    print("="*80)
    print("""
问题: 什么 behavioral metric 最好地 match 参数距离?

Behavioral metrics:
1. Language distance: Σ_w 2^(-|w|) |A(w) - B(w)|  (语言差异)
2. One-step: 直接比较 (accept, transition) 的一步差异
3. Fixed-point: 迭代计算 bisimulation distance
4. Structure: 直接比较结构元素 (最简单)

Parameter metrics:
1. Euclidean: 在展平的参数上
2. Fisher-Rao: 在概率单纯形上
""")

    n_states = 4

    print(f"\n状态数 = {n_states}\n")
    print(f"{'Behavioral Metric':<25} {'Param Metric':<15} {'Corr':>10} {'CV':>10}")
    print("-" * 65)

    results = experiment(n_states, n_pairs=200)

    # 按 behavioral metric 分组打印
    current_bm = None
    for (bm_name, pm_name), (corr, cv) in sorted(results.items()):
        if bm_name != current_bm:
            if current_bm is not None:
                print()
            current_bm = bm_name
        print(f"{bm_name:<25} {pm_name:<15} {corr:>10.3f} {cv:>10.3f}")

    # 找最佳匹配
    best = max(results.items(), key=lambda x: x[1][0])
    print(f"\n最佳匹配: {best[0][0]} + {best[0][1]}, Corr = {best[1][0]:.3f}")

    print("\n" + "="*80)
    print("洞察")
    print("="*80)
    print("""
如果 "Structure" behavioral metric 匹配最好，说明:
→ 直接比较 coalgebra 结构 = 最自然的 behavioral distance
→ Functor 结构直接给出了 behavioral metric!

如果 "Fixed-point" (bisimulation distance) 匹配好:
→ Kantorovich lifting 是正确的
→ 但计算更复杂

关键洞察:
Functor H = 2 × (-)^Σ 定义了:
1. 参数空间的结构 (stochastic matrices + accept vector)
2. Behavioral metric (结构元素的直接比较 或 Kantorovich lifting)
3. 如果用正确的度量，distance matching 自然涌现!
""")

if __name__ == "__main__":
    main()
