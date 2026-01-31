"""
MasterCard: 从 Metric Matching 理论预测学习效果

理论框架：
- Loss function 诱导 behavioral metric
- Metric 和参数距离的匹配程度决定"好学"程度
- 我们应该能从 metric matching 预测不同 loss 的学习效果
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random
import sys
from collections import Counter
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

np.random.seed(42)

# ============================================================================
# Soft Mealy Machine (numpy 版，用于 metric 计算)
# ============================================================================

class SoftMealy:
    def __init__(self, T_list, O_list):
        """T_list: list of transition matrices, O_list: list of output matrices"""
        self.n_states = T_list[0].shape[0]
        self.n_inputs = len(T_list)
        self.n_outputs = O_list[0].shape[1]
        self.T = T_list  # T[i] is n_states × n_states
        self.O = O_list  # O[i] is n_states × n_outputs

    def run(self, input_seq, start_state=0):
        """Return list of output distributions for each step"""
        state = np.zeros(self.n_states)
        state[start_state] = 1.0
        outputs = []
        for inp in input_seq:
            out_dist = state @ self.O[inp]
            outputs.append(out_dist)
            state = state @ self.T[inp]
        return outputs

def random_soft_mealy(n_states, n_inputs, n_outputs):
    T_list = [np.random.dirichlet(np.ones(n_states), size=n_states) for _ in range(n_inputs)]
    O_list = [np.random.dirichlet(np.ones(n_outputs), size=n_states) for _ in range(n_inputs)]
    return SoftMealy(T_list, O_list)

def perturb_mealy(m, noise=0.1):
    T_list = []
    O_list = []
    for T in m.T:
        T_new = T + np.random.randn(*T.shape) * noise
        T_new = np.clip(T_new, 0.01, None)
        T_new = T_new / T_new.sum(axis=1, keepdims=True)
        T_list.append(T_new)
    for O in m.O:
        O_new = O + np.random.randn(*O.shape) * noise
        O_new = np.clip(O_new, 0.01, None)
        O_new = O_new / O_new.sum(axis=1, keepdims=True)
        O_list.append(O_new)
    return SoftMealy(T_list, O_list)

# ============================================================================
# 不同 Loss 诱导的 Behavioral Metrics
# ============================================================================

def metric_onestep(m1, m2, n_samples=100, max_len=30):
    """
    One-step loss 诱导的 metric:
    比较每一步的输出分布，然后平均

    d(m1, m2) = E_seq [ mean_t TV(output1_t, output2_t) ]
    """
    total = 0.0
    for _ in range(n_samples):
        length = np.random.randint(5, max_len + 1)
        seq = [np.random.randint(m1.n_inputs) for _ in range(length)]

        outs1 = m1.run(seq)
        outs2 = m2.run(seq)

        # 平均每一步的 TV 距离
        step_dists = [0.5 * np.sum(np.abs(o1 - o2)) for o1, o2 in zip(outs1, outs2)]
        total += np.mean(step_dists)

    return total / n_samples

def metric_final_only(m1, m2, n_samples=100, max_len=30):
    """
    Final-only loss 诱导的 metric:
    只比较最后一步的输出分布

    d(m1, m2) = E_seq [ TV(output1_final, output2_final) ]
    """
    total = 0.0
    for _ in range(n_samples):
        length = np.random.randint(5, max_len + 1)
        seq = [np.random.randint(m1.n_inputs) for _ in range(length)]

        outs1 = m1.run(seq)
        outs2 = m2.run(seq)

        # 只看最后一步
        total += 0.5 * np.sum(np.abs(outs1[-1] - outs2[-1]))

    return total / n_samples

def metric_last_k(m1, m2, k=5, n_samples=100, max_len=30):
    """Last-k loss 诱导的 metric"""
    total = 0.0
    for _ in range(n_samples):
        length = np.random.randint(max(k, 5), max_len + 1)
        seq = [np.random.randint(m1.n_inputs) for _ in range(length)]

        outs1 = m1.run(seq)
        outs2 = m2.run(seq)

        # 最后 k 步的平均
        step_dists = [0.5 * np.sum(np.abs(o1 - o2)) for o1, o2 in zip(outs1[-k:], outs2[-k:])]
        total += np.mean(step_dists)

    return total / n_samples

def metric_weighted(m1, m2, n_samples=100, max_len=30):
    """Weighted loss 诱导的 metric (后面步骤权重更大)"""
    total = 0.0
    for _ in range(n_samples):
        length = np.random.randint(5, max_len + 1)
        seq = [np.random.randint(m1.n_inputs) for _ in range(length)]

        outs1 = m1.run(seq)
        outs2 = m2.run(seq)

        # 加权平均，后面权重更大
        weights = np.arange(1, length + 1) / length
        step_dists = [0.5 * np.sum(np.abs(o1 - o2)) for o1, o2 in zip(outs1, outs2)]
        total += np.average(step_dists, weights=weights)

    return total / n_samples

# ============================================================================
# 参数距离
# ============================================================================

def param_distance(m1, m2):
    """Euclidean distance on flattened parameters"""
    p1 = np.concatenate([T.flatten() for T in m1.T] + [O.flatten() for O in m1.O])
    p2 = np.concatenate([T.flatten() for T in m2.T] + [O.flatten() for O in m2.O])
    return np.linalg.norm(p1 - p2)

# ============================================================================
# 计算 Metric Matching (Correlation)
# ============================================================================

def compute_metric_matching(n_states, n_inputs, n_outputs, n_pairs=200):
    """计算不同 behavioral metric 和参数距离的 correlation"""

    metrics = {
        "One-step": metric_onestep,
        "Final-only": metric_final_only,
        "Last-5": lambda m1, m2: metric_last_k(m1, m2, k=5),
        "Weighted": metric_weighted,
    }

    results = {}

    for name, metric_fn in metrics.items():
        print(f"  Computing {name}...")
        d_metric = []
        d_param = []

        for _ in range(n_pairs):
            m1 = random_soft_mealy(n_states, n_inputs, n_outputs)
            m2 = perturb_mealy(m1, noise=0.2)  # 扰动，模拟梯度更新

            d_metric.append(metric_fn(m1, m2))
            d_param.append(param_distance(m1, m2))

        d_metric = np.array(d_metric)
        d_param = np.array(d_param)

        corr = np.corrcoef(d_metric, d_param)[0, 1]

        # 计算 κ 的近似：比值的变异系数
        ratios = d_metric / (d_param + 1e-8)
        cv = np.std(ratios) / (np.mean(ratios) + 1e-8)

        results[name] = (corr, cv, np.mean(d_metric), np.mean(d_param))

    return results

# ============================================================================
# 主实验
# ============================================================================

def main():
    print("=" * 70)
    print("MasterCard: Metric Matching 理论预测")
    print("=" * 70)
    print("""
理论框架：
1. Loss function 诱导 behavioral metric
2. Metric 和参数距离的匹配程度 (correlation) 决定"好学"
3. 我们应该能预测：correlation 高 → accuracy 高

实验结果 (已知)：
- One-step:    94% seq acc
- Final-only:  10% seq acc
- Last-5:      20% seq acc
- Weighted:    15% seq acc

预测：One-step 的 correlation 应该显著高于其他
""")

    # MasterCard 的参数
    n_states = 5
    n_inputs = 15
    n_outputs = 9

    print(f"\n计算 metric matching (n_states={n_states}, n_inputs={n_inputs})...\n")

    results = compute_metric_matching(n_states, n_inputs, n_outputs, n_pairs=150)

    print("\n" + "=" * 70)
    print("结果")
    print("=" * 70)

    # 实际的 accuracy 结果
    actual_acc = {
        "One-step": 94.0,
        "Final-only": 10.3,
        "Last-5": 19.7,
        "Weighted": 15.3,
    }

    print(f"\n{'Metric':<15} {'Correlation':>12} {'CV':>10} {'Actual Acc':>12}")
    print("-" * 55)

    for name, (corr, cv, _, _) in sorted(results.items(), key=lambda x: -x[1][0]):
        acc = actual_acc[name]
        print(f"{name:<15} {corr:>12.3f} {cv:>10.3f} {acc:>11.1f}%")

    # 计算 correlation 和 accuracy 的相关性
    corrs = [results[name][0] for name in actual_acc.keys()]
    accs = [actual_acc[name] for name in actual_acc.keys()]

    meta_corr = np.corrcoef(corrs, accs)[0, 1]

    print(f"\n理论预测能力：")
    print(f"  Correlation(metric_matching, accuracy) = {meta_corr:.3f}")

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)

    if meta_corr > 0.8:
        print(f"""
✓ 理论预测成功！(meta correlation = {meta_corr:.2f})

Metric matching (correlation) 能够预测学习效果 (accuracy)：
- 高 correlation → 高 accuracy
- 低 correlation → 低 accuracy

这验证了核心假说：
  Loss → Behavioral Metric → Metric Matching → 好学程度
""")
    else:
        print(f"""
理论预测部分成功 (meta correlation = {meta_corr:.2f})

需要进一步分析...
""")

if __name__ == "__main__":
    main()
