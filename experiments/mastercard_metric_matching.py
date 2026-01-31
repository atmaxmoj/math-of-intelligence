"""
MasterCard Mealy Machine: One-step vs Language Metric Matching

验证假说：one-step behavioral metric 匹配参数距离，
         language (global) metric 不匹配。

Mealy machine functor: H = O × (-)^I
- Soft version: transition S×I → Dist(S), output S×I → Dist(O)
"""

import numpy as np
import re
from pathlib import Path
from typing import List, Tuple, Dict

np.random.seed(42)

# ============================================================================
# Parse MasterCard .dot file
# ============================================================================

def parse_dot_mealy(filepath):
    with open(filepath) as f:
        content = f.read()
    states, transitions, inputs, outputs = set(), {}, set(), set()
    initial = None
    for match in re.finditer(r'^(s\d+)\s*\[', content, re.MULTILINE):
        states.add(match.group(1))
    for match in re.finditer(r'(\w+)\s*\[color="red"\]', content):
        initial = match.group(1)
    for match in re.finditer(r'(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\]', content):
        src, dst, label = match.groups()
        if '/' in label:
            inp, out = label.split('/', 1)
            states.add(src); states.add(dst)
            inputs.add(inp.strip()); outputs.add(out.strip())
            transitions[(src, inp.strip())] = (dst, out.strip())
    if initial is None and states:
        initial = sorted(states)[0]
    return {'states': sorted(states), 'inputs': sorted(inputs),
            'outputs': sorted(outputs), 'transitions': transitions, 'initial': initial}

# ============================================================================
# Soft Mealy Machine
# ============================================================================

class SoftMealy:
    """
    Soft Mealy machine with probabilistic transitions and outputs.

    - T[i] is n_states × n_states stochastic matrix for input i
    - O[i] is n_states × n_outputs stochastic matrix for input i
    """
    def __init__(self, transition: List[np.ndarray], output: List[np.ndarray]):
        self.n_states = transition[0].shape[0]
        self.n_inputs = len(transition)
        self.n_outputs = output[0].shape[1]
        self.transition = transition  # list of stochastic matrices
        self.output = output  # list of output distribution matrices

    def run(self, input_seq: List[int], start_state: int = 0) -> List[np.ndarray]:
        """Run on input sequence, return list of output distributions."""
        state_dist = np.zeros(self.n_states)
        state_dist[start_state] = 1.0

        output_dists = []
        for inp in input_seq:
            # Output distribution: weighted by state distribution
            out_dist = state_dist @ self.output[inp]
            output_dists.append(out_dist)
            # Transition
            state_dist = state_dist @ self.transition[inp]

        return output_dists

def random_soft_mealy(n_states: int, n_inputs: int, n_outputs: int) -> SoftMealy:
    """Generate random soft Mealy machine."""
    transitions = []
    outputs = []
    for _ in range(n_inputs):
        T = np.random.dirichlet(np.ones(n_states), size=n_states)
        transitions.append(T)
        O = np.random.dirichlet(np.ones(n_outputs), size=n_states)
        outputs.append(O)
    return SoftMealy(transitions, outputs)

def perturb_soft_mealy(mealy: SoftMealy, noise: float = 0.1) -> SoftMealy:
    """Create a perturbed version of a Mealy machine."""
    new_trans = []
    new_out = []
    for T in mealy.transition:
        T_new = T + np.random.randn(*T.shape) * noise
        T_new = np.clip(T_new, 0.01, None)
        T_new = T_new / T_new.sum(axis=1, keepdims=True)
        new_trans.append(T_new)
    for O in mealy.output:
        O_new = O + np.random.randn(*O.shape) * noise
        O_new = np.clip(O_new, 0.01, None)
        O_new = O_new / O_new.sum(axis=1, keepdims=True)
        new_out.append(O_new)
    return SoftMealy(new_trans, new_out)

# ============================================================================
# Behavioral Metrics
# ============================================================================

def one_step_distance(m1: SoftMealy, m2: SoftMealy) -> float:
    """
    One-step behavioral distance (one Kantorovich iteration).

    Compare transition and output distributions directly.
    """
    trans_diff = 0.0
    out_diff = 0.0

    for T1, T2 in zip(m1.transition, m2.transition):
        # Total variation for each row
        for i in range(m1.n_states):
            trans_diff += 0.5 * np.sum(np.abs(T1[i] - T2[i]))

    for O1, O2 in zip(m1.output, m2.output):
        for i in range(m1.n_states):
            out_diff += 0.5 * np.sum(np.abs(O1[i] - O2[i]))

    # Normalize
    trans_diff /= (m1.n_inputs * m1.n_states)
    out_diff /= (m1.n_inputs * m1.n_states)

    return trans_diff + out_diff

def trace_distance(m1: SoftMealy, m2: SoftMealy, max_length: int = 10) -> float:
    """
    Trace-based behavioral distance (global).

    Compare output distributions on all input sequences.
    """
    def enumerate_seqs(length, n_inputs):
        if length == 0:
            yield []
        else:
            for seq in enumerate_seqs(length - 1, n_inputs):
                for i in range(n_inputs):
                    yield seq + [i]

    distance = 0.0
    for length in range(1, max_length + 1):
        weight = 2 ** (-length)
        for seq in enumerate_seqs(length, m1.n_inputs):
            outs1 = m1.run(seq)
            outs2 = m2.run(seq)
            # Compare final output distribution
            dist = 0.5 * np.sum(np.abs(outs1[-1] - outs2[-1]))
            distance += weight * dist

    return distance

def trace_distance_sampled(m1: SoftMealy, m2: SoftMealy,
                           n_samples: int = 500, max_length: int = 20) -> float:
    """
    Sampled trace distance for larger input alphabets.
    """
    distance = 0.0
    for _ in range(n_samples):
        length = np.random.randint(1, max_length + 1)
        seq = [np.random.randint(m1.n_inputs) for _ in range(length)]
        weight = 2 ** (-length)

        outs1 = m1.run(seq)
        outs2 = m2.run(seq)
        dist = 0.5 * np.sum(np.abs(outs1[-1] - outs2[-1]))
        distance += weight * dist

    return distance / n_samples * max_length  # Normalize

# ============================================================================
# Parameter Distance
# ============================================================================

def param_distance_euclidean(m1: SoftMealy, m2: SoftMealy) -> float:
    """Euclidean distance on flattened parameters."""
    p1 = np.concatenate([T.flatten() for T in m1.transition] +
                        [O.flatten() for O in m1.output])
    p2 = np.concatenate([T.flatten() for T in m2.transition] +
                        [O.flatten() for O in m2.output])
    return np.linalg.norm(p1 - p2)

def param_distance_fisher(m1: SoftMealy, m2: SoftMealy) -> float:
    """Fisher-Rao distance on probability simplices."""
    total = 0.0

    for T1, T2 in zip(m1.transition, m2.transition):
        for i in range(m1.n_states):
            p1, p2 = T1[i] + 1e-10, T2[i] + 1e-10
            inner = np.sum(np.sqrt(p1 * p2))
            inner = np.clip(inner, -1, 1)
            total += (2 * np.arccos(inner)) ** 2

    for O1, O2 in zip(m1.output, m2.output):
        for i in range(m1.n_states):
            p1, p2 = O1[i] + 1e-10, O2[i] + 1e-10
            inner = np.sum(np.sqrt(p1 * p2))
            inner = np.clip(inner, -1, 1)
            total += (2 * np.arccos(inner)) ** 2

    return np.sqrt(total)

# ============================================================================
# Experiment
# ============================================================================

def experiment_random(n_states: int, n_inputs: int, n_outputs: int,
                      n_pairs: int = 200):
    """Test metric matching on random Mealy machines."""

    results = {}

    # All use sampled version for speed
    behavioral_metrics = [
        ("One-step", one_step_distance),
        ("Trace (len≤5)", lambda m1, m2: trace_distance_sampled(m1, m2, n_samples=300, max_length=5)),
        ("Trace (len≤10)", lambda m1, m2: trace_distance_sampled(m1, m2, n_samples=300, max_length=10)),
        ("Trace (len≤20)", lambda m1, m2: trace_distance_sampled(m1, m2, n_samples=300, max_length=20)),
    ]

    param_metrics = [
        ("Euclidean", param_distance_euclidean),
        ("Fisher-Rao", param_distance_fisher),
    ]

    for bm_name, bm_fn in behavioral_metrics:
        print(f"  Testing {bm_name}...")
        for pm_name, pm_fn in param_metrics:
            d_behavior = []
            d_param = []

            for _ in range(n_pairs):
                m1 = random_soft_mealy(n_states, n_inputs, n_outputs)
                m2 = random_soft_mealy(n_states, n_inputs, n_outputs)

                d_behavior.append(bm_fn(m1, m2))
                d_param.append(pm_fn(m1, m2))

            d_behavior = np.array(d_behavior)
            d_param = np.array(d_param)

            corr = np.corrcoef(d_behavior, d_param)[0, 1]
            ratios = d_behavior / (d_param + 1e-8)
            cv = np.std(ratios) / (np.mean(ratios) + 1e-8)

            results[(bm_name, pm_name)] = (corr, cv)

    return results

def experiment_perturbation(n_states: int, n_inputs: int, n_outputs: int,
                            n_pairs: int = 200):
    """Test metric matching with perturbations (more realistic)."""

    results = {}
    noise_levels = [0.05, 0.1, 0.2]

    for noise in noise_levels:
        d_one_step = []
        d_trace = []
        d_param = []

        for _ in range(n_pairs):
            m1 = random_soft_mealy(n_states, n_inputs, n_outputs)
            m2 = perturb_soft_mealy(m1, noise=noise)

            d_one_step.append(one_step_distance(m1, m2))
            d_trace.append(trace_distance_sampled(m1, m2, max_length=15))
            d_param.append(param_distance_euclidean(m1, m2))

        d_one_step = np.array(d_one_step)
        d_trace = np.array(d_trace)
        d_param = np.array(d_param)

        corr_one = np.corrcoef(d_one_step, d_param)[0, 1]
        corr_trace = np.corrcoef(d_trace, d_param)[0, 1]

        results[noise] = (corr_one, corr_trace)

    return results

def main():
    print("=" * 70)
    print("MasterCard Mealy Machine: Metric Matching Experiment")
    print("=" * 70)
    print("""
问题: 对于 Mealy machine (functor H = O × (-)^I),
      one-step metric 是否比 trace metric 更好地匹配参数距离?

这与 MasterCard 学习实验的关系:
- 短序列准确率高 → one-step 行为匹配好
- 长序列准确率低 → global 行为匹配差
""")

    # Load MasterCard structure
    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / \
               "principle" / "BenchmarkBankcard-AartsEtAl2013" / "1_learnresult_MasterCard_fix.dot"

    if dot_path.exists():
        auto = parse_dot_mealy(dot_path)
        print(f"MasterCard 结构: {len(auto['states'])} states, "
              f"{len(auto['inputs'])} inputs, {len(auto['outputs'])} outputs")
        n_states = len(auto['states'])
        n_inputs = len(auto['inputs'])
        n_outputs = len(auto['outputs'])
    else:
        print("MasterCard .dot 文件未找到，使用默认参数")
        n_states, n_inputs, n_outputs = 7, 11, 8

    # Experiment 1: Random pairs
    print("\n" + "=" * 70)
    print("实验 1: 随机 Mealy machine 对")
    print("=" * 70)

    print(f"\n参数: {n_states} states, {n_inputs} inputs, {n_outputs} outputs")
    print(f"\n{'Behavioral Metric':<20} {'Param Metric':<15} {'Corr':>10} {'CV':>10}")
    print("-" * 60)

    results = experiment_random(n_states, n_inputs, n_outputs, n_pairs=150)

    for (bm_name, pm_name), (corr, cv) in sorted(results.items()):
        print(f"{bm_name:<20} {pm_name:<15} {corr:>10.3f} {cv:>10.3f}")

    # Experiment 2: Perturbation (like gradient steps)
    print("\n" + "=" * 70)
    print("实验 2: 扰动实验 (模拟梯度更新)")
    print("=" * 70)

    print(f"\n{'Noise':<10} {'One-step Corr':>15} {'Trace Corr':>15}")
    print("-" * 45)

    results = experiment_perturbation(n_states, n_inputs, n_outputs, n_pairs=150)

    for noise, (corr_one, corr_trace) in sorted(results.items()):
        print(f"{noise:<10} {corr_one:>15.3f} {corr_trace:>15.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
如果 one-step metric 匹配显著好于 trace metric:
→ 验证了 "局部目标容易学，全局目标难学" 的假说

这解释了 MasterCard 学习实验中观察到的现象:
- 短序列 (len≤10): 高准确率 (接近 one-step 行为)
- 长序列 (len≤50): 准确率下降 (global 行为匹配难)

理论含义:
- 学习 next-output prediction 容易 (local)
- 学习 trace equivalence 困难 (global)
- Curriculum learning (短→长) 有理论依据
""")

if __name__ == "__main__":
    main()
