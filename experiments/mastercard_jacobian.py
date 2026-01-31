"""
MasterCard: 计算不同 loss 的 Jacobian 条件数 κ(J)

真正决定"好学"的是 Jacobian：
- J = ∂loss/∂θ 的结构
- κ(J) = σ_max / σ_min

One-step loss: 每一步都贡献梯度 → J 列向量线性独立 → κ(J) 小
Final-only loss: 只有最后一步 → J 很多列是 0 或相关 → κ(J) 大
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
import re
import sys
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)

# ============================================================================
# 复用之前的代码
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


def generate_trace(auto, initial_state=None, max_length=30):
    state = initial_state or auto['initial']
    trace = []
    for _ in range(max_length):
        valid = [i for i in auto['inputs'] if (state, i) in auto['transitions']]
        if not valid:
            break
        inp = random.choice(valid)
        next_state, out = auto['transitions'][(state, inp)]
        trace.append((state, inp, out))
        state = next_state
    return tuple(trace)


class FuzzyMealy(nn.Module):
    def __init__(self, num_states, num_inputs, num_outputs):
        super().__init__()
        self.num_states = num_states
        self._T = nn.Parameter(torch.randn(num_states, num_inputs, num_states) * 0.1)
        self._O = nn.Parameter(torch.randn(num_states, num_inputs, num_outputs) * 0.1)

    def forward(self, init_state_idx, input_seq):
        T = F.softmax(self._T, dim=-1)
        O = F.softmax(self._O, dim=-1)
        state = torch.zeros(self.num_states)
        state[init_state_idx] = 1.0
        outputs = []
        for inp in input_seq:
            outputs.append((O[:, inp, :] * state.unsqueeze(1)).sum(0))
            state = (T[:, inp, :] * state.unsqueeze(1)).sum(0)
        return torch.stack(outputs)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# 计算不同 loss 的 Jacobian
# ============================================================================

def compute_jacobian_for_loss(model, trace, state2idx, in2i, out2i, loss_type="onestep"):
    """
    计算 Jacobian: J[i,j] = ∂loss_i / ∂θ_j

    对于不同的 loss_type:
    - "onestep": loss 是每一步 CE 的平均
    - "final": loss 只是最后一步的 CE
    - "last5": loss 是最后 5 步的 CE
    """
    init_idx = state2idx[trace[0][0]]
    input_seq = [in2i[t[1]] for t in trace]
    output_seq = [out2i[t[2]] for t in trace]

    probs = model(init_idx, input_seq)
    targets = torch.tensor(output_seq)

    n_steps = len(targets)
    n_params = model.num_params()

    # 计算每一步的梯度
    all_grads = []

    for t in range(n_steps):
        model.zero_grad()
        loss_t = F.cross_entropy(probs[t:t+1], targets[t:t+1])
        loss_t.backward(retain_graph=True)

        grad_t = []
        for p in model.parameters():
            grad_t.append(p.grad.flatten().clone())
        all_grads.append(torch.cat(grad_t))

    # 根据 loss_type 组合梯度
    if loss_type == "onestep":
        # 平均所有步的梯度
        J = torch.stack(all_grads)  # [n_steps, n_params]
    elif loss_type == "final":
        # 只用最后一步
        J = all_grads[-1].unsqueeze(0)  # [1, n_params]
    elif loss_type == "last5":
        # 最后 5 步
        J = torch.stack(all_grads[-5:])  # [5, n_params]
    elif loss_type == "weighted":
        # 加权
        weights = torch.arange(1, n_steps + 1, dtype=torch.float32) / n_steps
        weighted_grads = [w * g for w, g in zip(weights, all_grads)]
        J = torch.stack(weighted_grads)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return J


def compute_kappa(J):
    """计算 Jacobian 的条件数"""
    # SVD
    try:
        U, S, V = torch.svd(J)
        s_max = S[0].item()
        s_min = S[-1].item()
        if s_min < 1e-10:
            return float('inf')
        return s_max / s_min
    except:
        return float('inf')


def compute_effective_rank(J):
    """计算有效秩（衡量梯度多样性）"""
    try:
        U, S, V = torch.svd(J)
        S = S / S.sum()  # 归一化
        entropy = -(S * torch.log(S + 1e-10)).sum()
        return torch.exp(entropy).item()
    except:
        return 0


# ============================================================================
# 主实验
# ============================================================================

def main():
    print("=" * 70)
    print("MasterCard: Jacobian 条件数分析")
    print("=" * 70)
    print("""
假说：κ(J) 决定学习效果
- One-step loss: 每一步都有梯度 → Jacobian 行多、列独立 → κ(J) 小
- Final-only: 只有最后一步 → Jacobian 只有 1 行 → κ(J) 大或无意义

更好的指标：有效秩 (effective rank)
- 衡量 Jacobian 的"梯度多样性"
- 高多样性 → 参数更新方向丰富 → 好学
""")

    # Load MasterCard
    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / \
               "principle" / "BenchmarkBankcard-AartsEtAl2013" / "1_learnresult_MasterCard_fix.dot"
    auto = parse_dot_mealy(dot_path)
    print(f"\nMasterCard: {len(auto['states'])} states")

    state2idx = {s: i for i, s in enumerate(auto['states'])}
    in2i = {x: i for i, x in enumerate(auto['inputs'])}
    out2i = {x: i for i, x in enumerate(auto['outputs'])}

    # 生成一些测试 traces
    random.seed(42)
    traces = []
    for _ in range(20):
        t = generate_trace(auto, initial_state=auto['initial'], max_length=30)
        if t and len(t) >= 10:
            traces.append(t)

    print(f"Test traces: {len(traces)}, avg length: {np.mean([len(t) for t in traces]):.1f}")

    # 创建模型
    torch.manual_seed(42)
    model = FuzzyMealy(len(auto['states']), len(auto['inputs']), len(auto['outputs']))
    print(f"Model params: {model.num_params()}")

    # 对每种 loss type 计算 Jacobian 统计量
    loss_types = ["onestep", "final", "last5", "weighted"]
    actual_acc = {
        "onestep": 94.0,
        "final": 10.3,
        "last5": 19.7,
        "weighted": 15.3,
    }

    results = {}

    for loss_type in loss_types:
        print(f"\nComputing for {loss_type}...")
        kappas = []
        eff_ranks = []
        j_rows = []

        for trace in traces:
            J = compute_jacobian_for_loss(model, trace, state2idx, in2i, out2i, loss_type)
            kappas.append(compute_kappa(J))
            eff_ranks.append(compute_effective_rank(J))
            j_rows.append(J.shape[0])

        results[loss_type] = {
            'kappa_mean': np.mean([k for k in kappas if k < 1e10]),
            'kappa_median': np.median([k for k in kappas if k < 1e10]),
            'eff_rank_mean': np.mean(eff_ranks),
            'j_rows': np.mean(j_rows),
            'actual_acc': actual_acc[loss_type],
        }

    # 打印结果
    print("\n" + "=" * 70)
    print("结果")
    print("=" * 70)

    print(f"\n{'Loss Type':<12} {'J rows':>8} {'Eff Rank':>10} {'κ(J) med':>10} {'Actual Acc':>12}")
    print("-" * 60)

    for loss_type in loss_types:
        r = results[loss_type]
        print(f"{loss_type:<12} {r['j_rows']:>8.1f} {r['eff_rank_mean']:>10.2f} "
              f"{r['kappa_median']:>10.1f} {r['actual_acc']:>11.1f}%")

    # 计算预测相关性
    eff_ranks = [results[lt]['eff_rank_mean'] for lt in loss_types]
    accs = [results[lt]['actual_acc'] for lt in loss_types]
    corr_rank_acc = np.corrcoef(eff_ranks, accs)[0, 1]

    print(f"\n预测能力:")
    print(f"  Correlation(effective_rank, accuracy) = {corr_rank_acc:.3f}")

    print("\n" + "=" * 70)
    print("解释")
    print("=" * 70)
    print(f"""
有效秩 (Effective Rank) 解释:
- One-step: J 有 ~30 行（每一步一行），有效秩高 → 梯度方向丰富
- Final-only: J 只有 1 行，有效秩 = 1 → 只有一个梯度方向
- Last-5: J 有 5 行，有效秩 ~5 → 中等

这解释了为什么:
- One-step 能学好: 每一步都提供独立的梯度信号
- Final-only 学不好: 所有参数共享一个梯度方向，容易陷入局部最优

理论连接:
- 有效秩 ≈ 参数空间被有效探索的维度
- 高有效秩 → 参数空间探索充分 → 好学
""")

if __name__ == "__main__":
    main()
