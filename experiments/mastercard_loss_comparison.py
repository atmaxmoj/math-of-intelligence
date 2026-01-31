"""
MasterCard: 只改变 loss function 的对比实验

假说：one-step loss 好学，global loss 难学
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random
import sys
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)

# ============================================================================
# 完全复制原来的代码
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


def generate_traces_until_coverage(auto, min_per_transition=20, max_traces=1000):
    traces = set()
    coverage = Counter()
    all_transitions = list(auto['transitions'].keys())
    attempts = 0
    while attempts < max_traces:
        start_state = random.choice(auto['states'])
        trace = generate_trace(auto, initial_state=start_state, max_length=30)
        if trace and trace not in traces:
            traces.add(trace)
            for state, inp, out in trace:
                coverage[(state, inp)] += 1
        attempts += 1
        min_cov = min(coverage.get(t, 0) for t in all_transitions)
        if min_cov >= min_per_transition:
            break
    return traces, coverage


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


def evaluate(model, traces, state2idx, in2i, out2i):
    correct = total = seq_correct = 0
    with torch.no_grad():
        for trace in traces:
            init_state = trace[0][0]
            init_idx = state2idx[init_state]
            input_seq = [in2i[t[1]] for t in trace]
            output_seq = [out2i[t[2]] for t in trace]
            preds = model(init_idx, input_seq).argmax(-1).tolist()
            correct += sum(p == t for p, t in zip(preds, output_seq))
            total += len(output_seq)
            if preds == output_seq:
                seq_correct += 1
    return correct / total, seq_correct / len(traces)


# ============================================================================
# 不同的 loss functions
# ============================================================================

def compute_loss_onestep(probs, targets):
    """原始 loss: cross_entropy 对所有步取平均"""
    return F.cross_entropy(probs, targets)


def compute_loss_final_only(probs, targets):
    """只看最后一步"""
    return F.cross_entropy(probs[-1:], targets[-1:])


def compute_loss_last_k(probs, targets, k=5):
    """只看最后 k 步"""
    return F.cross_entropy(probs[-k:], targets[-k:])


def compute_loss_weighted(probs, targets):
    """后面的步骤权重更大 (模拟 global 目标)"""
    n = len(targets)
    weights = torch.arange(1, n + 1, dtype=torch.float32) / n
    log_probs = F.log_softmax(probs, dim=-1)
    target_log_probs = log_probs[range(n), targets]
    return -(weights * target_log_probs).mean()


# ============================================================================
# 训练函数 (完全相同，只有 loss 不同)
# ============================================================================

def train(model, traces, state2idx, in2i, out2i, loss_fn, epochs=300):
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    trace_list = list(traces)

    for ep in range(epochs):
        random.shuffle(trace_list)
        correct = total = 0

        for trace in trace_list:
            init_state = trace[0][0]
            init_idx = state2idx[init_state]
            input_seq = [in2i[t[1]] for t in trace]
            output_seq = [out2i[t[2]] for t in trace]

            probs = model(init_idx, input_seq)
            targets = torch.tensor(output_seq)

            # 唯一的区别：loss function
            loss = loss_fn(probs, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            correct += (probs.argmax(-1) == targets).sum().item()
            total += len(targets)

        if ep % 100 == 0:
            print(f"  Epoch {ep}: {correct/total:.1%}")

    return correct / total


# ============================================================================
# 主实验
# ============================================================================

def main():
    print("=" * 60)
    print("MasterCard: Loss Function 对比实验")
    print("=" * 60)
    print("""
只改变 loss function，其他完全相同：
- 模型结构相同
- 训练数据相同
- 优化器相同
- epochs 相同
""")

    # Load data
    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / \
               "principle" / "BenchmarkBankcard-AartsEtAl2013" / "1_learnresult_MasterCard_fix.dot"
    auto = parse_dot_mealy(dot_path)
    print(f"MasterCard: {len(auto['states'])} states, {len(auto['transitions'])} transitions")

    state2idx = {s: i for i, s in enumerate(auto['states'])}
    in2i = {x: i for i, x in enumerate(auto['inputs'])}
    out2i = {x: i for i, x in enumerate(auto['outputs'])}

    # Generate training traces (same for all)
    random.seed(42)
    train_traces, _ = generate_traces_until_coverage(auto, min_per_transition=30, max_traces=500)
    print(f"Training traces: {len(train_traces)}")

    # Generate test traces (same for all)
    random.seed(999)
    test_traces_30 = set()
    for _ in range(300):
        t = generate_trace(auto, initial_state=auto['initial'], max_length=30)
        if t: test_traces_30.add(t)
    print(f"Test traces: {len(test_traces_30)}")

    # Loss functions to compare
    loss_functions = [
        ("One-step (原始)", compute_loss_onestep),
        ("Final-only", compute_loss_final_only),
        ("Last-5", lambda p, t: compute_loss_last_k(p, t, k=5)),
        ("Weighted (后重)", compute_loss_weighted),
    ]

    results = {}

    for name, loss_fn in loss_functions:
        print("\n" + "=" * 60)
        print(f"Training with: {name}")
        print("=" * 60)

        # 重置随机种子，确保初始化相同
        torch.manual_seed(42)
        model = FuzzyMealy(len(auto['states']), len(auto['inputs']), len(auto['outputs']))

        train(model, train_traces, state2idx, in2i, out2i, loss_fn, epochs=300)

        sym_acc, seq_acc = evaluate(model, test_traces_30, state2idx, in2i, out2i)
        results[name] = (sym_acc, seq_acc)
        print(f"\nTest: Symbol {sym_acc:.1%}, Sequence {seq_acc:.1%}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\n{'Loss Function':<25} {'Symbol Acc':>12} {'Seq Acc':>12}")
    print("-" * 50)
    for name, (sym, seq) in results.items():
        print(f"{name:<25} {sym:>12.1%} {seq:>12.1%}")

    print("\n" + "=" * 60)
    print("解释")
    print("=" * 60)
    print("""
如果 One-step >> Final-only:
→ 验证了 "局部目标好学，全局目标难学"
→ 与 coalgebra metric matching 实验一致

理论：
- One-step: 每一步都有梯度信号，对应 one-step behavioral metric
- Final-only: 只有最后一步有信号，需要通过整个序列传播
- 这就是为什么 RNN 难以学习长程依赖
""")


if __name__ == "__main__":
    main()
