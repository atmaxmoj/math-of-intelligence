"""
MasterCard: One-step vs Global Learning

验证假说：
- One-step objective (per-step CE) → 好学
- Global objective (sequence-level) → 难学

这直接对应我们的 coalgebra 发现：
- One-step Kantorovich metric 匹配参数距离 (corr ≈ 0.8)
- Global language metric 不匹配 (corr ≈ 0.3)
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


def train_onestep(model, traces, state2idx, in2i, out2i, epochs=200):
    """
    One-step learning: cross-entropy on each step independently.
    Loss = Σ_t CE(output_t, target_t)
    """
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    trace_list = list(traces)
    history = []

    for ep in range(epochs):
        random.shuffle(trace_list)
        total_loss = 0
        correct = total = 0

        for trace in trace_list:
            init_state = trace[0][0]
            init_idx = state2idx[init_state]
            input_seq = [in2i[t[1]] for t in trace]
            output_seq = [out2i[t[2]] for t in trace]

            probs = model(init_idx, input_seq)
            targets = torch.tensor(output_seq)

            # One-step loss: average CE over all steps
            loss = F.cross_entropy(probs, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            correct += (probs.argmax(-1) == targets).sum().item()
            total += len(targets)

        history.append((total_loss / len(trace_list), correct / total))
        if ep % 50 == 0:
            print(f"  Epoch {ep}: loss={total_loss/len(trace_list):.4f}, acc={correct/total:.1%}")

    return history


def train_global(model, traces, state2idx, in2i, out2i, epochs=200):
    """
    Global learning: loss only on whether entire sequence matches.
    Loss = -log P(all outputs correct)
         = -Σ_t log P(output_t = target_t)  (但梯度不同！)

    更极端版本：只看最后一步，或者只看整个序列是否完全正确
    """
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    trace_list = list(traces)
    history = []

    for ep in range(epochs):
        random.shuffle(trace_list)
        total_loss = 0
        correct = total = 0

        for trace in trace_list:
            init_state = trace[0][0]
            init_idx = state2idx[init_state]
            input_seq = [in2i[t[1]] for t in trace]
            output_seq = [out2i[t[2]] for t in trace]

            probs = model(init_idx, input_seq)
            targets = torch.tensor(output_seq)

            # Global loss: product of probabilities (sum of log probs)
            # 这让梯度更依赖于整个序列
            log_probs = F.log_softmax(probs, dim=-1)
            target_log_probs = log_probs[range(len(targets)), targets]

            # 关键区别：用 sum 而不是 mean，让长序列的梯度更大
            # 或者更极端：只用最后 k 步
            loss = -target_log_probs.sum()  # Global: sum over sequence

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            correct += (probs.argmax(-1) == targets).sum().item()
            total += len(targets)

        history.append((total_loss / len(trace_list), correct / total))
        if ep % 50 == 0:
            print(f"  Epoch {ep}: loss={total_loss/len(trace_list):.4f}, acc={correct/total:.1%}")

    return history


def train_final_only(model, traces, state2idx, in2i, out2i, epochs=200):
    """
    Extreme global: only care about final output.
    This is like language equivalence - only the final acceptance matters.
    """
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    trace_list = list(traces)
    history = []

    for ep in range(epochs):
        random.shuffle(trace_list)
        total_loss = 0
        correct = total = 0

        for trace in trace_list:
            init_state = trace[0][0]
            init_idx = state2idx[init_state]
            input_seq = [in2i[t[1]] for t in trace]
            output_seq = [out2i[t[2]] for t in trace]

            probs = model(init_idx, input_seq)
            targets = torch.tensor(output_seq)

            # Only final step loss
            loss = F.cross_entropy(probs[-1:], targets[-1:])

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            # Still measure all steps for fair comparison
            correct += (probs.argmax(-1) == targets).sum().item()
            total += len(targets)

        history.append((total_loss / len(trace_list), correct / total))
        if ep % 50 == 0:
            print(f"  Epoch {ep}: loss={total_loss/len(trace_list):.4f}, acc={correct/total:.1%}")

    return history


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


def main():
    print("=" * 70)
    print("MasterCard: One-step vs Global Learning")
    print("=" * 70)
    print("""
假说验证:
- One-step objective (per-step CE) → 对应 one-step metric → 好学
- Global objective (sequence-level) → 对应 language metric → 难学

这直接验证 coalgebra 实验的发现:
- One-step Kantorovich: corr ≈ 0.8 (好匹配 → 好学)
- Global language: corr ≈ 0.3 (差匹配 → 难学)
""")

    # Load MasterCard
    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / \
               "principle" / "BenchmarkBankcard-AartsEtAl2013" / "1_learnresult_MasterCard_fix.dot"
    auto = parse_dot_mealy(dot_path)
    print(f"MasterCard: {len(auto['states'])} states, {len(auto['inputs'])} inputs")

    state2idx = {s: i for i, s in enumerate(auto['states'])}
    in2i = {x: i for i, x in enumerate(auto['inputs'])}
    out2i = {x: i for i, x in enumerate(auto['outputs'])}

    # Generate traces
    random.seed(42)
    train_traces, _ = generate_traces_until_coverage(auto, min_per_transition=30, max_traces=500)
    print(f"Training traces: {len(train_traces)}")

    # Test traces
    random.seed(999)
    test_traces = set()
    for _ in range(200):
        t = generate_trace(auto, initial_state=auto['initial'], max_length=30)
        if t: test_traces.add(t)

    # Compare learning methods
    methods = [
        ("One-step (mean CE)", train_onestep),
        ("Global (sum CE)", train_global),
        ("Final-only", train_final_only),
    ]

    results = {}

    for name, train_fn in methods:
        print("\n" + "=" * 70)
        print(f"Training: {name}")
        print("=" * 70)

        torch.manual_seed(42)
        model = FuzzyMealy(len(auto['states']), len(auto['inputs']), len(auto['outputs']))
        history = train_fn(model, train_traces, state2idx, in2i, out2i, epochs=200)

        sym_acc, seq_acc = evaluate(model, test_traces, state2idx, in2i, out2i)
        results[name] = (sym_acc, seq_acc, history)
        print(f"\nFinal: Symbol acc = {sym_acc:.1%}, Sequence acc = {seq_acc:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\n{'Method':<25} {'Symbol Acc':>12} {'Seq Acc':>12}")
    print("-" * 50)
    for name, (sym, seq, _) in results.items():
        print(f"{name:<25} {sym:>12.1%} {seq:>12.1%}")

    print("\n" + "=" * 70)
    print("Interpretation")
    print("=" * 70)
    print("""
如果 One-step 显著优于 Global/Final-only:
→ 验证了 "局部目标好学，全局目标难学"
→ 与 coalgebra metric matching 实验一致

理论解释:
- One-step loss: 梯度直接传到每一步的参数
- Global loss: 梯度需要通过整个序列传播，信号衰减
- Final-only: 极端情况，只有最后一步有梯度信号

这就是为什么:
- Next-token prediction (GPT) 有效 → one-step
- 学习 trace equivalence 困难 → global
""")


if __name__ == "__main__":
    main()
