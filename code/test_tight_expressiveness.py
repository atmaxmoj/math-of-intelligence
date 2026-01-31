"""
Test Hypothesis: 表达力越紧致越好

Use different levels of models to learn the SAME task:
- Task too easy for model → might overfit or learn wrong patterns
- Task matches model → best performance
- Task too hard for model → can't learn

Test on:
1. Regular task (divisibility) with FSA vs Counter vs Multi-Counter
2. CF task (depth counting) with FSA vs Counter vs Multi-Counter
3. CS task (a^n b^n c^n) with FSA vs Counter vs Multi-Counter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys

sys.stdout.reconfigure(line_buffering=True)


# ============ Models at Different Levels ============

class FSA(nn.Module):
    """Level 3: Regular - finite memory."""
    def __init__(self, num_states, num_inputs, num_outputs):
        super().__init__()
        self._T = nn.Parameter(torch.randn(num_states, num_inputs, num_states) * 0.1)
        self._O = nn.Parameter(torch.randn(num_states, num_inputs, num_outputs) * 0.1)
        self._init = nn.Parameter(torch.zeros(num_states))
        self.level = "L3-FSA"

    def forward(self, input_seq):
        T = F.softmax(self._T, dim=-1)
        O = F.softmax(self._O, dim=-1)
        state = F.softmax(self._init, dim=0)
        outputs = []
        for inp in input_seq:
            out_prob = (O[:, inp, :] * state.unsqueeze(1)).sum(0)
            outputs.append(out_prob)
            state = (T[:, inp, :] * state.unsqueeze(1)).sum(0)
        return torch.stack(outputs)


class SingleCounter(nn.Module):
    """Level 2: Context-Free - one counter, unbounded counting."""
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self._delta = nn.Parameter(torch.zeros(num_inputs))
        self._out = nn.Linear(1, num_outputs)
        self.level = "L2-Counter"

    def forward(self, input_seq):
        counter = 0.0
        outputs = []
        for inp in input_seq:
            counter = counter + self._delta[inp]
            out_logits = self._out(counter.unsqueeze(0))
            outputs.append(F.softmax(out_logits, dim=-1).squeeze())
        return torch.stack(outputs)


class MultiCounter(nn.Module):
    """Level 1: Context-Sensitive - multiple counters."""
    def __init__(self, num_inputs, num_outputs, num_counters=3):
        super().__init__()
        self._delta = nn.Parameter(torch.zeros(num_counters, num_inputs))
        self._out = nn.Linear(num_counters, num_outputs)
        self.level = "L1-MultiCtr"
        self.num_counters = num_counters

    def forward(self, input_seq):
        counters = torch.zeros(self.num_counters)
        outputs = []
        for inp in input_seq:
            counters = counters + self._delta[:, inp]
            out_logits = self._out(counters)
            outputs.append(F.softmax(out_logits, dim=-1))
        return torch.stack(outputs)


class Hybrid(nn.Module):
    """Level 1+: State + Counter."""
    def __init__(self, num_states, num_inputs, num_outputs, num_counters=2):
        super().__init__()
        self._T = nn.Parameter(torch.randn(num_states, num_inputs, num_states) * 0.1)
        self._delta = nn.Parameter(torch.zeros(num_counters, num_inputs))
        self._out = nn.Linear(num_states + num_counters, num_outputs)
        self.num_states = num_states
        self.num_counters = num_counters
        self.level = "L1-Hybrid"

    def forward(self, input_seq):
        T = F.softmax(self._T, dim=-1)
        state = torch.zeros(self.num_states)
        state[0] = 1.0
        counters = torch.zeros(self.num_counters)
        outputs = []
        for inp in input_seq:
            counters = counters + self._delta[:, inp]
            state = (T[:, inp, :] * state.unsqueeze(1)).sum(0)
            features = torch.cat([state, counters])
            out_logits = self._out(features)
            outputs.append(F.softmax(out_logits, dim=-1))
        return torch.stack(outputs)


# ============ Tasks ============

def task_regular_mod3(num_samples, max_len):
    """Regular task: binary mod 3."""
    data = []
    for _ in range(num_samples):
        length = random.randint(1, max_len)
        s = [random.randint(0, 1) for _ in range(length)]
        val = 0
        labels = []
        for bit in s:
            val = (val * 2 + bit) % 3
            labels.append(val)
        data.append((s, labels))
    return data, 2, 3  # num_inputs, num_outputs


def task_cf_depth(num_samples, max_len, max_depth=10):
    """CF task: parenthesis depth."""
    data = []
    for _ in range(num_samples):
        seq = []
        depths = []
        depth = 0
        while len(seq) < max_len:
            if depth == 0:
                seq.append(0)  # (
                depth += 1
            elif depth >= max_depth or random.random() < 0.5:
                seq.append(1)  # )
                depth -= 1
            else:
                seq.append(0)
                depth += 1
            depths.append(min(depth, max_depth))
            if depth == 0 and len(seq) >= 4 and random.random() < 0.3:
                break
        data.append((seq, depths))
    return data, 2, max_depth + 1  # num_inputs, num_outputs


def task_cs_anbncn(num_samples, max_n):
    """CS task: a^n b^n c^n validity."""
    data = []
    for _ in range(num_samples):
        if random.random() < 0.5:
            n = random.randint(1, max_n)
            seq = [0] * n + [1] * n + [2] * n  # a, b, c
            labels = [1] * len(seq)
        else:
            na = random.randint(1, max_n)
            nb = random.randint(1, max_n)
            nc = random.randint(1, max_n)
            seq = [0] * na + [1] * nb + [2] * nc
            # Compute validity
            labels = []
            count_a = count_b = count_c = 0
            phase = 0  # 0=a, 1=b, 2=c
            valid = True
            for s in seq:
                if not valid:
                    labels.append(0)
                    continue
                if s == 0:
                    if phase > 0:
                        valid = False
                    count_a += 1
                elif s == 1:
                    if phase == 0:
                        phase = 1
                    elif phase > 1:
                        valid = False
                    count_b += 1
                    if count_b > count_a:
                        valid = False
                else:
                    if phase < 2:
                        if count_b != count_a:
                            valid = False
                        phase = 2
                    count_c += 1
                    if count_c > count_a:
                        valid = False
                labels.append(1 if valid else 0)
        data.append((seq, labels))
    return data, 3, 2  # num_inputs, num_outputs


# ============ Training & Evaluation ============

def train_eval(model, train_data, test_data, epochs=100, lr=0.1):
    """Train and evaluate model."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    for ep in range(epochs):
        random.shuffle(train_data)
        for seq, labels in train_data:
            probs = model(seq)
            loss = F.cross_entropy(probs, torch.tensor(labels))
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Evaluate
    correct = total = 0
    with torch.no_grad():
        for seq, labels in test_data:
            preds = model(seq).argmax(-1).tolist()
            correct += sum(p == t for p, t in zip(preds, labels))
            total += len(labels)

    return correct / total


def main():
    print("=" * 70)
    print("Testing Hypothesis: 表达力越紧致越好")
    print("=" * 70)

    results = {}

    # ============ Task 1: Regular (mod 3) ============
    print("\n" + "=" * 70)
    print("Task 1: Regular (Binary mod 3)")
    print("  Best model should be: FSA (L3)")
    print("=" * 70)

    random.seed(42)
    train_data, num_in, num_out = task_regular_mod3(200, 10)
    random.seed(123)
    test_data, _, _ = task_regular_mod3(100, 20)

    for name, model in [
        ("FSA-3st", FSA(3, num_in, num_out)),
        ("FSA-10st", FSA(10, num_in, num_out)),
        ("Counter", SingleCounter(num_in, num_out)),
        ("MultiCtr", MultiCounter(num_in, num_out, 3)),
        ("Hybrid", Hybrid(3, num_in, num_out, 2)),
    ]:
        acc = train_eval(model, train_data, test_data, epochs=100)
        print(f"  {name:<12}: {acc:.1%}")
        results[('Regular', name)] = acc

    # ============ Task 2: Context-Free (depth) ============
    print("\n" + "=" * 70)
    print("Task 2: Context-Free (Depth counting)")
    print("  Best model should be: Counter (L2)")
    print("=" * 70)

    random.seed(42)
    train_data, num_in, num_out = task_cf_depth(200, 15, max_depth=8)
    random.seed(123)
    test_data, _, _ = task_cf_depth(100, 30, max_depth=15)

    for name, model in [
        ("FSA-10st", FSA(10, num_in, num_out)),
        ("FSA-20st", FSA(20, num_in, num_out)),
        ("Counter", SingleCounter(num_in, num_out)),
        ("MultiCtr", MultiCounter(num_in, num_out, 3)),
        ("Hybrid", Hybrid(5, num_in, num_out, 2)),
    ]:
        acc = train_eval(model, train_data, test_data, epochs=100)
        print(f"  {name:<12}: {acc:.1%}")
        results[('CF', name)] = acc

    # ============ Task 3: Context-Sensitive (a^n b^n c^n) ============
    print("\n" + "=" * 70)
    print("Task 3: Context-Sensitive (a^n b^n c^n)")
    print("  Best model should be: MultiCounter or Hybrid (L1)")
    print("=" * 70)

    random.seed(42)
    train_data, num_in, num_out = task_cs_anbncn(300, 5)
    random.seed(123)
    test_data, _, _ = task_cs_anbncn(100, 10)

    for name, model in [
        ("FSA-10st", FSA(10, num_in, num_out)),
        ("FSA-30st", FSA(30, num_in, num_out)),
        ("Counter", SingleCounter(num_in, num_out)),
        ("MultiCtr", MultiCounter(num_in, num_out, 3)),
        ("Hybrid", Hybrid(4, num_in, num_out, 2)),
    ]:
        acc = train_eval(model, train_data, test_data, epochs=100, lr=0.05)
        print(f"  {name:<12}: {acc:.1%}")
        results[('CS', name)] = acc

    # ============ Summary ============
    print("\n" + "=" * 70)
    print("Summary: Best Model for Each Task")
    print("=" * 70)

    for task in ['Regular', 'CF', 'CS']:
        task_results = {k[1]: v for k, v in results.items() if k[0] == task}
        best = max(task_results.items(), key=lambda x: x[1])
        print(f"\n  {task}:")
        print(f"    Best: {best[0]} ({best[1]:.1%})")
        print(f"    All: {task_results}")

    print("\n" + "=" * 70)
    print("Conclusion:")
    print("  - Regular task: FSA should win (tight fit)")
    print("  - CF task: Counter should win (tight fit)")
    print("  - CS task: MultiCounter/Hybrid should win (tight fit)")
    print("=" * 70)


if __name__ == "__main__":
    main()
