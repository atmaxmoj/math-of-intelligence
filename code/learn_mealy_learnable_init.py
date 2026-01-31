"""
Learnable Initial State: Let the model learn which state to start from.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random
import sys

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


def generate_traces(automaton, num_traces=100, max_length=10):
    traces = []
    for _ in range(num_traces):
        state = automaton['initial']
        length = random.randint(1, max_length)
        input_seq, output_seq = [], []
        for _ in range(length):
            valid = [i for i in automaton['inputs'] if (state, i) in automaton['transitions']]
            if not valid: break
            inp = random.choice(valid)
            next_state, out = automaton['transitions'][(state, inp)]
            input_seq.append(inp); output_seq.append(out)
            state = next_state
        if input_seq:
            traces.append((input_seq, output_seq))
    return traces


class FuzzyMealyBaseline(nn.Module):
    """Baseline: hardcoded initial state 0"""
    def __init__(self, num_states, num_inputs, num_outputs):
        super().__init__()
        self.num_states = num_states
        self._T = nn.Parameter(torch.randn(num_states, num_inputs, num_states) * 0.1)
        self._O = nn.Parameter(torch.randn(num_states, num_inputs, num_outputs) * 0.1)

    def forward(self, input_seq):
        T = F.softmax(self._T, dim=-1)
        O = F.softmax(self._O, dim=-1)
        state = torch.zeros(self.num_states)
        state[0] = 1.0  # Hardcoded!
        outputs = []
        for inp in input_seq:
            outputs.append((O[:, inp, :] * state.unsqueeze(1)).sum(0))
            state = (T[:, inp, :] * state.unsqueeze(1)).sum(0)
        return torch.stack(outputs)


class FuzzyMealyLearnableInit(nn.Module):
    """Learnable initial state distribution"""
    def __init__(self, num_states, num_inputs, num_outputs):
        super().__init__()
        self.num_states = num_states
        self._T = nn.Parameter(torch.randn(num_states, num_inputs, num_states) * 0.1)
        self._O = nn.Parameter(torch.randn(num_states, num_inputs, num_outputs) * 0.1)
        self._init = nn.Parameter(torch.zeros(num_states))  # Learnable!

    def forward(self, input_seq):
        T = F.softmax(self._T, dim=-1)
        O = F.softmax(self._O, dim=-1)
        state = F.softmax(self._init, dim=0)  # Learned initial distribution
        outputs = []
        for inp in input_seq:
            outputs.append((O[:, inp, :] * state.unsqueeze(1)).sum(0))
            state = (T[:, inp, :] * state.unsqueeze(1)).sum(0)
        return torch.stack(outputs)

    def get_initial_state(self):
        return F.softmax(self._init, dim=0)


def train(model, traces, in2i, out2i, epochs=200):
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for ep in range(epochs):
        random.shuffle(traces)
        correct = total = 0
        for inp_seq, out_seq in traces:
            inp_idx = [in2i.get(x, 0) for x in inp_seq]
            out_idx = [out2i.get(x, 0) for x in out_seq]
            probs = model(inp_idx)
            loss = F.cross_entropy(probs, torch.tensor(out_idx))
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (probs.argmax(-1) == torch.tensor(out_idx)).sum().item()
            total += len(out_idx)
        if ep % 50 == 0:
            acc = correct / total
            print(f"    Epoch {ep}: {acc:.1%}")


def evaluate(model, traces, in2i, out2i):
    correct = total = seq_correct = 0
    for inp_seq, out_seq in traces:
        inp_idx = [in2i.get(x, 0) for x in inp_seq]
        out_idx = [out2i.get(x, 0) for x in out_seq]
        with torch.no_grad():
            preds = model(inp_idx).argmax(-1).tolist()
        correct += sum(p == t for p, t in zip(preds, out_idx))
        total += len(out_idx)
        if preds == out_idx: seq_correct += 1
    return correct / total, seq_correct / len(traces)


def main():
    print("=" * 60)
    print("Learnable Initial State Experiment")
    print("=" * 60)

    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / "principle" / "BenchmarkBankcard-AartsEtAl2013" / "1_learnresult_MasterCard_fix.dot"
    auto = parse_dot_mealy(dot_path)
    print(f"Ground truth: {len(auto['states'])} states\n")

    in2i = {x: i for i, x in enumerate(auto['inputs'])}
    out2i = {x: i for i, x in enumerate(auto['outputs'])}

    # Fixed test sets
    random.seed(999)
    test_10 = generate_traces(auto, 200, 10)
    test_30 = generate_traces(auto, 200, 30)
    test_50 = generate_traces(auto, 200, 50)

    # Train data
    random.seed(42)
    train_data = generate_traces(auto, 100, 30)

    # ============ Baseline ============
    print("Training BASELINE (hardcoded init=0)...")
    random.seed(42)
    baseline = FuzzyMealyBaseline(10, len(auto['inputs']), len(auto['outputs']))
    train(baseline, train_data, in2i, out2i, epochs=200)

    # ============ Learnable Init ============
    print("\nTraining LEARNABLE INIT...")
    random.seed(42)
    learnable = FuzzyMealyLearnableInit(10, len(auto['inputs']), len(auto['outputs']))
    train(learnable, train_data, in2i, out2i, epochs=200)

    # Show learned initial state
    init_dist = learnable.get_initial_state().detach().numpy()
    print(f"\n    Learned initial state distribution:")
    for i, p in enumerate(init_dist):
        if p > 0.01:
            bar = "█" * int(p * 30)
            print(f"      State {i}: {p:.3f} {bar}")

    learned_init = init_dist.argmax()
    init_conf = init_dist[learned_init]
    print(f"\n    → Converged to state {learned_init} (confidence: {init_conf:.3f})")

    # ============ Compare Results ============
    print("\n" + "=" * 60)
    print("Results Comparison")
    print("=" * 60)

    results = []
    for name, model in [("Baseline (init=0)", baseline), ("Learnable init", learnable)]:
        r = {}
        for test_name, test_data in [("len≤10", test_10), ("len≤30", test_30), ("len≤50", test_50)]:
            sym, seq = evaluate(model, test_data, in2i, out2i)
            r[test_name] = (sym, seq)
        results.append((name, r))

    print(f"\n{'Model':<25} {'len≤10':>12} {'len≤30':>12} {'len≤50':>12}")
    print("-" * 65)

    print("\nSymbol Accuracy:")
    for name, r in results:
        print(f"  {name:<23} {r['len≤10'][0]:>11.1%} {r['len≤30'][0]:>11.1%} {r['len≤50'][0]:>11.1%}")

    print("\nSequence Accuracy:")
    for name, r in results:
        print(f"  {name:<23} {r['len≤10'][1]:>11.1%} {r['len≤30'][1]:>11.1%} {r['len≤50'][1]:>11.1%}")

    # Improvement
    baseline_r = results[0][1]
    learnable_r = results[1][1]

    print("\n" + "-" * 60)
    print("Improvement (Baseline → Learnable):")
    for k in ['len≤10', 'len≤30', 'len≤50']:
        sym_diff = learnable_r[k][0] - baseline_r[k][0]
        seq_diff = learnable_r[k][1] - baseline_r[k][1]
        print(f"  {k}: symbol {sym_diff:+.1%}, sequence {seq_diff:+.1%}")


if __name__ == "__main__":
    main()
