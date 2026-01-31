"""
Quick Ablation: What causes long-sequence accuracy drop?
Lighter version - fewer epochs, faster results.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random
import sys

# Force unbuffered output
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


class FuzzyMealy(nn.Module):
    def __init__(self, num_states, num_inputs, num_outputs):
        super().__init__()
        self.num_states = num_states
        self._T = nn.Parameter(torch.randn(num_states, num_inputs, num_states) * 0.1)
        self._O = nn.Parameter(torch.randn(num_states, num_inputs, num_outputs) * 0.1)

    def forward(self, input_seq):
        T = F.softmax(self._T, dim=-1)
        O = F.softmax(self._O, dim=-1)
        state = torch.zeros(self.num_states)
        state[0] = 1.0
        outputs = []
        for inp in input_seq:
            outputs.append((O[:, inp, :] * state.unsqueeze(1)).sum(0))
            state = (T[:, inp, :] * state.unsqueeze(1)).sum(0)
        return torch.stack(outputs)


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
            print(f"    Epoch {ep}: {correct/total:.1%}")


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
    print("Quick Ablation Study")
    print("=" * 60)

    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / "principle" / "BenchmarkBankcard-AartsEtAl2013" / "1_learnresult_MasterCard_fix.dot"
    auto = parse_dot_mealy(dot_path)
    true_states = len(auto['states'])
    print(f"Ground truth: {true_states} states\n")

    in2i = {x: i for i, x in enumerate(auto['inputs'])}
    out2i = {x: i for i, x in enumerate(auto['outputs'])}

    # Fixed test sets
    random.seed(999)
    test_10 = generate_traces(auto, 200, 10)
    test_30 = generate_traces(auto, 200, 30)
    test_50 = generate_traces(auto, 200, 50)

    experiments = [
        ("Baseline (10 states, 100 traces, len≤10)", 10, 100, 10),
        ("Correct states (5)", 5, 100, 10),
        ("More data (300 traces)", 10, 300, 10),
        ("Longer train (len≤30)", 10, 100, 30),
        ("Best combo (5 states, 300 traces, len≤30)", 5, 300, 30),
    ]

    results = []
    for name, n_states, n_traces, max_len in experiments:
        print(f"\n{name}")
        random.seed(42)
        train_data = generate_traces(auto, n_traces, max_len)
        model = FuzzyMealy(n_states, len(auto['inputs']), len(auto['outputs']))
        train(model, train_data, in2i, out2i, epochs=200)

        r = {}
        for test_name, test_data in [("len≤10", test_10), ("len≤30", test_30), ("len≤50", test_50)]:
            sym, seq = evaluate(model, test_data, in2i, out2i)
            r[test_name] = (sym, seq)
        results.append((name, r))

    print("\n" + "=" * 60)
    print("Results: Symbol Accuracy")
    print("=" * 60)
    print(f"{'Experiment':<45} {'len≤10':>10} {'len≤30':>10} {'len≤50':>10}")
    print("-" * 75)
    for name, r in results:
        print(f"{name:<45} {r['len≤10'][0]:>9.1%} {r['len≤30'][0]:>9.1%} {r['len≤50'][0]:>9.1%}")

    print("\n" + "=" * 60)
    print("Results: Sequence Accuracy")
    print("=" * 60)
    print(f"{'Experiment':<45} {'len≤10':>10} {'len≤30':>10} {'len≤50':>10}")
    print("-" * 75)
    for name, r in results:
        print(f"{name:<45} {r['len≤10'][1]:>9.1%} {r['len≤30'][1]:>9.1%} {r['len≤50'][1]:>9.1%}")

    # Key insight
    print("\n" + "-" * 60)
    baseline = results[0][1]
    best = results[-1][1]
    print("Improvement (Baseline → Best):")
    for k in ['len≤10', 'len≤30', 'len≤50']:
        print(f"  {k}: {baseline[k][0]:.1%} → {best[k][0]:.1%} (symbol)")


if __name__ == "__main__":
    main()
