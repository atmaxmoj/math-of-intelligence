"""
Ablation Study: What causes accuracy drop on long sequences?

Test:
1. Correct number of states (5) vs overestimate (10)
2. More training data
3. Longer training sequences
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random


def parse_dot_mealy(filepath):
    with open(filepath) as f:
        content = f.read()

    states = set()
    transitions = {}
    inputs = set()
    outputs = set()
    initial = None

    for match in re.finditer(r'^(s\d+)\s*\[', content, re.MULTILINE):
        states.add(match.group(1))

    for match in re.finditer(r'(\w+)\s*\[color="red"\]', content):
        initial = match.group(1)
        states.add(initial)

    for match in re.finditer(r'(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\]', content):
        src, dst, label = match.groups()
        if '/' in label:
            inp, out = label.split('/', 1)
            states.add(src)
            states.add(dst)
            inputs.add(inp.strip())
            outputs.add(out.strip())
            transitions[(src, inp.strip())] = (dst, out.strip())

    if initial is None and states:
        initial = sorted(states)[0]

    return {
        'states': sorted(states),
        'inputs': sorted(inputs),
        'outputs': sorted(outputs),
        'transitions': transitions,
        'initial': initial
    }


def generate_traces(automaton, num_traces=100, max_length=10):
    traces = []
    for _ in range(num_traces):
        state = automaton['initial']
        length = random.randint(1, max_length)
        input_seq = []
        output_seq = []
        for _ in range(length):
            valid_inputs = [inp for inp in automaton['inputs']
                          if (state, inp) in automaton['transitions']]
            if not valid_inputs:
                break
            inp = random.choice(valid_inputs)
            next_state, out = automaton['transitions'][(state, inp)]
            input_seq.append(inp)
            output_seq.append(out)
            state = next_state
        if input_seq:
            traces.append((input_seq, output_seq))
    return traces


class FuzzyMealy(nn.Module):
    def __init__(self, num_states, num_inputs, num_outputs):
        super().__init__()
        self.num_states = num_states
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._T = nn.Parameter(torch.randn(num_states, num_inputs, num_states) * 0.1)
        self._O = nn.Parameter(torch.randn(num_states, num_inputs, num_outputs) * 0.1)

    @property
    def T(self):
        return F.softmax(self._T, dim=-1)

    @property
    def O(self):
        return F.softmax(self._O, dim=-1)

    def forward(self, input_seq, initial_state=0):
        state_dist = torch.zeros(self.num_states)
        state_dist[initial_state] = 1.0
        output_probs = []
        for inp in input_seq:
            out_prob = (self.O[:, inp, :] * state_dist.unsqueeze(1)).sum(dim=0)
            output_probs.append(out_prob)
            new_state_dist = (self.T[:, inp, :] * state_dist.unsqueeze(1)).sum(dim=0)
            state_dist = new_state_dist
        return torch.stack(output_probs)

    def sparsity_loss(self):
        T = self.T
        O = self.O
        T_entropy = -(T * (T + 1e-10).log()).sum()
        O_entropy = -(O * (O + 1e-10).log()).sum()
        return (T_entropy + O_entropy) / (self.num_states * self.num_inputs)


def train(model, traces, input_to_idx, output_to_idx, epochs=500, lr=0.05, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_count = 0

        random.shuffle(traces)

        for input_seq, output_seq in traces:
            inp_indices = [input_to_idx.get(x, 0) for x in input_seq]
            out_indices = [output_to_idx.get(x, 0) for x in output_seq]

            out_probs = model(inp_indices)
            targets = torch.tensor(out_indices)
            loss = F.cross_entropy(out_probs, targets)
            loss = loss + model.sparsity_loss() * 0.0001

            total_loss += loss.item()
            preds = out_probs.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += len(targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose and epoch % 100 == 0:
            acc = total_correct / total_count if total_count > 0 else 0
            print(f"    Epoch {epoch}: acc={acc:.2%}")

    return model


def evaluate(model, traces, input_to_idx, output_to_idx):
    total_symbols = 0
    correct_symbols = 0
    full_match = 0

    for input_seq, output_seq in traces:
        inp_indices = [input_to_idx.get(x, 0) for x in input_seq]
        out_indices = [output_to_idx.get(x, 0) for x in output_seq]

        with torch.no_grad():
            out_probs = model(inp_indices)
            preds = out_probs.argmax(dim=-1).tolist()

        for p, t in zip(preds, out_indices):
            total_symbols += 1
            if p == t:
                correct_symbols += 1

        if preds == out_indices:
            full_match += 1

    return {
        'symbol_acc': correct_symbols / total_symbols if total_symbols > 0 else 0,
        'seq_acc': full_match / len(traces) if traces else 0,
    }


def run_experiment(name, num_states, num_traces, train_max_len, automaton, input_to_idx, output_to_idx, test_sets):
    print(f"\n  {name}")
    print(f"    States: {num_states}, Traces: {num_traces}, Train max_len: {train_max_len}")

    random.seed(42)
    train_data = generate_traces(automaton, num_traces=num_traces, max_length=train_max_len)

    model = FuzzyMealy(num_states, len(automaton['inputs']), len(automaton['outputs']))
    model = train(model, train_data, input_to_idx, output_to_idx, epochs=500, verbose=False)

    results = {}
    for test_name, test_data in test_sets.items():
        r = evaluate(model, test_data, input_to_idx, output_to_idx)
        results[test_name] = r

    return results


def main():
    print("=" * 70)
    print("Ablation Study: What causes long-sequence accuracy drop?")
    print("=" * 70)

    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / "principle" / "BenchmarkBankcard-AartsEtAl2013" / "1_learnresult_MasterCard_fix.dot"

    automaton = parse_dot_mealy(dot_path)
    true_states = len(automaton['states'])
    print(f"\nGround truth: {true_states} states")

    input_to_idx = {x: i for i, x in enumerate(automaton['inputs'])}
    output_to_idx = {x: i for i, x in enumerate(automaton['outputs'])}

    # Fixed test sets
    random.seed(123)
    test_sets = {
        'len≤10': generate_traces(automaton, num_traces=300, max_length=10),
        'len≤30': generate_traces(automaton, num_traces=300, max_length=30),
        'len≤50': generate_traces(automaton, num_traces=300, max_length=50),
        'len≤100': generate_traces(automaton, num_traces=300, max_length=100),
    }

    experiments = [
        # (name, num_states, num_traces, train_max_len)
        ("Baseline (10 states, 100 traces, len≤10)", 10, 100, 10),
        ("Correct states (5)", 5, 100, 10),
        ("More data (500 traces)", 10, 500, 10),
        ("Longer train (len≤30)", 10, 100, 30),
        ("All improvements", 5, 500, 30),
    ]

    all_results = {}

    print("\n" + "=" * 70)
    print("Running experiments...")
    print("=" * 70)

    for name, num_states, num_traces, train_max_len in experiments:
        results = run_experiment(name, num_states, num_traces, train_max_len,
                                automaton, input_to_idx, output_to_idx, test_sets)
        all_results[name] = results

    # Print results table
    print("\n" + "=" * 70)
    print("Results: Symbol Accuracy")
    print("=" * 70)

    print(f"\n{'Experiment':<45} {'len≤10':>10} {'len≤30':>10} {'len≤50':>10} {'len≤100':>10}")
    print("-" * 85)

    for name, results in all_results.items():
        row = f"{name:<45}"
        for test_name in ['len≤10', 'len≤30', 'len≤50', 'len≤100']:
            acc = results[test_name]['symbol_acc']
            row += f" {acc:>9.2%}"
        print(row)

    print("\n" + "=" * 70)
    print("Results: Sequence Accuracy")
    print("=" * 70)

    print(f"\n{'Experiment':<45} {'len≤10':>10} {'len≤30':>10} {'len≤50':>10} {'len≤100':>10}")
    print("-" * 85)

    for name, results in all_results.items():
        row = f"{name:<45}"
        for test_name in ['len≤10', 'len≤30', 'len≤50', 'len≤100']:
            acc = results[test_name]['seq_acc']
            row += f" {acc:>9.2%}"
        print(row)

    # Highlight best
    print("\n" + "-" * 70)
    print("Key findings:")
    print("-" * 70)

    baseline = all_results["Baseline (10 states, 100 traces, len≤10)"]
    best = all_results["All improvements"]

    for test_name in ['len≤50', 'len≤100']:
        b_sym = baseline[test_name]['symbol_acc']
        best_sym = best[test_name]['symbol_acc']
        improvement = best_sym - b_sym
        print(f"  {test_name}: {b_sym:.2%} → {best_sym:.2%} ({improvement:+.2%})")


if __name__ == "__main__":
    main()
