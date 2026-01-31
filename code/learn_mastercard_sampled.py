"""
Learn MasterCard protocol with multi-start sampling (like we did for BB).
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
    """Generate one trace from a given starting state."""
    state = initial_state or auto['initial']
    trace = []  # (state, input, output)

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
    """Generate traces until each (state, input) pair is covered enough."""
    traces = set()
    coverage = Counter()
    all_transitions = list(auto['transitions'].keys())

    attempts = 0
    while attempts < max_traces:
        # Random starting state
        start_state = random.choice(auto['states'])
        trace = generate_trace(auto, initial_state=start_state, max_length=30)

        if trace and trace not in traces:
            traces.add(trace)
            for state, inp, out in trace:
                coverage[(state, inp)] += 1

        attempts += 1

        # Check coverage
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


def train(model, traces, state2idx, in2i, out2i, epochs=300):
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
            loss = F.cross_entropy(probs, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            correct += (probs.argmax(-1) == targets).sum().item()
            total += len(targets)

        if ep % 100 == 0:
            print(f"  Epoch {ep}: {correct/total:.1%}")

    return correct / total


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
    print("=" * 60)
    print("MasterCard Learning with Multi-Start Sampling")
    print("=" * 60)

    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / "principle" / "BenchmarkBankcard-AartsEtAl2013" / "1_learnresult_MasterCard_fix.dot"
    auto = parse_dot_mealy(dot_path)

    print(f"\nGround truth: {len(auto['states'])} states, {len(auto['transitions'])} transitions")

    state2idx = {s: i for i, s in enumerate(auto['states'])}
    in2i = {x: i for i, x in enumerate(auto['inputs'])}
    out2i = {x: i for i, x in enumerate(auto['outputs'])}

    # Generate training traces with coverage guarantee
    print("\nGenerating traces until each transition appears 30+ times...")
    random.seed(42)
    train_traces, coverage = generate_traces_until_coverage(
        auto, min_per_transition=30, max_traces=500
    )

    total_steps = sum(len(t) for t in train_traces)
    print(f"Generated {len(train_traces)} unique traces, {total_steps} total steps")

    # Show coverage
    print(f"\nTransition coverage (min/max/avg):")
    counts = list(coverage.values())
    print(f"  Min: {min(counts)}, Max: {max(counts)}, Avg: {sum(counts)/len(counts):.1f}")

    # Generate test traces (from initial state only, like original)
    print("\nGenerating test traces (from initial state)...")
    random.seed(999)
    test_traces_10 = set()
    test_traces_30 = set()
    test_traces_50 = set()

    for _ in range(300):
        t = generate_trace(auto, initial_state=auto['initial'], max_length=10)
        if t: test_traces_10.add(t)
    for _ in range(300):
        t = generate_trace(auto, initial_state=auto['initial'], max_length=30)
        if t: test_traces_30.add(t)
    for _ in range(300):
        t = generate_trace(auto, initial_state=auto['initial'], max_length=50)
        if t: test_traces_50.add(t)

    print(f"  Test len≤10: {len(test_traces_10)} traces")
    print(f"  Test len≤30: {len(test_traces_30)} traces")
    print(f"  Test len≤50: {len(test_traces_50)} traces")

    # Train with correct number of states
    print("\n" + "=" * 60)
    print(f"Training with {len(auto['states'])} states (ground truth)...")
    print("=" * 60)

    torch.manual_seed(42)
    model = FuzzyMealy(len(auto['states']), len(auto['inputs']), len(auto['outputs']))
    train(model, train_traces, state2idx, in2i, out2i, epochs=300)

    # Evaluate
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    print(f"\n{'Test Set':<20} {'Symbol Acc':<15} {'Sequence Acc':<15}")
    print("-" * 50)

    for name, test_set in [("len≤10", test_traces_10),
                           ("len≤30", test_traces_30),
                           ("len≤50", test_traces_50)]:
        sym_acc, seq_acc = evaluate(model, test_set, state2idx, in2i, out2i)
        print(f"{name:<20} {sym_acc:<15.1%} {seq_acc:<15.1%}")

    # Compare with previous baseline
    print("\n" + "=" * 60)
    print("Comparison with previous results")
    print("=" * 60)
    print("""
Previous (single-start, len≤30 train):
  len≤50: Symbol 97.7%, Sequence 84.5%

Now (multi-start sampling):
  See results above
""")


if __name__ == "__main__":
    main()
