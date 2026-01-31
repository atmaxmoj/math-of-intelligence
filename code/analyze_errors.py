"""
Analyze where errors occur in long sequences.
Are they uniformly distributed or concentrated at certain positions?
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
        for inp_seq, out_seq in traces:
            inp_idx = [in2i.get(x, 0) for x in inp_seq]
            out_idx = [out2i.get(x, 0) for x in out_seq]
            probs = model(inp_idx)
            loss = F.cross_entropy(probs, torch.tensor(out_idx))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0:
            print(f"  Epoch {ep}")


def analyze_position_errors(model, traces, in2i, out2i, max_pos=50):
    """Analyze accuracy at each position."""
    correct_at_pos = [0] * max_pos
    total_at_pos = [0] * max_pos

    # Also track first error position in wrong sequences
    first_error_positions = []

    for inp_seq, out_seq in traces:
        inp_idx = [in2i.get(x, 0) for x in inp_seq]
        out_idx = [out2i.get(x, 0) for x in out_seq]

        with torch.no_grad():
            preds = model(inp_idx).argmax(-1).tolist()

        first_error = None
        for pos, (p, t) in enumerate(zip(preds, out_idx)):
            if pos < max_pos:
                total_at_pos[pos] += 1
                if p == t:
                    correct_at_pos[pos] += 1
                elif first_error is None:
                    first_error = pos

        if first_error is not None:
            first_error_positions.append(first_error)

    return correct_at_pos, total_at_pos, first_error_positions


def main():
    print("=" * 60)
    print("Error Position Analysis")
    print("=" * 60)

    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / "principle" / "BenchmarkBankcard-AartsEtAl2013" / "1_learnresult_MasterCard_fix.dot"
    auto = parse_dot_mealy(dot_path)
    print(f"Ground truth: {len(auto['states'])} states\n")

    in2i = {x: i for i, x in enumerate(auto['inputs'])}
    out2i = {x: i for i, x in enumerate(auto['outputs'])}

    # Train with len≤30 (our best config)
    print("Training with len≤30...")
    random.seed(42)
    train_data = generate_traces(auto, 100, 30)
    model = FuzzyMealy(10, len(auto['inputs']), len(auto['outputs']))
    train(model, train_data, in2i, out2i, epochs=200)

    # Test on len≤50
    print("\nAnalyzing errors on len≤50 test set...")
    random.seed(999)
    test_data = generate_traces(auto, 500, 50)  # More samples for statistics

    correct, total, first_errors = analyze_position_errors(model, test_data, in2i, out2i)

    # Print position-wise accuracy
    print("\n" + "=" * 60)
    print("Position-wise Accuracy")
    print("=" * 60)
    print(f"{'Position':<12} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 42)

    # Group by 5 positions
    for start in range(0, 50, 5):
        end = start + 5
        c = sum(correct[start:end])
        t = sum(total[start:end])
        if t > 0:
            acc = c / t
            print(f"{start:>2}-{end-1:<8} {c:<10} {t:<10} {acc:>8.1%}")

    # First error distribution
    print("\n" + "=" * 60)
    print("First Error Position Distribution")
    print("=" * 60)

    if first_errors:
        print(f"Total sequences with errors: {len(first_errors)}")
        print(f"Average first error position: {sum(first_errors)/len(first_errors):.1f}")
        print(f"Median first error position: {sorted(first_errors)[len(first_errors)//2]}")

        # Histogram
        print("\nHistogram of first error positions:")
        bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
        for start, end in bins:
            count = sum(1 for e in first_errors if start <= e < end)
            bar = "█" * (count // 2)
            print(f"  {start:>2}-{end-1:<3}: {count:>4} {bar}")
    else:
        print("No errors found!")

    # Detailed single-position accuracy for first 20 positions
    print("\n" + "=" * 60)
    print("Detailed: First 20 Positions")
    print("=" * 60)
    for pos in range(20):
        if total[pos] > 0:
            acc = correct[pos] / total[pos]
            bar = "█" * int(acc * 20)
            print(f"  pos {pos:>2}: {acc:>6.1%} {bar}")


if __name__ == "__main__":
    main()
