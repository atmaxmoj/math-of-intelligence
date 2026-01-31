"""
Investigate why errors concentrate at early positions.
What specific state-input combinations are being misclassified?
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random
import sys
from collections import Counter, defaultdict

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
        input_seq, output_seq, state_seq = [], [], [state]
        for _ in range(length):
            valid = [i for i in automaton['inputs'] if (state, i) in automaton['transitions']]
            if not valid: break
            inp = random.choice(valid)
            next_state, out = automaton['transitions'][(state, inp)]
            input_seq.append(inp); output_seq.append(out)
            state = next_state
            state_seq.append(state)
        if input_seq:
            traces.append((input_seq, output_seq, state_seq))
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
        for inp_seq, out_seq, _ in traces:
            inp_idx = [in2i.get(x, 0) for x in inp_seq]
            out_idx = [out2i.get(x, 0) for x in out_seq]
            probs = model(inp_idx)
            loss = F.cross_entropy(probs, torch.tensor(out_idx))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0:
            print(f"  Epoch {ep}")


def main():
    print("=" * 60)
    print("Error Case Analysis")
    print("=" * 60)

    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / "principle" / "BenchmarkBankcard-AartsEtAl2013" / "1_learnresult_MasterCard_fix.dot"
    auto = parse_dot_mealy(dot_path)

    print(f"Ground truth: {len(auto['states'])} states")
    print(f"Inputs: {auto['inputs']}")
    print(f"Outputs: {auto['outputs']}")
    print(f"Initial state: {auto['initial']}")

    in2i = {x: i for i, x in enumerate(auto['inputs'])}
    out2i = {x: i for i, x in enumerate(auto['outputs'])}
    i2out = {i: x for x, i in out2i.items()}

    # Train with len≤30
    print("\nTraining with len≤30...")
    random.seed(42)
    train_data = generate_traces(auto, 100, 30)
    model = FuzzyMealy(10, len(auto['inputs']), len(auto['outputs']))
    train(model, train_data, in2i, out2i, epochs=200)

    # Test and collect error cases
    print("\nAnalyzing error cases...")
    random.seed(999)
    test_data = generate_traces(auto, 500, 50)

    # Collect errors by (true_state, input) -> (expected_output, predicted_output)
    error_by_state_input = defaultdict(list)
    error_by_position = defaultdict(list)

    # Also track training data coverage
    train_state_input_counts = Counter()
    for inp_seq, out_seq, state_seq in train_data:
        for i, (inp, state) in enumerate(zip(inp_seq, state_seq[:-1])):
            train_state_input_counts[(state, inp)] += 1

    for inp_seq, out_seq, state_seq in test_data:
        inp_idx = [in2i.get(x, 0) for x in inp_seq]
        out_idx = [out2i.get(x, 0) for x in out_seq]

        with torch.no_grad():
            preds = model(inp_idx).argmax(-1).tolist()

        for pos, (p, t, inp, state) in enumerate(zip(preds, out_idx, inp_seq, state_seq[:-1])):
            if p != t:
                error_by_state_input[(state, inp)].append({
                    'pos': pos,
                    'expected': i2out[t],
                    'predicted': i2out[p],
                })
                error_by_position[pos].append({
                    'state': state,
                    'input': inp,
                    'expected': i2out[t],
                    'predicted': i2out[p],
                })

    # Report errors by state-input combination
    print("\n" + "=" * 60)
    print("Errors by (State, Input) Combination")
    print("=" * 60)

    sorted_errors = sorted(error_by_state_input.items(), key=lambda x: -len(x[1]))

    print(f"\n{'State':<8} {'Input':<25} {'Errors':<8} {'Train Cnt':<10} {'Expected → Predicted'}")
    print("-" * 80)

    for (state, inp), errors in sorted_errors[:15]:
        expected = errors[0]['expected']
        predicted = errors[0]['predicted']
        train_cnt = train_state_input_counts.get((state, inp), 0)
        print(f"{state:<8} {inp:<25} {len(errors):<8} {train_cnt:<10} {expected} → {predicted}")

    # Check if errors correlate with training frequency
    print("\n" + "=" * 60)
    print("Training Coverage Analysis")
    print("=" * 60)

    # All state-input combinations in ground truth
    all_combos = set(auto['transitions'].keys())
    trained_combos = set(train_state_input_counts.keys())
    error_combos = set(error_by_state_input.keys())

    print(f"\nTotal (state, input) combinations in automaton: {len(all_combos)}")
    print(f"Combinations seen in training: {len(trained_combos)}")
    print(f"Combinations with errors: {len(error_combos)}")

    # Combos with errors that were never seen in training
    never_trained_errors = error_combos - trained_combos
    rarely_trained_errors = {c for c in error_combos if train_state_input_counts.get(c, 0) < 3}

    print(f"\nError combos NEVER seen in training: {len(never_trained_errors)}")
    print(f"Error combos seen < 3 times in training: {len(rarely_trained_errors)}")

    if never_trained_errors:
        print("\nNever-trained combos with errors:")
        for state, inp in list(never_trained_errors)[:10]:
            errors = error_by_state_input[(state, inp)]
            print(f"  ({state}, {inp}): {len(errors)} errors")

    # Position analysis for position 0-9 errors
    print("\n" + "=" * 60)
    print("Position 0-9 Error Details")
    print("=" * 60)

    early_error_combos = Counter()
    for pos in range(10):
        for err in error_by_position[pos]:
            early_error_combos[(err['state'], err['input'])] += 1

    print(f"\nMost common (state, input) causing early errors:")
    for (state, inp), count in early_error_combos.most_common(10):
        train_cnt = train_state_input_counts.get((state, inp), 0)
        print(f"  ({state}, {inp}): {count} errors, trained {train_cnt} times")

    # Check learned vs true transitions
    print("\n" + "=" * 60)
    print("Learned vs True Output Function")
    print("=" * 60)

    # Get the learned output matrix
    with torch.no_grad():
        O = F.softmax(model._O, dim=-1)

    # For the top error combo, show what the model learned
    if sorted_errors:
        top_state, top_inp = sorted_errors[0][0]
        top_errors = sorted_errors[0][1]
        expected_out = top_errors[0]['expected']
        predicted_out = top_errors[0]['predicted']

        print(f"\nTop error combo: ({top_state}, {top_inp})")
        print(f"  Expected output: {expected_out}")
        print(f"  Predicted output: {predicted_out}")

        inp_idx = in2i[top_inp]
        print(f"\n  Learned O matrix for input '{top_inp}':")
        print(f"  (each row is a learned state's output distribution)")
        for s in range(min(5, model.num_states)):
            out_dist = O[s, inp_idx, :].numpy()
            top_out_idx = out_dist.argmax()
            print(f"    State {s}: max output = {i2out[top_out_idx]} ({out_dist[top_out_idx]:.3f})")


if __name__ == "__main__":
    main()
