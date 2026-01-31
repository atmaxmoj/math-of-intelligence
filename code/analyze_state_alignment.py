"""
Analyze state alignment: does the model's state trajectory match the true automaton?
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


def generate_traces_with_states(automaton, num_traces=100, max_length=10):
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

    def forward_with_states(self, input_seq):
        """Return outputs AND state distributions at each step."""
        T = F.softmax(self._T, dim=-1)
        O = F.softmax(self._O, dim=-1)
        state = torch.zeros(self.num_states)
        state[0] = 1.0
        outputs = []
        states = [state.clone()]
        for inp in input_seq:
            outputs.append((O[:, inp, :] * state.unsqueeze(1)).sum(0))
            state = (T[:, inp, :] * state.unsqueeze(1)).sum(0)
            states.append(state.clone())
        return torch.stack(outputs), states


def train(model, traces, in2i, out2i, epochs=200):
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for ep in range(epochs):
        random.shuffle(traces)
        for inp_seq, out_seq, _ in traces:
            inp_idx = [in2i.get(x, 0) for x in inp_seq]
            out_idx = [out2i.get(x, 0) for x in out_seq]
            out_probs, _ = model.forward_with_states(inp_idx)
            loss = F.cross_entropy(out_probs, torch.tensor(out_idx))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0:
            print(f"  Epoch {ep}")


def main():
    print("=" * 60)
    print("State Alignment Analysis")
    print("=" * 60)

    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / "principle" / "BenchmarkBankcard-AartsEtAl2013" / "1_learnresult_MasterCard_fix.dot"
    auto = parse_dot_mealy(dot_path)

    in2i = {x: i for i, x in enumerate(auto['inputs'])}
    out2i = {x: i for i, x in enumerate(auto['outputs'])}
    i2out = {i: x for x, i in out2i.items()}
    state2i = {s: i for i, s in enumerate(auto['states'])}

    print(f"True states: {auto['states']}")
    print(f"Initial: {auto['initial']}")

    # Train
    print("\nTraining...")
    random.seed(42)
    train_data = generate_traces_with_states(auto, 100, 30)
    model = FuzzyMealy(10, len(auto['inputs']), len(auto['outputs']))
    train(model, train_data, in2i, out2i, epochs=200)

    # Find state correspondence
    print("\n" + "=" * 60)
    print("Learning State Correspondence")
    print("=" * 60)

    # For each true state, collect the model state distributions
    # when the true automaton is in that state
    from collections import defaultdict
    import numpy as np

    state_distributions = defaultdict(list)  # true_state -> list of model state dists

    random.seed(999)
    test_data = generate_traces_with_states(auto, 200, 30)

    for inp_seq, out_seq, true_states in test_data:
        inp_idx = [in2i.get(x, 0) for x in inp_seq]
        with torch.no_grad():
            _, model_states = model.forward_with_states(inp_idx)

        for true_s, model_s in zip(true_states, model_states):
            state_distributions[true_s].append(model_s.numpy())

    # Average model state distribution for each true state
    print("\nAverage model state distribution when in each true state:")
    print("(Higher values = model thinks it's in that state)\n")

    avg_dists = {}
    for true_s in auto['states']:
        dists = state_distributions[true_s]
        avg = np.mean(dists, axis=0)
        avg_dists[true_s] = avg
        top_model_state = np.argmax(avg)
        print(f"True {true_s}: model state {top_model_state} ({avg[top_model_state]:.2f})")
        print(f"         distribution: {[f'{x:.2f}' for x in avg[:5]]}")

    # Build correspondence map
    print("\n" + "=" * 60)
    print("State Correspondence Map")
    print("=" * 60)

    correspondence = {}
    used_model_states = set()

    for true_s in auto['states']:
        avg = avg_dists[true_s]
        # Find best unused model state
        for model_s in np.argsort(avg)[::-1]:
            if model_s not in used_model_states:
                correspondence[true_s] = model_s
                used_model_states.add(model_s)
                print(f"{true_s} <-> model state {model_s} (confidence: {avg[model_s]:.2f})")
                break

    # Now trace a few error cases
    print("\n" + "=" * 60)
    print("Tracing Error Cases")
    print("=" * 60)

    error_count = 0
    for inp_seq, out_seq, true_states in test_data[:50]:
        inp_idx = [in2i.get(x, 0) for x in inp_seq]
        out_idx = [out2i.get(x, 0) for x in out_seq]

        with torch.no_grad():
            out_probs, model_states = model.forward_with_states(inp_idx)
            preds = out_probs.argmax(-1).tolist()

        # Find first error
        for pos, (p, t) in enumerate(zip(preds, out_idx)):
            if p != t and error_count < 5:
                error_count += 1
                true_s = true_states[pos]
                expected_model_s = correspondence[true_s]
                actual_model_s = model_states[pos].argmax().item()
                model_conf = model_states[pos][actual_model_s].item()

                print(f"\nError #{error_count} at position {pos}:")
                print(f"  Input: {inp_seq[pos]}")
                print(f"  True state: {true_s}")
                print(f"  Expected model state: {expected_model_s}")
                print(f"  Actual model state: {actual_model_s} (conf: {model_conf:.2f})")
                print(f"  Expected output: {i2out[t]}")
                print(f"  Predicted output: {i2out[p]}")
                print(f"  Model state dist: {[f'{x:.2f}' for x in model_states[pos].numpy()[:5]]}")

                # Show sequence up to error
                print(f"  Sequence prefix: {' → '.join(inp_seq[:pos+1])}")
                print(f"  True state path: {' → '.join(true_states[:pos+1])}")
                break

    # Confusion analysis
    print("\n" + "=" * 60)
    print("State Confusion Analysis")
    print("=" * 60)

    confusion = defaultdict(lambda: defaultdict(int))

    for inp_seq, out_seq, true_states in test_data:
        inp_idx = [in2i.get(x, 0) for x in inp_seq]
        with torch.no_grad():
            _, model_states = model.forward_with_states(inp_idx)

        for true_s, model_s in zip(true_states, model_states):
            actual_model_s = model_s.argmax().item()
            expected_model_s = correspondence[true_s]
            if actual_model_s != expected_model_s:
                confusion[true_s][(actual_model_s, expected_model_s)] += 1

    print("\nWhen model state mismatches expected:")
    for true_s in auto['states']:
        if confusion[true_s]:
            print(f"\n  {true_s}:")
            for (actual, expected), count in sorted(confusion[true_s].items(), key=lambda x: -x[1])[:3]:
                print(f"    Expected state {expected}, got state {actual}: {count} times")


if __name__ == "__main__":
    main()
