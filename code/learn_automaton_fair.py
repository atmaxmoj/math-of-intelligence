"""
Fair Automata Learning Experiment

Changes from original:
1. Model has MORE states than ground truth (doesn't know true size)
2. Less training data
3. Test on longer sequences (generalization)
4. Report per-symbol accuracy, not just full-sequence match
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random


def parse_dot(filepath):
    """Parse Mealy machine from .dot file"""
    with open(filepath) as f:
        content = f.read()

    states = set()
    transitions = {}
    inputs = set()
    outputs = set()
    initial = None

    init_match = re.search(r'__start\d*\s*->\s*(\w+)', content)
    if init_match:
        initial = init_match.group(1)

    for match in re.finditer(r'(\w+)\s*\[shape="circle"', content):
        states.add(match.group(1))

    # Also match states without shape specification
    for match in re.finditer(r'^(\w+)\s*\[label=', content, re.MULTILINE):
        if not match.group(1).startswith('__'):
            states.add(match.group(1))

    for match in re.finditer(r'(\w+)\s*->\s*(\w+)\s*\[label="([^/]+)\s*/\s*([^"]+)"\]', content):
        src, dst, inp, out = match.groups()
        states.add(src)
        states.add(dst)
        inputs.add(inp)
        outputs.add(out)
        transitions[(src, inp)] = (dst, out)

    # If no initial found, use first state
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
    """Generate input-output traces from automaton"""
    traces = []
    for _ in range(num_traces):
        state = automaton['initial']
        length = random.randint(1, max_length)
        input_seq = []
        output_seq = []
        for _ in range(length):
            inp = random.choice(automaton['inputs'])
            if (state, inp) in automaton['transitions']:
                next_state, out = automaton['transitions'][(state, inp)]
                input_seq.append(inp)
                output_seq.append(out)
                state = next_state
            else:
                break
        if input_seq:
            traces.append((input_seq, output_seq))
    return traces


class FuzzyMealyMachine(nn.Module):
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
        return T_entropy + O_entropy

    def count_active_states(self, threshold=0.1):
        """Count states that are actually used (have significant incoming probability)"""
        T = self.T.detach()
        # A state is "active" if it has significant outgoing transitions
        max_out = T.max(dim=-1)[0].max(dim=-1)[0]  # max transition prob for each state
        return (max_out > threshold).sum().item()


def train_on_traces(model, traces, input_to_idx, output_to_idx, epochs=500, lr=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_count = 0

        random.shuffle(traces)

        for input_seq, output_seq in traces:
            inp_indices = [input_to_idx[x] for x in input_seq]
            out_indices = [output_to_idx[x] for x in output_seq]

            out_probs = model(inp_indices)
            targets = torch.tensor(out_indices)
            loss = F.cross_entropy(out_probs, targets)
            loss = loss + model.sparsity_loss() * 0.0001  # smaller sparsity weight

            total_loss += loss.item()
            preds = out_probs.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += len(targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            acc = total_correct / total_count
            active = model.count_active_states()
            print(f"Epoch {epoch}: loss={total_loss/len(traces):.4f}, "
                  f"train_acc={acc:.2%}, active_states={active}/{model.num_states}")

    return model


def evaluate(model, traces, input_to_idx, output_to_idx):
    """Evaluate with multiple metrics"""
    total_symbols = 0
    correct_symbols = 0
    full_match = 0
    total_seqs = 0

    for input_seq, output_seq in traces:
        inp_indices = [input_to_idx[x] for x in input_seq]
        out_indices = [output_to_idx[x] for x in output_seq]

        with torch.no_grad():
            out_probs = model(inp_indices)
            preds = out_probs.argmax(dim=-1).tolist()

        # Per-symbol accuracy
        for p, t in zip(preds, out_indices):
            total_symbols += 1
            if p == t:
                correct_symbols += 1

        # Full sequence match
        if preds == out_indices:
            full_match += 1
        total_seqs += 1

    return {
        'symbol_acc': correct_symbols / total_symbols,
        'seq_acc': full_match / total_seqs,
        'total_seqs': total_seqs
    }


def main():
    print("=" * 60)
    print("FAIR Automata Learning Experiment")
    print("=" * 60)

    # Load benchmark
    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / "principle" / "BenchmarkToyModels" / "cacm.dot"

    print(f"\nLoading: {dot_path.name}")
    automaton = parse_dot(dot_path)

    true_states = len(automaton['states'])
    true_transitions = len(automaton['transitions'])

    print(f"Ground truth: {true_states} states, {true_transitions} transitions")
    print(f"Inputs: {automaton['inputs']}")
    print(f"Outputs: {automaton['outputs']}")

    input_to_idx = {x: i for i, x in enumerate(automaton['inputs'])}
    output_to_idx = {x: i for i, x in enumerate(automaton['outputs'])}
    idx_to_output = {i: x for x, i in output_to_idx.items()}

    # ========================================
    # Experiment settings
    # ========================================
    model_states = true_states * 2  # Give model 2x states (doesn't know true number)
    train_traces = 30  # Fewer training traces
    train_max_len = 8  # Shorter training sequences
    test_traces = 100
    test_max_len = 25  # Longer test sequences (generalization!)

    print(f"\n{'='*60}")
    print(f"Experiment: model_states={model_states} (true={true_states})")
    print(f"            train_traces={train_traces}, train_max_len={train_max_len}")
    print(f"            test_traces={test_traces}, test_max_len={test_max_len}")
    print(f"{'='*60}")

    # Generate data
    random.seed(42)
    train_data = generate_traces(automaton, num_traces=train_traces, max_length=train_max_len)
    test_data = generate_traces(automaton, num_traces=test_traces, max_length=test_max_len)

    print(f"\nTrain: {len(train_data)} traces")
    print(f"Test:  {len(test_data)} traces")

    # Create model with MORE states than ground truth
    model = FuzzyMealyMachine(
        num_states=model_states,
        num_inputs=len(automaton['inputs']),
        num_outputs=len(automaton['outputs'])
    )

    # Train
    print("\nTraining...")
    print("-" * 40)
    model = train_on_traces(model, train_data, input_to_idx, output_to_idx, epochs=500)

    # Evaluate
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    train_results = evaluate(model, train_data, input_to_idx, output_to_idx)
    test_results = evaluate(model, test_data, input_to_idx, output_to_idx)

    print(f"\nTrain (len≤{train_max_len}):")
    print(f"  Symbol accuracy: {train_results['symbol_acc']:.2%}")
    print(f"  Sequence accuracy: {train_results['seq_acc']:.2%}")

    print(f"\nTest (len≤{test_max_len}, LONGER than train):")
    print(f"  Symbol accuracy: {test_results['symbol_acc']:.2%}")
    print(f"  Sequence accuracy: {test_results['seq_acc']:.2%}")

    print(f"\nModel used {model.count_active_states()}/{model_states} states")
    print(f"(Ground truth has {true_states} states)")

    # Show some predictions
    print("\n" + "-" * 40)
    print("Sample predictions (test set):")
    print("-" * 40)
    for i, (inp_seq, out_seq) in enumerate(test_data[:5]):
        inp_indices = [input_to_idx[x] for x in inp_seq]
        with torch.no_grad():
            preds = model(inp_indices).argmax(dim=-1).tolist()
        pred_out = [idx_to_output[p] for p in preds]
        match = "✓" if pred_out == out_seq else "✗"
        print(f"{match} Input:  {' '.join(inp_seq)}")
        print(f"  Target: {' '.join(out_seq)}")
        print(f"  Pred:   {' '.join(pred_out)}")
        print()


if __name__ == "__main__":
    main()
