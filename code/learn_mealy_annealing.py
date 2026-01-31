"""
Mealy Machine Learning with Temperature Annealing

Like DeepDFA: gradually sharpen softmax to make transitions crisp.
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


class FuzzyMealyAnnealing(nn.Module):
    """Fuzzy Mealy Machine with temperature annealing"""

    def __init__(self, num_states, num_inputs, num_outputs):
        super().__init__()
        self.num_states = num_states
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._T = nn.Parameter(torch.randn(num_states, num_inputs, num_states) * 0.1)
        self._O = nn.Parameter(torch.randn(num_states, num_inputs, num_outputs) * 0.1)
        self.temperature = 1.0  # Will be annealed during training

    def set_temperature(self, t):
        self.temperature = max(t, 0.01)  # Don't go below 0.01

    @property
    def T(self):
        return F.softmax(self._T / self.temperature, dim=-1)

    @property
    def O(self):
        return F.softmax(self._O / self.temperature, dim=-1)

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
        # With low temperature, this becomes less important
        # but still helps guide learning
        T = self.T
        O = self.O
        T_entropy = -(T * (T + 1e-10).log()).sum()
        O_entropy = -(O * (O + 1e-10).log()).sum()
        return (T_entropy + O_entropy) / (self.num_states * self.num_inputs)

    def get_sharpness(self):
        """Measure how close T and O are to one-hot"""
        T_max = self.T.max(dim=-1)[0].mean().item()
        O_max = self.O.max(dim=-1)[0].mean().item()
        return T_max, O_max


class FuzzyMealyBaseline(nn.Module):
    """Original without annealing"""

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

    def get_sharpness(self):
        T_max = self.T.max(dim=-1)[0].mean().item()
        O_max = self.O.max(dim=-1)[0].mean().item()
        return T_max, O_max


def train_annealing(model, traces, input_to_idx, output_to_idx, epochs=600, lr=0.05):
    """Train with temperature annealing"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Temperature schedule: 1.0 → 0.01 over training
    temp_start = 1.0
    temp_end = 0.01

    for epoch in range(epochs):
        # Anneal temperature
        progress = epoch / (epochs - 1)
        temperature = temp_start * (temp_end / temp_start) ** progress
        model.set_temperature(temperature)

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

        if epoch % 100 == 0:
            acc = total_correct / total_count if total_count > 0 else 0
            T_sharp, O_sharp = model.get_sharpness()
            print(f"  Epoch {epoch}: acc={acc:.2%}, τ={temperature:.4f}, "
                  f"T_max={T_sharp:.3f}, O_max={O_sharp:.3f}")

    return model


def train_baseline(model, traces, input_to_idx, output_to_idx, epochs=600, lr=0.05):
    """Train without annealing"""
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

        if epoch % 100 == 0:
            acc = total_correct / total_count if total_count > 0 else 0
            T_sharp, O_sharp = model.get_sharpness()
            print(f"  Epoch {epoch}: acc={acc:.2%}, "
                  f"T_max={T_sharp:.3f}, O_max={O_sharp:.3f}")

    return model


def evaluate(model, traces, input_to_idx, output_to_idx):
    total_symbols = 0
    correct_symbols = 0
    full_match = 0
    total_seqs = 0
    position_correct = {}
    position_total = {}

    for input_seq, output_seq in traces:
        inp_indices = [input_to_idx.get(x, 0) for x in input_seq]
        out_indices = [output_to_idx.get(x, 0) for x in output_seq]

        with torch.no_grad():
            out_probs = model(inp_indices)
            preds = out_probs.argmax(dim=-1).tolist()

        for i, (p, t) in enumerate(zip(preds, out_indices)):
            total_symbols += 1
            position_total[i] = position_total.get(i, 0) + 1
            if p == t:
                correct_symbols += 1
                position_correct[i] = position_correct.get(i, 0) + 1

        if preds == out_indices:
            full_match += 1
        total_seqs += 1

    return {
        'symbol_acc': correct_symbols / total_symbols if total_symbols > 0 else 0,
        'seq_acc': full_match / total_seqs if total_seqs > 0 else 0,
        'position_acc': {i: position_correct.get(i, 0) / position_total[i]
                        for i in sorted(position_total.keys())}
    }


def main():
    print("=" * 70)
    print("Temperature Annealing vs Baseline")
    print("=" * 70)

    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / "principle" / "BenchmarkBankcard-AartsEtAl2013" / "1_learnresult_MasterCard_fix.dot"

    if not dot_path.exists():
        print(f"File not found: {dot_path}")
        return

    print(f"\nLoading: {dot_path.name}")
    automaton = parse_dot_mealy(dot_path)
    print(f"Ground truth: {len(automaton['states'])} states, {len(automaton['transitions'])} transitions")

    input_to_idx = {x: i for i, x in enumerate(automaton['inputs'])}
    output_to_idx = {x: i for i, x in enumerate(automaton['outputs'])}

    random.seed(42)
    train_data = generate_traces(automaton, num_traces=100, max_length=10)
    test_short = generate_traces(automaton, num_traces=200, max_length=10)
    test_long = generate_traces(automaton, num_traces=200, max_length=30)
    test_very_long = generate_traces(automaton, num_traces=200, max_length=50)

    print(f"\nData:")
    print(f"  Train: {len(train_data)} traces, max_len=10")
    print(f"  Test short: max_len=10")
    print(f"  Test long: max_len=30")
    print(f"  Test very long: max_len=50")

    num_states = 10
    num_inputs = len(automaton['inputs'])
    num_outputs = len(automaton['outputs'])

    # Train Baseline
    print("\n" + "=" * 70)
    print("Training BASELINE...")
    print("=" * 70)
    baseline = FuzzyMealyBaseline(num_states, num_inputs, num_outputs)
    baseline = train_baseline(baseline, train_data, input_to_idx, output_to_idx, epochs=600)

    # Train Annealing
    print("\n" + "=" * 70)
    print("Training with TEMPERATURE ANNEALING...")
    print("=" * 70)
    annealing = FuzzyMealyAnnealing(num_states, num_inputs, num_outputs)
    annealing = train_annealing(annealing, train_data, input_to_idx, output_to_idx, epochs=600)

    # Set final temperature for evaluation
    annealing.set_temperature(0.01)

    # Evaluate
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    results = {}
    for name, model in [("Baseline", baseline), ("Annealing", annealing)]:
        results[name] = {
            'train': evaluate(model, train_data, input_to_idx, output_to_idx),
            'test_short': evaluate(model, test_short, input_to_idx, output_to_idx),
            'test_long': evaluate(model, test_long, input_to_idx, output_to_idx),
            'test_very_long': evaluate(model, test_very_long, input_to_idx, output_to_idx),
        }

    print("\n" + "-" * 70)
    print(f"{'Metric':<30} {'Baseline':>15} {'Annealing':>15} {'Δ':>10}")
    print("-" * 70)

    for split in ['train', 'test_short', 'test_long', 'test_very_long']:
        split_label = {'train': 'Train (len≤10)',
                      'test_short': 'Test (len≤10)',
                      'test_long': 'Test (len≤30)',
                      'test_very_long': 'Test (len≤50)'}[split]

        b_sym = results['Baseline'][split]['symbol_acc']
        a_sym = results['Annealing'][split]['symbol_acc']
        delta_sym = a_sym - b_sym
        better_sym = "✓" if delta_sym > 0.01 else ""
        print(f"{split_label} Symbol     {b_sym:>14.2%} {a_sym:>14.2%} {delta_sym:>+9.2%} {better_sym}")

        b_seq = results['Baseline'][split]['seq_acc']
        a_seq = results['Annealing'][split]['seq_acc']
        delta_seq = a_seq - b_seq
        better_seq = "✓" if delta_seq > 0.01 else ""
        print(f"{split_label} Sequence   {b_seq:>14.2%} {a_seq:>14.2%} {delta_seq:>+9.2%} {better_seq}")
        print()

    # Position analysis for very long sequences
    print("-" * 70)
    print("Position-wise accuracy on VERY LONG sequences (len≤50):")
    print("-" * 70)
    print(f"{'Position':<12} {'Baseline':>15} {'Annealing':>15} {'Δ':>10}")

    b_pos = results['Baseline']['test_very_long']['position_acc']
    a_pos = results['Annealing']['test_very_long']['position_acc']

    for start in range(0, 50, 10):
        end = min(start + 10, 50)
        b_vals = [b_pos.get(i, 0) for i in range(start, end) if i in b_pos]
        a_vals = [a_pos.get(i, 0) for i in range(start, end) if i in a_pos]
        if b_vals:
            b_avg = sum(b_vals) / len(b_vals)
            a_avg = sum(a_vals) / len(a_vals) if a_vals else 0
            delta = a_avg - b_avg
            better = "✓" if delta > 0.02 else ""
            print(f"{start:>2}-{end-1:<2}         {b_avg:>14.2%} {a_avg:>14.2%} {delta:>+9.2%} {better}")

    # Final sharpness
    print("\n" + "-" * 70)
    print("Final sharpness (higher = more crisp, 1.0 = perfect one-hot):")
    print("-" * 70)
    b_T, b_O = baseline.get_sharpness()
    a_T, a_O = annealing.get_sharpness()
    print(f"Baseline:  T_max={b_T:.4f}, O_max={b_O:.4f}")
    print(f"Annealing: T_max={a_T:.4f}, O_max={a_O:.4f}")


if __name__ == "__main__":
    main()
