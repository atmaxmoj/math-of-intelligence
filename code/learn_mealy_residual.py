"""
Mealy Machine Learning with Residual Connection

Test if residual helps prevent fuzzy state drift on long sequences.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random


def parse_dot_mealy(filepath):
    """Parse Mealy machine from .dot file"""
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
            inp = inp.strip()
            out = out.strip()
            states.add(src)
            states.add(dst)
            inputs.add(inp)
            outputs.add(out)
            transitions[(src, inp)] = (dst, out)

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
    """Generate input-output traces"""
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


class FuzzyMealyResidual(nn.Module):
    """Fuzzy Mealy Machine with residual connection to prevent state drift"""

    def __init__(self, num_states, num_inputs, num_outputs):
        super().__init__()
        self.num_states = num_states
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._T = nn.Parameter(torch.randn(num_states, num_inputs, num_states) * 0.01)
        self._O = nn.Parameter(torch.randn(num_states, num_inputs, num_outputs) * 0.01)
        # Learnable residual weight (initialized to keep ~10% of old state)
        self._alpha = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12

    @property
    def T(self):
        return F.softmax(self._T, dim=-1)

    @property
    def O(self):
        return F.softmax(self._O, dim=-1)

    @property
    def alpha(self):
        return torch.sigmoid(self._alpha)

    def forward(self, input_seq, initial_state=0):
        state_dist = torch.zeros(self.num_states)
        state_dist[initial_state] = 1.0
        output_probs = []

        for inp in input_seq:
            # Output
            out_prob = (self.O[:, inp, :] * state_dist.unsqueeze(1)).sum(dim=0)
            output_probs.append(out_prob)

            # Transition with residual
            transition = (self.T[:, inp, :] * state_dist.unsqueeze(1)).sum(dim=0)
            # Residual: keep some of old state to prevent drift
            new_state_dist = self.alpha * state_dist + (1 - self.alpha) * transition
            # Renormalize
            state_dist = new_state_dist / (new_state_dist.sum() + 1e-10)

        return torch.stack(output_probs)

    def sparsity_loss(self):
        T = self.T
        O = self.O
        T_entropy = -(T * (T + 1e-10).log()).sum()
        O_entropy = -(O * (O + 1e-10).log()).sum()
        return (T_entropy + O_entropy) / (self.num_states * self.num_inputs)


class FuzzyMealyBaseline(nn.Module):
    """Original Fuzzy Mealy Machine without residual (for comparison)"""

    def __init__(self, num_states, num_inputs, num_outputs):
        super().__init__()
        self.num_states = num_states
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._T = nn.Parameter(torch.randn(num_states, num_inputs, num_states) * 0.01)
        self._O = nn.Parameter(torch.randn(num_states, num_inputs, num_outputs) * 0.01)

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


def train_model(model, traces, input_to_idx, output_to_idx, epochs=500, lr=0.05):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

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
            loss = loss + model.sparsity_loss() * 0.001

            total_loss += loss.item()
            preds = out_probs.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += len(targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch % 100 == 0:
            acc = total_correct / total_count if total_count > 0 else 0
            alpha_str = f", α={model.alpha.item():.3f}" if hasattr(model, 'alpha') else ""
            print(f"  Epoch {epoch}: loss={total_loss/max(len(traces),1):.4f}, acc={acc:.2%}{alpha_str}")

    return model


def evaluate(model, traces, input_to_idx, output_to_idx):
    total_symbols = 0
    correct_symbols = 0
    full_match = 0
    total_seqs = 0

    # Per-position accuracy
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
    print("Residual vs Baseline: Fuzzy Mealy Machine")
    print("=" * 70)

    # Load MasterCard benchmark
    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / "principle" / "BenchmarkBankcard-AartsEtAl2013" / "1_learnresult_MasterCard_fix.dot"

    if not dot_path.exists():
        print(f"File not found: {dot_path}")
        return

    print(f"\nLoading: {dot_path.name}")
    automaton = parse_dot_mealy(dot_path)

    print(f"Ground truth: {len(automaton['states'])} states, {len(automaton['transitions'])} transitions")

    input_to_idx = {x: i for i, x in enumerate(automaton['inputs'])}
    output_to_idx = {x: i for i, x in enumerate(automaton['outputs'])}

    # Generate data
    random.seed(42)
    train_data = generate_traces(automaton, num_traces=100, max_length=10)
    test_short = generate_traces(automaton, num_traces=200, max_length=10)
    test_long = generate_traces(automaton, num_traces=200, max_length=30)  # Even longer!

    print(f"\nData:")
    print(f"  Train: {len(train_data)} traces, max_len=10")
    print(f"  Test short: {len(test_short)} traces, max_len=10")
    print(f"  Test long: {len(test_long)} traces, max_len=30")

    num_states = 10
    num_inputs = len(automaton['inputs'])
    num_outputs = len(automaton['outputs'])

    # Train Baseline
    print("\n" + "=" * 70)
    print("Training BASELINE (no residual)...")
    print("=" * 70)
    baseline = FuzzyMealyBaseline(num_states, num_inputs, num_outputs)
    baseline = train_model(baseline, train_data, input_to_idx, output_to_idx, epochs=500)

    # Train Residual
    print("\n" + "=" * 70)
    print("Training RESIDUAL...")
    print("=" * 70)
    residual = FuzzyMealyResidual(num_states, num_inputs, num_outputs)
    residual = train_model(residual, train_data, input_to_idx, output_to_idx, epochs=500)

    # Evaluate
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    results = {}
    for name, model in [("Baseline", baseline), ("Residual", residual)]:
        results[name] = {
            'train': evaluate(model, train_data, input_to_idx, output_to_idx),
            'test_short': evaluate(model, test_short, input_to_idx, output_to_idx),
            'test_long': evaluate(model, test_long, input_to_idx, output_to_idx),
        }

    # Print comparison table
    print("\n" + "-" * 70)
    print(f"{'Metric':<25} {'Baseline':>15} {'Residual':>15} {'Δ':>10}")
    print("-" * 70)

    for split in ['train', 'test_short', 'test_long']:
        split_label = {'train': 'Train (len≤10)',
                      'test_short': 'Test Short (len≤10)',
                      'test_long': 'Test Long (len≤30)'}[split]

        b_sym = results['Baseline'][split]['symbol_acc']
        r_sym = results['Residual'][split]['symbol_acc']
        print(f"{split_label} Symbol Acc    {b_sym:>14.2%} {r_sym:>14.2%} {r_sym-b_sym:>+9.2%}")

        b_seq = results['Baseline'][split]['seq_acc']
        r_seq = results['Residual'][split]['seq_acc']
        print(f"{split_label} Seq Acc       {b_seq:>14.2%} {r_seq:>14.2%} {r_seq-b_seq:>+9.2%}")
        print()

    # Position-wise accuracy for long sequences
    print("-" * 70)
    print("Position-wise accuracy on LONG sequences (len≤30):")
    print("-" * 70)
    print(f"{'Position':<10} {'Baseline':>15} {'Residual':>15}")

    b_pos = results['Baseline']['test_long']['position_acc']
    r_pos = results['Residual']['test_long']['position_acc']

    # Group by ranges
    for start in [0, 5, 10, 15, 20, 25]:
        end = min(start + 5, 30)
        b_avg = sum(b_pos.get(i, 0) for i in range(start, end)) / max(1, sum(1 for i in range(start, end) if i in b_pos))
        r_avg = sum(r_pos.get(i, 0) for i in range(start, end)) / max(1, sum(1 for i in range(start, end) if i in r_pos))
        if any(i in b_pos for i in range(start, end)):
            print(f"{start:>2}-{end-1:<2}       {b_avg:>14.2%} {r_avg:>14.2%}")

    if hasattr(residual, 'alpha'):
        print(f"\nLearned residual α = {residual.alpha.item():.4f}")


if __name__ == "__main__":
    main()
