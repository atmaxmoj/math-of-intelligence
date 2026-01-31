"""
Large Mealy Machine Learning - Bankcard Protocol

Test on real-world protocol: MasterCard EMV
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random


def parse_dot_mealy(filepath):
    """Parse Mealy machine from .dot file (handles various formats)"""
    with open(filepath) as f:
        content = f.read()

    states = set()
    transitions = {}
    inputs = set()
    outputs = set()
    initial = None

    # Find states (various formats)
    for match in re.finditer(r'^(s\d+)\s*\[', content, re.MULTILINE):
        states.add(match.group(1))

    for match in re.finditer(r'^(q\d+)\s*\[', content, re.MULTILINE):
        states.add(match.group(1))

    # Find initial state
    for match in re.finditer(r'(\w+)\s*\[color="red"\]', content):
        initial = match.group(1)
        states.add(initial)

    # Find transitions: s0 -> s1[label="input/output"]
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
            # Only pick inputs that are valid from current state
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


class FuzzyMealyMachine(nn.Module):
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


def train_on_traces(model, traces, input_to_idx, output_to_idx, epochs=1000, lr=0.05):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

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
            print(f"Epoch {epoch}: loss={total_loss/max(len(traces),1):.4f}, acc={acc:.2%}")

    return model


def evaluate(model, traces, input_to_idx, output_to_idx):
    total_symbols = 0
    correct_symbols = 0
    full_match = 0
    total_seqs = 0

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
        total_seqs += 1

    return {
        'symbol_acc': correct_symbols / total_symbols if total_symbols > 0 else 0,
        'seq_acc': full_match / total_seqs if total_seqs > 0 else 0,
        'total_seqs': total_seqs
    }


def main():
    print("=" * 70)
    print("Large Mealy Machine Learning - MasterCard Protocol")
    print("=" * 70)

    # Load MasterCard benchmark
    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / "principle" / "BenchmarkBankcard-AartsEtAl2013" / "1_learnresult_MasterCard_fix.dot"

    if not dot_path.exists():
        print(f"File not found: {dot_path}")
        return

    print(f"\nLoading: {dot_path.name}")
    automaton = parse_dot_mealy(dot_path)

    true_states = len(automaton['states'])
    true_transitions = len(automaton['transitions'])
    num_inputs = len(automaton['inputs'])
    num_outputs = len(automaton['outputs'])

    print(f"\n{'='*70}")
    print(f"Ground Truth:")
    print(f"  States: {true_states}")
    print(f"  Transitions: {true_transitions}")
    print(f"  Input symbols: {num_inputs}")
    print(f"  Output symbols: {num_outputs}")
    print(f"  Initial: {automaton['initial']}")
    print(f"{'='*70}")

    if true_states == 0 or num_inputs == 0:
        print("Failed to parse automaton!")
        return

    input_to_idx = {x: i for i, x in enumerate(automaton['inputs'])}
    output_to_idx = {x: i for i, x in enumerate(automaton['outputs'])}
    idx_to_output = {i: x for x, i in output_to_idx.items()}

    # Experiment settings
    model_states = min(true_states * 2, 20)  # Cap at 20 for memory
    train_traces_num = 100
    train_max_len = 10
    test_traces_num = 200
    test_max_len = 20

    print(f"\nExperiment Settings:")
    print(f"  Model states: {model_states} (true: {true_states})")
    print(f"  Train: {train_traces_num} traces, max_len={train_max_len}")
    print(f"  Test: {test_traces_num} traces, max_len={test_max_len}")

    # Generate data
    random.seed(42)
    train_data = generate_traces(automaton, num_traces=train_traces_num, max_length=train_max_len)
    test_data = generate_traces(automaton, num_traces=test_traces_num, max_length=test_max_len)

    print(f"\nGenerated {len(train_data)} train traces, {len(test_data)} test traces")

    if len(train_data) == 0:
        print("No traces generated! Check automaton parsing.")
        return

    # Show sample trace
    if train_data:
        sample_in, sample_out = train_data[0]
        print(f"Sample trace: {sample_in[:5]}... -> {sample_out[:5]}...")

    # Create model
    model = FuzzyMealyMachine(
        num_states=model_states,
        num_inputs=num_inputs,
        num_outputs=num_outputs
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")

    # Train
    print("\n" + "=" * 70)
    print("Training...")
    print("=" * 70)
    model = train_on_traces(model, train_data, input_to_idx, output_to_idx,
                           epochs=1000, lr=0.05)

    # Evaluate
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    train_results = evaluate(model, train_data, input_to_idx, output_to_idx)
    test_results = evaluate(model, test_data, input_to_idx, output_to_idx)

    print(f"\nTrain:")
    print(f"  Symbol accuracy: {train_results['symbol_acc']:.2%}")
    print(f"  Sequence accuracy: {train_results['seq_acc']:.2%}")

    print(f"\nTest (longer sequences):")
    print(f"  Symbol accuracy: {test_results['symbol_acc']:.2%}")
    print(f"  Sequence accuracy: {test_results['seq_acc']:.2%}")

    # Show some predictions
    print("\n" + "-" * 70)
    print("Sample predictions:")
    print("-" * 70)
    for i, (inp_seq, out_seq) in enumerate(test_data[:3]):
        inp_indices = [input_to_idx.get(x, 0) for x in inp_seq]
        with torch.no_grad():
            preds = model(inp_indices).argmax(dim=-1).tolist()
        pred_out = [idx_to_output.get(p, '?') for p in preds]

        # Count matches
        matches = sum(1 for p, t in zip(pred_out, out_seq) if p == t)
        print(f"\nTrace {i+1} ({matches}/{len(out_seq)} correct):")
        print(f"  Input:  {inp_seq[:5]}...")
        print(f"  Target: {out_seq[:5]}...")
        print(f"  Pred:   {pred_out[:5]}...")


if __name__ == "__main__":
    main()
