"""
Multi-trace learning: Sample from multiple starting configurations.

For a TM, we can vary:
1. Initial tape content (not all zeros)
2. Initial head position
3. Or: directly sample (state, symbol) pairs uniformly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys

sys.stdout.reconfigure(line_buffering=True)


def encode_output(write, move):
    return write * 2 + (0 if move == 'L' else 1)


class TuringMachine:
    def __init__(self, transitions, initial_state='A'):
        self.transitions = transitions
        self.initial_state = initial_state
        self.states = list(set(s for s, _ in transitions.keys()))

    def run_from(self, initial_tape=None, initial_head=0, initial_state=None, max_steps=50):
        """Run from a specific configuration."""
        tape = dict(initial_tape) if initial_tape else {}
        head = initial_head
        state = initial_state or self.initial_state
        trace = []
        state_seq = [state]

        for _ in range(max_steps):
            read_sym = tape.get(head, 0)
            if (state, read_sym) not in self.transitions:
                break
            write_sym, move, next_state = self.transitions[(state, read_sym)]
            trace.append((state, read_sym, write_sym, move, next_state))
            tape[head] = write_sym
            head += 1 if move == 'R' else -1
            state = next_state
            state_seq.append(state)
            if next_state == 'HALT':
                break

        return trace

    def generate_diverse_traces(self, num_traces=20, max_steps=30):
        """Generate traces from diverse starting points."""
        traces = []

        # 1. Standard trace from empty tape
        traces.append(self.run_from(max_steps=max_steps))

        # 2. Traces from random tape configurations
        for _ in range(num_traces - 1):
            # Random tape with some 1s
            tape = {}
            for pos in range(-5, 6):
                if random.random() < 0.3:
                    tape[pos] = 1

            # Random starting state (not HALT)
            start_state = random.choice(self.states)

            trace = self.run_from(
                initial_tape=tape,
                initial_head=0,
                initial_state=start_state,
                max_steps=max_steps
            )
            if trace:  # Only add non-empty traces
                traces.append(trace)

        return traces


BB4 = TuringMachine({
    ('A', 0): (1, 'R', 'B'),
    ('A', 1): (1, 'L', 'B'),
    ('B', 0): (1, 'L', 'A'),
    ('B', 1): (0, 'L', 'C'),
    ('C', 0): (1, 'R', 'HALT'),
    ('C', 1): (1, 'L', 'D'),
    ('D', 0): (1, 'R', 'D'),
    ('D', 1): (0, 'R', 'A'),
})


class FuzzyTM(nn.Module):
    def __init__(self, num_states):
        super().__init__()
        self.num_states = num_states
        self._T = nn.Parameter(torch.randn(num_states, 2, num_states) * 0.1)
        self._O = nn.Parameter(torch.randn(num_states, 2, 4) * 0.1)

    def forward(self, state_dist, input_seq):
        """Forward pass with explicit initial state distribution."""
        T = F.softmax(self._T, dim=-1)
        O = F.softmax(self._O, dim=-1)
        state = state_dist
        outputs = []
        for sym in input_seq:
            out_prob = (O[:, sym, :] * state.unsqueeze(1)).sum(dim=0)
            outputs.append(out_prob)
            state = (T[:, sym, :] * state.unsqueeze(1)).sum(dim=0)
        return torch.stack(outputs)


def traces_to_training_data(traces, state_to_idx):
    """Convert traces to (initial_state, input_seq, output_seq) tuples."""
    data = []
    for trace in traces:
        if not trace:
            continue
        init_state = trace[0][0]  # First state
        input_seq = [t[1] for t in trace]  # read symbols
        output_seq = [encode_output(t[2], t[3]) for t in trace]  # (write, move)
        data.append((state_to_idx[init_state], input_seq, output_seq))
    return data


def train(model, data, state_to_idx, epochs=300):
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    num_states = model.num_states

    for ep in range(epochs):
        random.shuffle(data)
        total_correct = 0
        total_count = 0

        for init_state_idx, input_seq, output_seq in data:
            # One-hot initial state
            init_dist = torch.zeros(num_states)
            init_dist[init_state_idx] = 1.0

            out_probs = model(init_dist, input_seq)
            targets = torch.tensor(output_seq)
            loss = F.cross_entropy(out_probs, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            preds = out_probs.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += len(targets)

        if ep % 100 == 0:
            print(f"  Epoch {ep}: {total_correct/total_count:.1%}")


def evaluate(model, data, state_to_idx):
    num_states = model.num_states
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for init_state_idx, input_seq, output_seq in data:
            init_dist = torch.zeros(num_states)
            init_dist[init_state_idx] = 1.0

            preds = model(init_dist, input_seq).argmax(dim=-1).tolist()
            total_correct += sum(p == t for p, t in zip(preds, output_seq))
            total_count += len(output_seq)

    return total_correct / total_count


def main():
    print("=" * 60)
    print("Multi-Trace Learning for BB(4)")
    print("=" * 60)

    state_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    # Generate the original single trace
    single_trace = [BB4.run_from(max_steps=107)]
    single_data = traces_to_training_data(single_trace, state_to_idx)

    # Check transition coverage in single trace
    from collections import Counter
    single_coverage = Counter()
    for trace in single_trace:
        for state, sym, _, _, _ in trace:
            single_coverage[(state, sym)] += 1

    print(f"\nSingle trace: {len(single_trace[0])} steps")
    print("Transition coverage:")
    for (s, sym), cnt in sorted(single_coverage.items()):
        print(f"  ({s}, {sym}): {cnt}")

    # Generate diverse traces
    random.seed(42)
    diverse_traces = BB4.generate_diverse_traces(num_traces=50, max_steps=30)
    diverse_data = traces_to_training_data(diverse_traces, state_to_idx)

    diverse_coverage = Counter()
    for trace in diverse_traces:
        for state, sym, _, _, _ in trace:
            diverse_coverage[(state, sym)] += 1

    total_steps = sum(len(t) for t in diverse_traces)
    print(f"\nDiverse traces: {len(diverse_traces)} traces, {total_steps} total steps")
    print("Transition coverage:")
    for (s, sym), cnt in sorted(diverse_coverage.items()):
        print(f"  ({s}, {sym}): {cnt}")

    # Train on single trace
    print("\n" + "=" * 60)
    print("Training on SINGLE trace (107 steps from empty tape)")
    print("=" * 60)
    torch.manual_seed(42)
    model_single = FuzzyTM(num_states=4)
    train(model_single, single_data, state_to_idx, epochs=300)

    # Train on diverse traces
    print("\n" + "=" * 60)
    print("Training on DIVERSE traces (50 traces, various starts)")
    print("=" * 60)
    torch.manual_seed(42)
    model_diverse = FuzzyTM(num_states=4)
    train(model_diverse, diverse_data, state_to_idx, epochs=300)

    # Evaluate both on the original trace
    print("\n" + "=" * 60)
    print("Results on original BB(4) trace (107 steps)")
    print("=" * 60)

    acc_single = evaluate(model_single, single_data, state_to_idx)
    acc_diverse = evaluate(model_diverse, single_data, state_to_idx)

    print(f"\n  Single-trace model:  {acc_single:.1%}")
    print(f"  Diverse-trace model: {acc_diverse:.1%}")

    # Extract and compare transition tables
    print("\n" + "=" * 60)
    print("Learned Transition Tables")
    print("=" * 60)

    def extract_table(model):
        T = F.softmax(model._T, dim=-1)
        O = F.softmax(model._O, dim=-1)
        table = {}
        for s in range(4):
            for sym in range(2):
                next_s = T[s, sym].argmax().item()
                out = O[s, sym].argmax().item()
                write = out // 2
                move = 'L' if out % 2 == 0 else 'R'
                table[(s, sym)] = (write, move, next_s)
        return table

    print("\nGround truth:")
    for (state, sym), (write, move, next_state) in BB4.transitions.items():
        s_idx = state_to_idx[state]
        n_idx = state_to_idx.get(next_state, -1)
        print(f"  ({s_idx}, {sym}) → ({write}, {move}, {n_idx})")

    print("\nSingle-trace model:")
    for (s, sym), (write, move, next_s) in sorted(extract_table(model_single).items()):
        print(f"  ({s}, {sym}) → ({write}, {move}, {next_s})")

    print("\nDiverse-trace model:")
    for (s, sym), (write, move, next_s) in sorted(extract_table(model_diverse).items()):
        print(f"  ({s}, {sym}) → ({write}, {move}, {next_s})")


if __name__ == "__main__":
    main()
