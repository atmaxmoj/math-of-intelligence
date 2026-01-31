"""
Sample many traces from BB(4) with different starting configurations.
Deduplicate, collect until enough coverage, then learn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)


def encode_output(write, move):
    return write * 2 + (0 if move == 'L' else 1)


BB4_TRANSITIONS = {
    ('A', 0): (1, 'R', 'B'),
    ('A', 1): (1, 'L', 'B'),
    ('B', 0): (1, 'L', 'A'),
    ('B', 1): (0, 'L', 'C'),
    ('C', 0): (1, 'R', 'HALT'),
    ('C', 1): (1, 'L', 'D'),
    ('D', 0): (1, 'R', 'D'),
    ('D', 1): (0, 'R', 'A'),
}

STATES = ['A', 'B', 'C', 'D']
STATE2IDX = {s: i for i, s in enumerate(STATES)}


def run_bb4(initial_tape=None, initial_state='A', max_steps=30):
    """Run BB4 and return trace as tuple (for hashing)."""
    tape = dict(initial_tape) if initial_tape else {}
    head = 0
    state = initial_state
    trace = []

    for _ in range(max_steps):
        read_sym = tape.get(head, 0)
        if (state, read_sym) not in BB4_TRANSITIONS:
            break
        write_sym, move, next_state = BB4_TRANSITIONS[(state, read_sym)]
        trace.append((state, read_sym, write_sym, move))
        tape[head] = write_sym
        head += 1 if move == 'R' else -1
        state = next_state
        if next_state == 'HALT':
            break

    return tuple(trace)


def generate_traces_until_coverage(min_per_transition=10, max_traces=1000):
    """Generate traces until each transition is covered at least N times."""
    traces = set()
    coverage = Counter()
    all_transitions = [(s, sym) for s in STATES for sym in [0, 1]]

    attempts = 0
    while attempts < max_traces:
        # Random starting configuration
        tape = {}
        for pos in range(-3, 4):
            if random.random() < 0.4:
                tape[pos] = 1
        start_state = random.choice(STATES)

        trace = run_bb4(initial_tape=tape, initial_state=start_state, max_steps=30)

        if trace and trace not in traces:
            traces.add(trace)
            for state, sym, _, _ in trace:
                coverage[(state, sym)] += 1

        attempts += 1

        # Check coverage
        min_coverage = min(coverage.get(t, 0) for t in all_transitions)
        if min_coverage >= min_per_transition:
            break

    return traces, coverage


class FuzzyTM(nn.Module):
    def __init__(self, num_states=4):
        super().__init__()
        self.num_states = num_states
        self._T = nn.Parameter(torch.randn(num_states, 2, num_states) * 0.1)
        self._O = nn.Parameter(torch.randn(num_states, 2, 4) * 0.1)

    def forward(self, init_state_idx, input_seq):
        T = F.softmax(self._T, dim=-1)
        O = F.softmax(self._O, dim=-1)
        state = torch.zeros(self.num_states)
        state[init_state_idx] = 1.0
        outputs = []
        for sym in input_seq:
            out_prob = (O[:, sym, :] * state.unsqueeze(1)).sum(dim=0)
            outputs.append(out_prob)
            state = (T[:, sym, :] * state.unsqueeze(1)).sum(dim=0)
        return torch.stack(outputs)

    def extract_table(self):
        T = F.softmax(self._T, dim=-1)
        O = F.softmax(self._O, dim=-1)
        table = {}
        for s in range(self.num_states):
            for sym in range(2):
                next_s = T[s, sym].argmax().item()
                out = O[s, sym].argmax().item()
                write = out // 2
                move = 'L' if out % 2 == 0 else 'R'
                table[(s, sym)] = (write, move, next_s)
        return table


def train(model, traces, epochs=300):
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    trace_list = list(traces)

    for ep in range(epochs):
        random.shuffle(trace_list)
        total_correct = 0
        total_count = 0

        for trace in trace_list:
            init_state = trace[0][0]
            init_idx = STATE2IDX[init_state]
            input_seq = [t[1] for t in trace]
            output_seq = [encode_output(t[2], t[3]) for t in trace]

            out_probs = model(init_idx, input_seq)
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

    return total_correct / total_count


def main():
    print("=" * 60)
    print("BB(4) Learning with Sampled Traces")
    print("=" * 60)

    # Generate traces with good coverage
    print("\nGenerating traces until each transition appears 20+ times...")
    random.seed(42)
    traces, coverage = generate_traces_until_coverage(min_per_transition=20, max_traces=500)

    total_steps = sum(len(t) for t in traces)
    print(f"\nGenerated {len(traces)} unique traces, {total_steps} total steps")
    print("\nTransition coverage:")
    for (s, sym), cnt in sorted(coverage.items()):
        bar = "█" * (cnt // 5)
        print(f"  ({s}, {sym}): {cnt:>3} {bar}")

    # Train
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    torch.manual_seed(42)
    model = FuzzyTM(num_states=4)
    final_acc = train(model, traces, epochs=300)

    # Extract learned table
    print("\n" + "=" * 60)
    print("Learned vs Ground Truth")
    print("=" * 60)

    learned = model.extract_table()

    print(f"\n{'Trans':<10} {'Learned':<20} {'Truth':<20} {'Match'}")
    print("-" * 55)

    matches = 0
    for s in range(4):
        for sym in range(2):
            state_name = STATES[s]
            l_write, l_move, l_next = learned[(s, sym)]
            t_write, t_move, t_next = BB4_TRANSITIONS[(state_name, sym)]
            t_next_idx = STATE2IDX.get(t_next, -1)

            match = (l_write == t_write and l_move == t_move and l_next == t_next_idx)
            if match:
                matches += 1

            print(f"({s}, {sym})    ({l_write}, {l_move}, {l_next})"
                  f"           ({t_write}, {t_move}, {t_next_idx})"
                  f"           {'✓' if match else '✗'}")

    print(f"\nTransitions correct: {matches}/8")

    # Test on original BB(4) trace
    print("\n" + "=" * 60)
    print("Test on original BB(4) (107 steps from empty tape)")
    print("=" * 60)

    original_trace = run_bb4(max_steps=200)
    input_seq = [t[1] for t in original_trace]
    output_seq = [encode_output(t[2], t[3]) for t in original_trace]

    with torch.no_grad():
        preds = model(STATE2IDX['A'], input_seq).argmax(dim=-1).tolist()

    correct = sum(p == t for p, t in zip(preds, output_seq))
    print(f"\nAccuracy: {correct}/{len(output_seq)} = {correct/len(output_seq):.1%}")


if __name__ == "__main__":
    main()
