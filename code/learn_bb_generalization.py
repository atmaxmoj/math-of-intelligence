"""
Train on different tape lengths, test on multiple lengths.
See generalization pattern.
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

    def run(self, max_steps=1000):
        tape = {}
        head = 0
        state = self.initial_state
        trace = []
        states = [state]

        for _ in range(max_steps):
            read_sym = tape.get(head, 0)
            if (state, read_sym) not in self.transitions:
                break
            write_sym, move, next_state = self.transitions[(state, read_sym)]
            trace.append((read_sym, write_sym, move))
            tape[head] = write_sym
            head += 1 if move == 'R' else -1
            state = next_state
            states.append(state)
            if next_state == 'HALT':
                break

        return trace, states


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
        self._init = nn.Parameter(torch.zeros(num_states))

    def forward(self, input_seq):
        T = F.softmax(self._T, dim=-1)
        O = F.softmax(self._O, dim=-1)
        state = F.softmax(self._init, dim=0)
        outputs = []
        for sym in input_seq:
            out_prob = (O[:, sym, :] * state.unsqueeze(1)).sum(dim=0)
            outputs.append(out_prob)
            state = (T[:, sym, :] * state.unsqueeze(1)).sum(dim=0)
        return torch.stack(outputs)


def train(model, input_seq, output_seq, epochs=300):
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(epochs):
        out_probs = model(input_seq)
        loss = F.cross_entropy(out_probs, torch.tensor(output_seq))
        opt.zero_grad()
        loss.backward()
        opt.step()


def evaluate(model, input_seq, output_seq):
    with torch.no_grad():
        preds = model(input_seq).argmax(dim=-1).tolist()
    return sum(p == t for p, t in zip(preds, output_seq)) / len(output_seq)


def main():
    print("=" * 70)
    print("BB(4) Generalization: Train Length vs Test Length")
    print("=" * 70)

    trace, states = BB4.run()
    full_input = [t[0] for t in trace]
    full_output = [encode_output(t[1], t[2]) for t in trace]
    total_len = len(trace)

    print(f"BB(4) total: {total_len} steps\n")

    train_lengths = [10, 20, 30, 50, 70, 107]
    test_lengths = [10, 20, 30, 50, 70, 107]

    # Train models for each length
    models = {}
    for train_len in train_lengths:
        print(f"Training on {train_len} steps...", end=" ", flush=True)
        torch.manual_seed(42)
        model = FuzzyTM(num_states=4)
        train(model, full_input[:train_len], full_output[:train_len], epochs=300)
        models[train_len] = model
        print("done")

    # Results table
    print("\n" + "=" * 70)
    print("Results: Accuracy (row=train length, col=test length)")
    print("=" * 70)

    header = f"{'Train↓ Test→':<14}" + "".join(f"{t:>10}" for t in test_lengths)
    print(header)
    print("-" * (14 + 10 * len(test_lengths)))

    for train_len in train_lengths:
        model = models[train_len]
        row = f"{train_len:<14}"
        for test_len in test_lengths:
            acc = evaluate(model, full_input[:test_len], full_output[:test_len])
            row += f"{acc:>9.0%} "
        print(row)

    # Analysis
    print("\n" + "=" * 70)
    print("Key observations:")
    print("=" * 70)

    # Diagonal (train = test)
    print("\nDiagonal (train len = test len):")
    for train_len in train_lengths:
        acc = evaluate(models[train_len], full_input[:train_len], full_output[:train_len])
        print(f"  {train_len:>3} steps: {acc:.0%}")

    # Generalization to full
    print("\nGeneralization to full 107 steps:")
    for train_len in train_lengths:
        acc = evaluate(models[train_len], full_input, full_output)
        print(f"  Train {train_len:>3} → Test 107: {acc:.0%}")


if __name__ == "__main__":
    main()
