"""
Test: Does longer trace → better accuracy for Busy Beaver?

For BB(4) with 107 steps, train on first N steps, test on full trace.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys

sys.stdout.reconfigure(line_buffering=True)


def encode_output(write, move):
    return write * 2 + (0 if move == 'L' else 1)

def decode_output(code):
    write = code // 2
    move = 'L' if code % 2 == 0 else 'R'
    return write, move


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

        ones = sum(1 for v in tape.values() if v == 1)
        return trace, states, ones


# BB(4): 4 states, 107 steps, 13 ones
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

# BB(5) champion (runs for 47,176,870 steps!) - too long, use a simpler 5-state
# Let's use a known 5-state TM that runs for a moderate number of steps
BB5_simple = TuringMachine({
    ('A', 0): (1, 'R', 'B'),
    ('A', 1): (1, 'L', 'C'),
    ('B', 0): (1, 'R', 'C'),
    ('B', 1): (1, 'R', 'B'),
    ('C', 0): (1, 'R', 'D'),
    ('C', 1): (0, 'L', 'E'),
    ('D', 0): (1, 'L', 'A'),
    ('D', 1): (1, 'L', 'D'),
    ('E', 0): (1, 'R', 'HALT'),
    ('E', 1): (0, 'L', 'A'),
})


class FuzzyTM(nn.Module):
    def __init__(self, num_states, num_symbols=2, num_outputs=4):
        super().__init__()
        self.num_states = num_states
        self._T = nn.Parameter(torch.randn(num_states, num_symbols, num_states) * 0.1)
        self._O = nn.Parameter(torch.randn(num_states, num_symbols, num_outputs) * 0.1)
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


def train(model, input_seq, output_seq, epochs=500):
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for ep in range(epochs):
        out_probs = model(input_seq)
        targets = torch.tensor(output_seq)
        loss = F.cross_entropy(out_probs, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def evaluate(model, input_seq, output_seq):
    with torch.no_grad():
        preds = model(input_seq).argmax(dim=-1).tolist()
    correct = sum(p == t for p, t in zip(preds, output_seq))
    return correct / len(output_seq)


def main():
    print("=" * 60)
    print("Trace Length vs Accuracy for Busy Beaver")
    print("=" * 60)

    # Generate BB(4) trace
    trace, states, ones = BB4.run()
    full_input = [t[0] for t in trace]
    full_output = [encode_output(t[1], t[2]) for t in trace]

    print(f"\nBB(4): {len(trace)} steps, {ones} ones")
    print(f"Unique (state, symbol) pairs visited: ", end="")

    visited = set()
    for i, (read, write, move) in enumerate(trace):
        visited.add((states[i], read))
    print(f"{len(visited)} / 8 possible")

    # Test different training lengths
    print("\n" + "=" * 60)
    print("Training on first N steps, testing on full 107 steps")
    print("=" * 60)

    lengths = [10, 20, 30, 50, 70, 100, 107]

    print(f"\n{'Train len':<12} {'Train acc':<12} {'Full acc':<12} {'Coverage':<12}")
    print("-" * 50)

    results = []
    for n in lengths:
        train_input = full_input[:n]
        train_output = full_output[:n]

        # Count coverage
        coverage = set()
        for i in range(n):
            coverage.add((states[i], full_input[i]))

        # Train
        random.seed(42)
        torch.manual_seed(42)
        model = FuzzyTM(num_states=4)
        model = train(model, train_input, train_output, epochs=500)

        # Evaluate
        train_acc = evaluate(model, train_input, train_output)
        full_acc = evaluate(model, full_input, full_output)

        print(f"{n:<12} {train_acc:<12.1%} {full_acc:<12.1%} {len(coverage)}/8")
        results.append((n, train_acc, full_acc, len(coverage)))

    # Also try with more states (overparameterized)
    print("\n" + "=" * 60)
    print("Effect of state count (training on full 107 steps)")
    print("=" * 60)

    print(f"\n{'States':<12} {'Accuracy':<12}")
    print("-" * 25)

    for num_states in [3, 4, 5, 6, 8, 10]:
        random.seed(42)
        torch.manual_seed(42)
        model = FuzzyTM(num_states=num_states)
        model = train(model, full_input, full_output, epochs=500)
        acc = evaluate(model, full_input, full_output)
        marker = "← correct" if num_states == 4 else ""
        print(f"{num_states:<12} {acc:<12.1%} {marker}")

    # Try BB5_simple
    print("\n" + "=" * 60)
    print("Testing on 5-state TM")
    print("=" * 60)

    trace5, states5, ones5 = BB5_simple.run(max_steps=500)
    if len(trace5) > 0:
        input5 = [t[0] for t in trace5]
        output5 = [encode_output(t[1], t[2]) for t in trace5]

        print(f"\n5-state TM: {len(trace5)} steps, {ones5} ones")

        visited5 = set()
        for i, (read, write, move) in enumerate(trace5):
            visited5.add((states5[i], read))
        print(f"Coverage: {len(visited5)} / 10 possible transitions")

        if len(trace5) >= 10:
            lengths5 = [min(l, len(trace5)) for l in [10, 20, 50, 100, len(trace5)]]
            lengths5 = sorted(set(lengths5))

            print(f"\n{'Train len':<12} {'Train acc':<12} {'Full acc':<12}")
            print("-" * 40)

            for n in lengths5:
                random.seed(42)
                torch.manual_seed(42)
                model = FuzzyTM(num_states=5)
                model = train(model, input5[:n], output5[:n], epochs=500)
                train_acc = evaluate(model, input5[:n], output5[:n])
                full_acc = evaluate(model, input5, output5)
                print(f"{n:<12} {train_acc:<12.1%} {full_acc:<12.1%}")
    else:
        print("5-state TM halted immediately or has no valid transitions")


if __name__ == "__main__":
    main()
