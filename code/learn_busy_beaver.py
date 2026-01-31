"""
Learn Busy Beaver from execution traces!

A Turing Machine step: (state, read_symbol) → (write_symbol, move_direction, next_state)

We model this as a Mealy Machine:
- Input: read_symbol (0 or 1)
- Output: (write_symbol, move_direction) encoded as integer
- State transitions: learned implicitly

BB(2): 2 states, runs 6 steps, writes 4 ones
BB(3): 3 states, runs 21 steps, writes 6 ones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys

sys.stdout.reconfigure(line_buffering=True)


# Output encoding: (write, move) → integer
# 0 = (0, L), 1 = (0, R), 2 = (1, L), 3 = (1, R)
def encode_output(write, move):
    return write * 2 + (0 if move == 'L' else 1)

def decode_output(code):
    write = code // 2
    move = 'L' if code % 2 == 0 else 'R'
    return write, move


class TuringMachine:
    """Simulate a Turing Machine and generate execution trace."""

    def __init__(self, transitions, initial_state='A', halt_state='HALT'):
        """
        transitions: dict of (state, symbol) → (write, move, next_state)
        """
        self.transitions = transitions
        self.initial_state = initial_state
        self.halt_state = halt_state

    def run(self, max_steps=1000):
        """Run the TM and return execution trace."""
        tape = {}  # sparse tape, default 0
        head = 0
        state = self.initial_state

        trace = []  # list of (read_symbol, write_symbol, move)
        states = [state]  # state before each step

        for _ in range(max_steps):
            read_sym = tape.get(head, 0)

            if (state, read_sym) not in self.transitions:
                break  # undefined transition = halt

            write_sym, move, next_state = self.transitions[(state, read_sym)]
            trace.append((read_sym, write_sym, move))

            tape[head] = write_sym
            head += 1 if move == 'R' else -1
            state = next_state
            states.append(state)

            if state == self.halt_state:
                break

        # Count ones on tape
        ones = sum(1 for v in tape.values() if v == 1)
        return trace, states, ones


# Known Busy Beavers
BB2 = TuringMachine({
    ('A', 0): (1, 'R', 'B'),
    ('A', 1): (1, 'L', 'B'),
    ('B', 0): (1, 'L', 'A'),
    ('B', 1): (1, 'R', 'HALT'),
})

BB3 = TuringMachine({
    ('A', 0): (1, 'R', 'B'),
    ('A', 1): (1, 'L', 'C'),
    ('B', 0): (1, 'L', 'A'),
    ('B', 1): (1, 'R', 'B'),
    ('C', 0): (1, 'L', 'B'),
    ('C', 1): (1, 'R', 'HALT'),
})

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


class FuzzyTuringMachine(nn.Module):
    """Fuzzy/differentiable Turing Machine."""

    def __init__(self, num_states, num_symbols=2, num_outputs=4):
        super().__init__()
        self.num_states = num_states
        self.num_symbols = num_symbols
        self.num_outputs = num_outputs

        # Transition matrix: (state, symbol) → next_state distribution
        self._T = nn.Parameter(torch.randn(num_states, num_symbols, num_states) * 0.1)
        # Output matrix: (state, symbol) → output distribution
        self._O = nn.Parameter(torch.randn(num_states, num_symbols, num_outputs) * 0.1)
        # Initial state
        self._init = nn.Parameter(torch.zeros(num_states))

    def forward(self, input_seq):
        """
        input_seq: list of symbols (0 or 1)
        returns: output probabilities for each step
        """
        T = F.softmax(self._T, dim=-1)
        O = F.softmax(self._O, dim=-1)
        state = F.softmax(self._init, dim=0)

        outputs = []
        for sym in input_seq:
            # Output distribution for current state and input
            out_prob = (O[:, sym, :] * state.unsqueeze(1)).sum(dim=0)
            outputs.append(out_prob)
            # Transition to next state
            state = (T[:, sym, :] * state.unsqueeze(1)).sum(dim=0)

        return torch.stack(outputs)

    def get_initial_state(self):
        return F.softmax(self._init, dim=0)

    def get_transition_table(self):
        """Extract the learned transition table."""
        T = F.softmax(self._T, dim=-1)
        O = F.softmax(self._O, dim=-1)
        init = F.softmax(self._init, dim=0)

        table = []
        init_state = init.argmax().item()

        for s in range(self.num_states):
            for sym in range(self.num_symbols):
                next_s = T[s, sym].argmax().item()
                out = O[s, sym].argmax().item()
                write, move = decode_output(out)
                table.append((s, sym, write, move, next_s))

        return table, init_state


def train(model, traces, epochs=500, lr=0.1):
    """Train on multiple traces."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        random.shuffle(traces)
        total_loss = 0
        correct = 0
        total = 0

        for input_seq, output_seq in traces:
            out_probs = model(input_seq)
            targets = torch.tensor(output_seq)
            loss = F.cross_entropy(out_probs, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            preds = out_probs.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += len(targets)

        if ep % 100 == 0:
            acc = correct / total if total > 0 else 0
            print(f"  Epoch {ep}: loss={total_loss/len(traces):.4f}, acc={acc:.1%}")


def main():
    print("=" * 60)
    print("Learning Busy Beaver from Execution Traces")
    print("=" * 60)

    for name, tm, true_states in [("BB(2)", BB2, 2), ("BB(3)", BB3, 3), ("BB(4)", BB4, 4)]:
        print(f"\n{'=' * 60}")
        print(f"{name}")
        print("=" * 60)

        # Generate trace
        trace, states, ones = tm.run()
        print(f"Execution: {len(trace)} steps, writes {ones} ones")
        print(f"State sequence: {' → '.join(states)}")

        # Convert to input/output sequences
        input_seq = [t[0] for t in trace]  # read symbols
        output_seq = [encode_output(t[1], t[2]) for t in trace]  # (write, move) encoded

        print(f"Input (read symbols):  {input_seq}")
        print(f"Output (write, move):  {[decode_output(o) for o in output_seq]}")

        # Create training data - just this one trace
        # But we can also generate variations by running from different initial states
        train_data = [(input_seq, output_seq)]

        # Train fuzzy TM
        print(f"\nTraining Fuzzy TM with {true_states} states...")
        model = FuzzyTuringMachine(num_states=true_states, num_symbols=2, num_outputs=4)
        train(model, train_data, epochs=500, lr=0.1)

        # Extract learned transitions
        print("\nLearned transition table:")
        table, init_state = model.get_transition_table()
        print(f"Initial state: {init_state}")
        print(f"{'State':<8} {'Read':<6} {'Write':<7} {'Move':<6} {'Next':<6}")
        print("-" * 35)
        for s, sym, write, move, next_s in table:
            print(f"  {s:<6} {sym:<6} {write:<7} {move:<6} {next_s:<6}")

        # Compare with ground truth
        print("\nGround truth transitions:")
        for (state, sym), (write, move, next_state) in tm.transitions.items():
            print(f"  {state:<6} {sym:<6} {write:<7} {move:<6} {next_state}")

        # Test: run the learned TM
        print("\nTesting learned model on trace...")
        with torch.no_grad():
            preds = model(input_seq).argmax(dim=-1).tolist()
            pred_outputs = [decode_output(p) for p in preds]
            true_outputs = [decode_output(o) for o in output_seq]

            matches = sum(p == t for p, t in zip(pred_outputs, true_outputs))
            print(f"Accuracy: {matches}/{len(trace)} = {matches/len(trace):.1%}")

            if pred_outputs != true_outputs:
                print("Differences:")
                for i, (p, t) in enumerate(zip(pred_outputs, true_outputs)):
                    if p != t:
                        print(f"  Step {i}: predicted {p}, actual {t}")


if __name__ == "__main__":
    main()
