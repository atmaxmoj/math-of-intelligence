"""
Fixed PDA: Expose stack HEIGHT to the model, not just content.

Key insight: Depth counting needs stack HEIGHT, not stack CONTENT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys

sys.stdout.reconfigure(line_buffering=True)


class DifferentiableCounter(nn.Module):
    """
    Learnable counter: simpler than full stack, perfect for depth counting.

    The counter value is a soft distribution over possible counts.
    """
    def __init__(self, max_count=30):
        super().__init__()
        self.max_count = max_count

    def forward(self, inc_probs, dec_probs):
        """
        Process sequence of increment/decrement probabilities.

        inc_probs: [seq_len] - probability of incrementing at each step
        dec_probs: [seq_len] - probability of decrementing at each step

        Returns: [seq_len, max_count] - distribution over count values
        """
        seq_len = len(inc_probs)

        # Count distribution: probability of being at each count value
        count_dist = torch.zeros(self.max_count)
        count_dist[0] = 1.0  # Start at 0

        outputs = []

        for t in range(seq_len):
            outputs.append(count_dist.clone())

            inc = inc_probs[t]
            dec = dec_probs[t]
            noop = 1 - inc - dec
            noop = torch.clamp(noop, min=0)

            # New distribution after operations
            new_dist = torch.zeros_like(count_dist)

            # Increment: shift right
            new_dist[1:] += inc * count_dist[:-1]
            new_dist[0] += inc * count_dist[-1]  # Overflow stays at max (or could clamp)

            # Decrement: shift left (but clamp at 0)
            new_dist[:-1] += dec * count_dist[1:]
            new_dist[0] += dec * count_dist[0]  # Can't go below 0

            # No-op: stay same
            new_dist += noop * count_dist

            count_dist = new_dist

        return torch.stack(outputs)


class FuzzyPDAWithCounter(nn.Module):
    """
    PDA that learns to use a counter for depth tracking.
    """
    def __init__(self, num_states, num_inputs, num_outputs, max_count=30):
        super().__init__()
        self.num_states = num_states
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.max_count = max_count

        # State transition
        self._T = nn.Parameter(torch.randn(num_states, num_inputs, num_states) * 0.1)

        # Counter actions: (state, input) → (inc_logit, dec_logit)
        self._inc = nn.Parameter(torch.randn(num_states, num_inputs) * 0.1)
        self._dec = nn.Parameter(torch.randn(num_states, num_inputs) * 0.1)

        # Output: from counter distribution to output
        self._out = nn.Linear(max_count, num_outputs)

        # Initial state
        self._init = nn.Parameter(torch.zeros(num_states))

        self.counter = DifferentiableCounter(max_count)

    def forward(self, input_seq):
        seq_len = len(input_seq)

        # Compute state sequence
        T = F.softmax(self._T, dim=-1)
        state = F.softmax(self._init, dim=0)

        inc_probs = []
        dec_probs = []

        for t in range(seq_len):
            inp = input_seq[t]

            # Compute inc/dec probabilities weighted by state
            inc_logit = (self._inc[:, inp] * state).sum()
            dec_logit = (self._dec[:, inp] * state).sum()

            # Softmax over (inc, dec, noop)
            action_logits = torch.stack([inc_logit, dec_logit, torch.tensor(0.0)])
            action_probs = F.softmax(action_logits, dim=0)

            inc_probs.append(action_probs[0])
            dec_probs.append(action_probs[1])

            # State transition
            state = (T[:, inp, :] * state.unsqueeze(1)).sum(0)

        inc_probs = torch.stack(inc_probs)
        dec_probs = torch.stack(dec_probs)

        # Run counter
        count_dists = self.counter(inc_probs, dec_probs)  # [seq_len, max_count]

        # Output from count distribution
        outputs = self._out(count_dists)  # [seq_len, num_outputs]
        outputs = F.softmax(outputs, dim=-1)

        return outputs


class SimpleFSA(nn.Module):
    """Baseline FSA."""
    def __init__(self, num_states, num_inputs, num_outputs):
        super().__init__()
        self._T = nn.Parameter(torch.randn(num_states, num_inputs, num_states) * 0.1)
        self._O = nn.Parameter(torch.randn(num_states, num_inputs, num_outputs) * 0.1)
        self._init = nn.Parameter(torch.zeros(num_states))

    def forward(self, input_seq):
        T = F.softmax(self._T, dim=-1)
        O = F.softmax(self._O, dim=-1)
        state = F.softmax(self._init, dim=0)
        outputs = []
        for inp in input_seq:
            out_prob = (O[:, inp, :] * state.unsqueeze(1)).sum(0)
            outputs.append(out_prob)
            state = (T[:, inp, :] * state.unsqueeze(1)).sum(0)
        return torch.stack(outputs)


def generate_depth_task(max_depth=10, max_length=30):
    """Generate balanced parentheses with depth labels."""
    seq = []
    depths = []
    depth = 0

    while len(seq) < max_length:
        if depth == 0:
            seq.append('(')
            depth += 1
        elif depth >= max_depth:
            seq.append(')')
            depth -= 1
        elif random.random() < 0.5:
            seq.append('(')
            depth += 1
        else:
            seq.append(')')
            depth -= 1
        depths.append(depth)

        if depth == 0 and len(seq) >= 4 and random.random() < 0.3:
            break

    return seq, depths


def main():
    print("=" * 60)
    print("Fixed PDA: Counter-based Depth Learning")
    print("=" * 60)

    in2i = {'(': 0, ')': 1}

    # Generate training data with LIMITED depth
    print("\nGenerating training data (max_depth=8)...")
    random.seed(42)
    train_data = []
    for _ in range(300):
        seq, depths = generate_depth_task(max_depth=8, max_length=20)
        train_data.append((seq, depths))

    max_depth_train = max(max(d) for _, d in train_data)
    print(f"Actual max depth in training: {max_depth_train}")

    num_outputs = 25  # Support depths 0-24

    # ============ Train FSA ============
    print("\n" + "=" * 60)
    print("Training FSA (10 states - less than max depth)...")
    print("=" * 60)

    fsa = SimpleFSA(10, 2, num_outputs)  # Only 10 states!
    opt_fsa = torch.optim.Adam(fsa.parameters(), lr=0.1)

    for ep in range(100):
        random.shuffle(train_data)
        correct = total = 0
        for seq, depths in train_data:
            inp = [in2i[s] for s in seq]
            out = depths
            probs = fsa(inp)
            loss = F.cross_entropy(probs, torch.tensor(out))
            opt_fsa.zero_grad()
            loss.backward()
            opt_fsa.step()
            correct += (probs.argmax(-1) == torch.tensor(out)).sum().item()
            total += len(out)
        if ep % 25 == 0:
            print(f"  Epoch {ep}: {correct/total:.1%}")

    # ============ Train PDA with Counter ============
    print("\n" + "=" * 60)
    print("Training PDA with Differentiable Counter...")
    print("=" * 60)

    pda = FuzzyPDAWithCounter(num_states=3, num_inputs=2, num_outputs=num_outputs, max_count=30)
    opt_pda = torch.optim.Adam(pda.parameters(), lr=0.1)

    for ep in range(100):
        random.shuffle(train_data)
        correct = total = 0
        for seq, depths in train_data:
            inp = [in2i[s] for s in seq]
            out = depths
            probs = pda(inp)
            loss = F.cross_entropy(probs, torch.tensor(out))
            opt_pda.zero_grad()
            loss.backward()
            opt_pda.step()
            correct += (probs.argmax(-1) == torch.tensor(out)).sum().item()
            total += len(out)
        if ep % 25 == 0:
            print(f"  Epoch {ep}: {correct/total:.1%}")

    # ============ Test generalization ============
    print("\n" + "=" * 60)
    print("Testing Generalization (trained on max_depth=8)")
    print("=" * 60)

    print(f"\n{'Max Depth':<12} {'FSA (10 st)':<15} {'PDA+Counter':<15}")
    print("-" * 42)

    for test_depth in [5, 8, 10, 15, 20]:
        random.seed(789 + test_depth)
        test_data = []
        for _ in range(100):
            seq, depths = generate_depth_task(max_depth=test_depth, max_length=40)
            test_data.append((seq, depths))

        fsa_correct = fsa_total = 0
        pda_correct = pda_total = 0

        with torch.no_grad():
            for seq, depths in test_data:
                inp = [in2i[s] for s in seq]
                out = depths

                fsa_preds = fsa(inp).argmax(-1).tolist()
                pda_preds = pda(inp).argmax(-1).tolist()

                fsa_correct += sum(p == t for p, t in zip(fsa_preds, out))
                pda_correct += sum(p == t for p, t in zip(pda_preds, out))
                fsa_total += len(out)
                pda_total += len(out)

        fsa_acc = fsa_correct / fsa_total
        pda_acc = pda_correct / pda_total

        if test_depth <= 8:
            note = "(in-dist)"
        else:
            note = "(OOD!)" if test_depth > 10 else "(edge)"

        winner = "PDA!" if pda_acc > fsa_acc + 0.02 else ("FSA" if fsa_acc > pda_acc + 0.02 else "tie")
        print(f"{test_depth:<12} {fsa_acc:<15.1%} {pda_acc:<15.1%} {note} → {winner}")

    # Show learned counter behavior
    print("\n" + "=" * 60)
    print("Learned Counter Actions")
    print("=" * 60)

    with torch.no_grad():
        for inp_name, inp_idx in [("'('", 0), ("')'", 1)]:
            inc = torch.sigmoid(pda._inc[0, inp_idx]).item()
            dec = torch.sigmoid(pda._dec[0, inp_idx]).item()
            print(f"  {inp_name}: inc={inc:.2f}, dec={dec:.2f}")

    # Example
    print("\n" + "=" * 60)
    print("Example (depth > training)")
    print("=" * 60)

    random.seed(12345)
    test_seq, test_depths = generate_depth_task(max_depth=15, max_length=25)
    inp = [in2i[s] for s in test_seq]

    with torch.no_grad():
        fsa_pred = fsa(inp).argmax(-1).tolist()
        pda_pred = pda(inp).argmax(-1).tolist()

    print(f"\nInput:  {' '.join(test_seq)}")
    print(f"Truth:  {' '.join(map(str, test_depths))}")
    print(f"FSA:    {' '.join(map(str, fsa_pred))}")
    print(f"PDA:    {' '.join(map(str, pda_pred))}")

    fsa_err = sum(1 for p, t in zip(fsa_pred, test_depths) if p != t)
    pda_err = sum(1 for p, t in zip(pda_pred, test_depths) if p != t)
    print(f"\nErrors: FSA={fsa_err}, PDA={pda_err}")


if __name__ == "__main__":
    main()
