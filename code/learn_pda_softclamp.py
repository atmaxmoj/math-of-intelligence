"""
Scalar Counter PDA with Soft Clamp - Fix gradient death.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys

sys.stdout.reconfigure(line_buffering=True)


def soft_clamp(x, min_val, max_val, sharpness=10.0):
    """
    Soft clamp that allows gradients to flow.
    Uses sigmoid to smoothly enforce bounds.
    """
    # Normalize to [0, 1] range
    x_norm = (x - min_val) / (max_val - min_val + 1e-6)
    # Apply sigmoid for soft bounds
    x_clamped = torch.sigmoid(sharpness * (x_norm - 0.5))
    # Scale back
    return min_val + x_clamped * (max_val - min_val)


class ScalarCounterPDA(nn.Module):
    """
    PDA with scalar counter and proper gradient flow.
    """
    def __init__(self, num_inputs, max_output=25):
        super().__init__()
        self.max_output = max_output

        # Initialize with hint: '(' should be +1, ')' should be -1
        self._delta = nn.Parameter(torch.tensor([0.5, -0.5]))  # Better init

        # Output scale
        self._out_scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, input_seq):
        counter = torch.tensor(0.0)  # Start at 0
        outputs = []

        for inp in input_seq:
            # Update counter (no clamp during forward, let gradient flow)
            delta = self._delta[inp]
            counter = counter + delta

            # Soft clamp for output computation only
            counter_soft = soft_clamp(counter, 0.0, float(self.max_output - 1))

            # Output: soft discretization
            positions = torch.arange(self.max_output, dtype=torch.float)
            distances = (positions - counter_soft).abs()
            logits = -distances * self._out_scale
            out_prob = F.softmax(logits, dim=0)
            outputs.append(out_prob)

        return torch.stack(outputs)


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
    print("Scalar Counter PDA with Soft Clamp")
    print("=" * 60)

    in2i = {'(': 0, ')': 1}

    # Generate training data
    print("\nGenerating training data (max_depth=8)...")
    random.seed(42)
    train_data = []
    for _ in range(300):
        seq, depths = generate_depth_task(max_depth=8, max_length=20)
        train_data.append((seq, depths))

    print(f"Actual max depth in training: {max(max(d) for _, d in train_data)}")

    num_outputs = 25

    # ============ Train FSA ============
    print("\n" + "=" * 60)
    print("Training FSA (10 states)...")
    print("=" * 60)

    fsa = SimpleFSA(10, 2, num_outputs)
    opt_fsa = torch.optim.Adam(fsa.parameters(), lr=0.1)

    for ep in range(100):
        random.shuffle(train_data)
        correct = total = 0
        for seq, depths in train_data:
            inp = [in2i[s] for s in seq]
            probs = fsa(inp)
            loss = F.cross_entropy(probs, torch.tensor(depths))
            opt_fsa.zero_grad()
            loss.backward()
            opt_fsa.step()
            correct += (probs.argmax(-1) == torch.tensor(depths)).sum().item()
            total += len(depths)
        if ep % 25 == 0:
            print(f"  Epoch {ep}: {correct/total:.1%}")

    # ============ Train Scalar Counter PDA ============
    print("\n" + "=" * 60)
    print("Training Scalar Counter PDA (soft clamp)...")
    print("=" * 60)

    pda = ScalarCounterPDA(num_inputs=2, max_output=num_outputs)
    opt_pda = torch.optim.Adam(pda.parameters(), lr=0.1)

    for ep in range(200):  # More epochs
        random.shuffle(train_data)
        correct = total = 0
        for seq, depths in train_data:
            inp = [in2i[s] for s in seq]
            probs = pda(inp)
            loss = F.cross_entropy(probs, torch.tensor(depths))
            opt_pda.zero_grad()
            loss.backward()
            opt_pda.step()
            correct += (probs.argmax(-1) == torch.tensor(depths)).sum().item()
            total += len(depths)
        if ep % 50 == 0:
            delta_vals = pda._delta.detach().tolist()
            print(f"  Epoch {ep}: {correct/total:.1%}  delta=[{delta_vals[0]:+.3f}, {delta_vals[1]:+.3f}]")

    # Show learned deltas
    print("\n  Learned deltas:")
    print(f"    '(' → {pda._delta[0].item():+.4f}")
    print(f"    ')' → {pda._delta[1].item():+.4f}")
    print(f"  (Expected: '(' → +1.0, ')' → -1.0)")

    # ============ Test generalization ============
    print("\n" + "=" * 60)
    print("Testing Generalization (trained on max_depth=8)")
    print("=" * 60)

    print(f"\n{'Max Depth':<12} {'FSA (10 st)':<15} {'Scalar PDA':<15}")
    print("-" * 42)

    for test_depth in [5, 8, 10, 15, 20, 30]:
        random.seed(789 + test_depth)
        test_data = []
        for _ in range(100):
            seq, depths = generate_depth_task(max_depth=test_depth, max_length=50)
            test_data.append((seq, depths))

        fsa_correct = fsa_total = 0
        pda_correct = pda_total = 0

        with torch.no_grad():
            for seq, depths in test_data:
                inp = [in2i[s] for s in seq]

                fsa_preds = fsa(inp).argmax(-1).tolist()
                pda_preds = pda(inp).argmax(-1).tolist()

                fsa_correct += sum(p == t for p, t in zip(fsa_preds, depths))
                pda_correct += sum(p == t for p, t in zip(pda_preds, depths))
                fsa_total += len(depths)
                pda_total += len(depths)

        fsa_acc = fsa_correct / fsa_total
        pda_acc = pda_correct / pda_total

        if test_depth <= 8:
            note = "(in-dist)"
        elif test_depth <= 10:
            note = "(edge)"
        else:
            note = "(OOD!)"

        winner = "PDA!" if pda_acc > fsa_acc + 0.02 else ("FSA" if fsa_acc > pda_acc + 0.02 else "tie")
        print(f"{test_depth:<12} {fsa_acc:<15.1%} {pda_acc:<15.1%} {note} → {winner}")

    # Example
    print("\n" + "=" * 60)
    print("Example (depth=15, beyond training)")
    print("=" * 60)

    random.seed(12345)
    test_seq, test_depths = generate_depth_task(max_depth=15, max_length=30)
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
