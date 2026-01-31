"""
Counter as Regression: Predict depth directly, not as classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys

sys.stdout.reconfigure(line_buffering=True)


class CounterRegression(nn.Module):
    """
    Learn counter increments directly.
    Output = counter value (regression).
    """
    def __init__(self, num_inputs):
        super().__init__()
        # Learn delta for each input symbol
        # Initialize close to expected values
        self._delta = nn.Parameter(torch.tensor([1.0, -1.0]))

    def forward(self, input_seq):
        counter = 0.0
        outputs = []

        for inp in input_seq:
            delta = self._delta[inp]
            counter = counter + delta
            # ReLU to keep non-negative (with gradient for negative values via leaky)
            counter_out = F.leaky_relu(counter, negative_slope=0.01)
            outputs.append(counter_out)

        return torch.stack(outputs)


class SimpleFSA(nn.Module):
    """Baseline FSA for comparison."""
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
    print("Counter as Regression")
    print("=" * 60)

    in2i = {'(': 0, ')': 1}

    # Generate training data
    print("\nGenerating training data (max_depth=8)...")
    random.seed(42)
    train_data = []
    for _ in range(300):
        seq, depths = generate_depth_task(max_depth=8, max_length=20)
        train_data.append((seq, depths))

    print(f"Max depth in training: {max(max(d) for _, d in train_data)}")

    # ============ Train Counter Regression ============
    print("\n" + "=" * 60)
    print("Training Counter (regression, MSE loss)...")
    print("=" * 60)

    counter = CounterRegression(num_inputs=2)
    opt = torch.optim.Adam(counter.parameters(), lr=0.1)

    for ep in range(100):
        random.shuffle(train_data)
        total_loss = 0
        correct = total = 0

        for seq, depths in train_data:
            inp = [in2i[s] for s in seq]
            targets = torch.tensor(depths, dtype=torch.float)

            preds = counter(inp)
            loss = F.mse_loss(preds, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            correct += (preds.round() == targets).sum().item()
            total += len(depths)

        if ep % 25 == 0:
            delta_vals = counter._delta.detach().tolist()
            print(f"  Epoch {ep}: acc={correct/total:.1%}, loss={total_loss/len(train_data):.3f}, delta=[{delta_vals[0]:+.3f}, {delta_vals[1]:+.3f}]")

    print(f"\n  Final deltas: '(' → {counter._delta[0].item():+.4f}, ')' → {counter._delta[1].item():+.4f}")
    print(f"  Expected:     '(' → +1.0000, ')' → -1.0000")

    # ============ Train FSA for comparison ============
    print("\n" + "=" * 60)
    print("Training FSA (10 states, classification)...")
    print("=" * 60)

    num_outputs = 25
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

    # ============ Test generalization ============
    print("\n" + "=" * 60)
    print("Testing Generalization (trained on max_depth=8)")
    print("=" * 60)

    print(f"\n{'Max Depth':<12} {'FSA (10 st)':<15} {'Counter Reg':<15}")
    print("-" * 42)

    for test_depth in [5, 8, 10, 15, 20, 30, 50]:
        random.seed(789 + test_depth)
        test_data = []
        for _ in range(100):
            seq, depths = generate_depth_task(max_depth=test_depth, max_length=60)
            test_data.append((seq, depths))

        fsa_correct = fsa_total = 0
        ctr_correct = ctr_total = 0

        with torch.no_grad():
            for seq, depths in test_data:
                inp = [in2i[s] for s in seq]
                targets = torch.tensor(depths)

                fsa_preds = fsa(inp).argmax(-1)
                ctr_preds = counter(inp).round().long().clamp(0, num_outputs-1)

                fsa_correct += (fsa_preds == targets).sum().item()
                ctr_correct += (ctr_preds == targets).sum().item()
                fsa_total += len(depths)
                ctr_total += len(depths)

        fsa_acc = fsa_correct / fsa_total
        ctr_acc = ctr_correct / ctr_total

        if test_depth <= 8:
            note = "(in-dist)"
        elif test_depth <= 10:
            note = "(edge)"
        else:
            note = "(OOD!)"

        winner = "Counter!" if ctr_acc > fsa_acc + 0.02 else ("FSA" if fsa_acc > ctr_acc + 0.02 else "tie")
        print(f"{test_depth:<12} {fsa_acc:<15.1%} {ctr_acc:<15.1%} {note} → {winner}")

    # Example
    print("\n" + "=" * 60)
    print("Example (depth=20, way beyond training)")
    print("=" * 60)

    random.seed(54321)
    test_seq, test_depths = generate_depth_task(max_depth=20, max_length=40)
    inp = [in2i[s] for s in test_seq]

    with torch.no_grad():
        fsa_pred = fsa(inp).argmax(-1).tolist()
        ctr_pred = counter(inp).round().long().clamp(0, 24).tolist()

    print(f"\nInput:  {' '.join(test_seq)}")
    print(f"Truth:  {' '.join(map(str, test_depths))}")
    print(f"FSA:    {' '.join(map(str, fsa_pred))}")
    print(f"Counter:{' '.join(map(str, ctr_pred))}")

    fsa_err = sum(1 for p, t in zip(fsa_pred, test_depths) if p != t)
    ctr_err = sum(1 for p, t in zip(ctr_pred, test_depths) if p != t)
    print(f"\nErrors: FSA={fsa_err}, Counter={ctr_err}")


if __name__ == "__main__":
    main()
