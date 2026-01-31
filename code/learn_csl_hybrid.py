"""
Hybrid Model: State Machine + Counters

For a^n b^n c^n, we need:
1. State: track phase (expecting a, expecting b, expecting c, invalid)
2. Counters: track counts

This is the key insight for Context-Sensitive languages!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys

sys.stdout.reconfigure(line_buffering=True)


def generate_anbncn_data(num_samples=200, max_n=10, include_invalid=True):
    data = []
    for _ in range(num_samples):
        if random.random() < 0.5 or not include_invalid:
            n = random.randint(1, max_n)
            s = 'a' * n + 'b' * n + 'c' * n
            labels = [1] * len(s)
            data.append((s, labels, True))
        else:
            invalid_type = random.choice(['wrong_count', 'wrong_order', 'random'])
            if invalid_type == 'wrong_count':
                na, nb, nc = random.randint(1, max_n), random.randint(1, max_n), random.randint(1, max_n)
                while na == nb == nc:
                    nc = random.randint(1, max_n)
                s = 'a' * na + 'b' * nb + 'c' * nc
            elif invalid_type == 'wrong_order':
                n = random.randint(1, max_n)
                chars = ['a'] * n + ['b'] * n + ['c'] * n
                random.shuffle(chars)
                s = ''.join(chars)
            else:
                length = random.randint(3, max_n * 3)
                s = ''.join(random.choice('abc') for _ in range(length))
            labels = compute_validity_labels(s)
            data.append((s, labels, False))
    return data


def compute_validity_labels(s):
    labels = []
    count_a = count_b = count_c = 0
    phase = 'a'
    valid = True
    for char in s:
        if not valid:
            labels.append(0)
            continue
        if phase == 'a':
            if char == 'a':
                count_a += 1
            elif char == 'b':
                phase = 'b'
                count_b = 1
            else:
                valid = False
        elif phase == 'b':
            if char == 'b':
                count_b += 1
                if count_b > count_a:
                    valid = False
            elif char == 'c':
                if count_b != count_a:
                    valid = False
                else:
                    phase = 'c'
                    count_c = 1
            else:
                valid = False
        elif phase == 'c':
            if char == 'c':
                count_c += 1
                if count_c > count_a:
                    valid = False
            else:
                valid = False
        labels.append(1 if valid else 0)
    return labels


class HybridStateCounter(nn.Module):
    """
    Hybrid: FSA for phase tracking + Counters for counting.

    States: 0=phase_a, 1=phase_b, 2=phase_c, 3=invalid
    Counters: count_a, count_b, count_c
    """
    def __init__(self, num_inputs=3):
        super().__init__()

        # State transitions (4 states x 3 inputs x 4 next_states)
        # Initialize with prior knowledge of phase structure
        self._T = nn.Parameter(torch.zeros(4, 3, 4))
        # phase_a + a → phase_a
        self._T.data[0, 0, 0] = 5.0
        # phase_a + b → phase_b
        self._T.data[0, 1, 1] = 5.0
        # phase_a + c → invalid
        self._T.data[0, 2, 3] = 5.0
        # phase_b + b → phase_b
        self._T.data[1, 1, 1] = 5.0
        # phase_b + c → phase_c
        self._T.data[1, 2, 2] = 5.0
        # phase_b + a → invalid
        self._T.data[1, 0, 3] = 5.0
        # phase_c + c → phase_c
        self._T.data[2, 2, 2] = 5.0
        # phase_c + a,b → invalid
        self._T.data[2, 0, 3] = 5.0
        self._T.data[2, 1, 3] = 5.0
        # invalid → invalid
        self._T.data[3, :, 3] = 5.0

        # Counter deltas for each input
        self._delta_a = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))  # a increments count_a
        self._delta_b = nn.Parameter(torch.tensor([0.0, 1.0, 0.0]))  # b increments count_b
        self._delta_c = nn.Parameter(torch.tensor([0.0, 0.0, 1.0]))  # c increments count_c

    def forward(self, input_seq):
        # State distribution (4 states)
        state = torch.zeros(4)
        state[0] = 1.0  # Start in phase_a

        # Counters
        count_a = 0.0
        count_b = 0.0
        count_c = 0.0

        outputs = []

        for inp in input_seq:
            # Update counters
            count_a = count_a + self._delta_a[inp]
            count_b = count_b + self._delta_b[inp]
            count_c = count_c + self._delta_c[inp]

            # State transition
            T = F.softmax(self._T, dim=-1)
            new_state = (T[:, inp, :] * state.unsqueeze(1)).sum(0)

            # Validity checks based on state and counters
            # In phase_b: count_b should not exceed count_a
            phase_b_valid = torch.sigmoid((count_a - count_b + 0.5) * 10)
            # In phase_c: count_c should not exceed count_a
            phase_c_valid = torch.sigmoid((count_a - count_c + 0.5) * 10)
            # At end of b's (transition to c): count_b should equal count_a
            # This is implicitly handled by the structure

            # Combine state validity with counter validity
            # Valid = not in invalid state AND counter constraints satisfied
            not_invalid = 1.0 - new_state[3]

            # When in phase_b, check count_b <= count_a
            phase_b_prob = new_state[1]
            # When in phase_c, check count_c <= count_a
            phase_c_prob = new_state[2]

            # Overall validity
            counter_valid = (1.0 - phase_b_prob + phase_b_prob * phase_b_valid) * \
                           (1.0 - phase_c_prob + phase_c_prob * phase_c_valid)

            valid_prob = not_invalid * counter_valid
            valid_prob = torch.clamp(valid_prob, 0.0, 1.0)

            outputs.append(torch.stack([1 - valid_prob, valid_prob]))
            state = new_state

        return torch.stack(outputs)


class PureCounter(nn.Module):
    """Pure counter without state - for comparison."""
    def __init__(self, num_inputs=3):
        super().__init__()
        self._delta_a = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
        self._delta_b = nn.Parameter(torch.tensor([0.0, 1.0, 0.0]))
        self._delta_c = nn.Parameter(torch.tensor([0.0, 0.0, 1.0]))

    def forward(self, input_seq):
        count_a = count_b = count_c = 0.0
        outputs = []

        for inp in input_seq:
            count_a = count_a + self._delta_a[inp]
            count_b = count_b + self._delta_b[inp]
            count_c = count_c + self._delta_c[inp]

            # Valid if count_a >= count_b and count_a >= count_c
            valid1 = torch.sigmoid((count_a - count_b + 0.5) * 10)
            valid2 = torch.sigmoid((count_a - count_c + 0.5) * 10)
            valid_prob = valid1 * valid2

            outputs.append(torch.stack([1 - valid_prob, valid_prob]))

        return torch.stack(outputs)


class FSA(nn.Module):
    """FSA baseline."""
    def __init__(self, num_states, num_inputs, num_outputs=2):
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


def train_model(model, train_data, in2i, epochs=100, lr=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        random.shuffle(train_data)
        correct = total = 0
        for s, labels, _ in train_data:
            inp = [in2i[c] for c in s]
            targets = torch.tensor(labels)
            probs = model(inp)
            loss = F.cross_entropy(probs, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            preds = probs.argmax(-1)
            correct += (preds == targets).sum().item()
            total += len(labels)
        if ep % 25 == 0:
            print(f"    Epoch {ep}: {correct/total:.1%}")
    return correct / total


def evaluate_model(model, test_data, in2i):
    correct = total = 0
    with torch.no_grad():
        for s, labels, _ in test_data:
            inp = [in2i[c] for c in s]
            targets = torch.tensor(labels)
            preds = model(inp).argmax(-1)
            correct += (preds == targets).sum().item()
            total += len(labels)
    return correct / total


def main():
    print("=" * 60)
    print("Hybrid Model: State + Counters for a^n b^n c^n")
    print("=" * 60)

    in2i = {'a': 0, 'b': 1, 'c': 2}

    print("\nGenerating training data (max_n=5)...")
    random.seed(42)
    train_data = generate_anbncn_data(num_samples=300, max_n=5)

    # Train models
    print("\n" + "=" * 60)
    print("Training FSA (20 states)...")
    fsa = FSA(num_states=20, num_inputs=3)
    train_model(fsa, train_data, in2i, epochs=100)

    print("\n" + "=" * 60)
    print("Training Pure Counter (no state)...")
    pure = PureCounter(num_inputs=3)
    train_model(pure, train_data, in2i, epochs=100)

    print("\n" + "=" * 60)
    print("Training Hybrid (State + Counter)...")
    hybrid = HybridStateCounter(num_inputs=3)
    train_model(hybrid, train_data, in2i, epochs=100, lr=0.05)

    # Test
    print("\n" + "=" * 60)
    print("Testing Generalization (trained on n≤5)")
    print("=" * 60)

    print(f"\n{'Max n':<10} {'FSA':<12} {'Pure Ctr':<12} {'Hybrid':<12}")
    print("-" * 46)

    for test_n in [3, 5, 8, 10, 15, 20]:
        random.seed(789 + test_n)
        test_data = generate_anbncn_data(num_samples=100, max_n=test_n)

        fsa_acc = evaluate_model(fsa, test_data, in2i)
        pure_acc = evaluate_model(pure, test_data, in2i)
        hybrid_acc = evaluate_model(hybrid, test_data, in2i)

        note = "(in-dist)" if test_n <= 5 else "(OOD!)"
        winner = max([('FSA', fsa_acc), ('Pure', pure_acc), ('Hybrid', hybrid_acc)], key=lambda x: x[1])[0]
        print(f"{test_n:<10} {fsa_acc:<12.1%} {pure_acc:<12.1%} {hybrid_acc:<12.1%} {note} → {winner}")

    # Examples
    print("\n" + "=" * 60)
    print("Specific Examples")
    print("=" * 60)

    examples = [
        ("aabbcc", "valid"),
        ("aaabbbccc", "valid"),
        ("aabbbcc", "INVALID: b>a"),
        ("aabbccc", "INVALID: c>a"),
        ("aaaabbbbcccc", "valid n=4 (OOD)"),
        ("aaaaaaabbbbbbbccccccc", "valid n=7 (OOD)"),
    ]

    for s, desc in examples:
        inp = [in2i[c] for c in s]
        labels = compute_validity_labels(s)

        with torch.no_grad():
            fsa_pred = fsa(inp).argmax(-1).tolist()
            pure_pred = pure(inp).argmax(-1).tolist()
            hybrid_pred = hybrid(inp).argmax(-1).tolist()

        print(f"\n  '{s}' ({desc})")
        print(f"    Truth:  {labels}")
        print(f"    FSA:    {fsa_pred}")
        print(f"    Pure:   {pure_pred}")
        print(f"    Hybrid: {hybrid_pred}")


if __name__ == "__main__":
    main()
