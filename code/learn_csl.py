"""
Chomsky Level 1: Context-Sensitive Languages

Task: a^n b^n c^n recognition
- Valid: aabbcc, aaabbbccc
- Invalid: aabbc, aabbccc

This requires tracking MULTIPLE counts simultaneously.
- Single counter (CF): can do a^n b^n
- Multiple counters (CS): can do a^n b^n c^n
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys

sys.stdout.reconfigure(line_buffering=True)


def generate_anbncn(n):
    """Generate valid a^n b^n c^n string."""
    return 'a' * n + 'b' * n + 'c' * n


def generate_anbncn_data(num_samples=200, max_n=10, include_invalid=True):
    """Generate training data for a^n b^n c^n recognition."""
    data = []

    for _ in range(num_samples):
        if random.random() < 0.5 or not include_invalid:
            # Valid string
            n = random.randint(1, max_n)
            s = generate_anbncn(n)
            # Label each position: 1 if valid so far, else 0
            # Actually, let's output the "state" at each position
            # For a^n b^n c^n: output (count_a, count_b, count_c, valid)
            # Simpler: just output 1 (valid) or 0 (invalid) at each position
            labels = [1] * len(s)  # Valid string, all positions valid
            data.append((s, labels, True))
        else:
            # Invalid string - various types
            invalid_type = random.choice(['wrong_count', 'wrong_order', 'random'])

            if invalid_type == 'wrong_count':
                # Different counts
                na = random.randint(1, max_n)
                nb = random.randint(1, max_n)
                nc = random.randint(1, max_n)
                while na == nb == nc:  # Make sure it's actually invalid
                    nc = random.randint(1, max_n)
                s = 'a' * na + 'b' * nb + 'c' * nc

            elif invalid_type == 'wrong_order':
                # Wrong order
                n = random.randint(1, max_n)
                chars = ['a'] * n + ['b'] * n + ['c'] * n
                random.shuffle(chars)
                s = ''.join(chars)

            else:
                # Random string
                length = random.randint(3, max_n * 3)
                s = ''.join(random.choice('abc') for _ in range(length))

            # For invalid strings, mark where it becomes invalid
            labels = compute_validity_labels(s)
            data.append((s, labels, False))

    return data


def compute_validity_labels(s):
    """
    Compute validity at each position.
    Valid means: could still be a prefix of some a^n b^n c^n.
    """
    labels = []
    count_a = count_b = count_c = 0
    phase = 'a'  # Expecting 'a', then 'b', then 'c'
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


# ============ Models ============

class SingleCounter(nn.Module):
    """Single counter - can do a^n b^n but NOT a^n b^n c^n."""
    def __init__(self, num_inputs):
        super().__init__()
        self._delta = nn.Parameter(torch.tensor([1.0, 0.0, -1.0]))  # a, b, c

    def forward(self, input_seq):
        counter = 0.0
        outputs = []
        for inp in input_seq:
            counter = counter + self._delta[inp]
            # Valid if counter >= 0
            valid_prob = torch.sigmoid(counter * 5)
            outputs.append(torch.stack([1 - valid_prob, valid_prob]))
        return torch.stack(outputs)


class DualCounter(nn.Module):
    """Two counters - can do a^n b^n c^n."""
    def __init__(self, num_inputs):
        super().__init__()
        # Counter 1: tracks a vs b
        self._delta1 = nn.Parameter(torch.tensor([1.0, -1.0, 0.0]))  # a:+1, b:-1, c:0
        # Counter 2: tracks a vs c (or b vs c)
        self._delta2 = nn.Parameter(torch.tensor([1.0, 0.0, -1.0]))  # a:+1, b:0, c:-1

    def forward(self, input_seq):
        counter1 = 0.0
        counter2 = 0.0
        outputs = []

        for inp in input_seq:
            counter1 = counter1 + self._delta1[inp]
            counter2 = counter2 + self._delta2[inp]

            # Valid if both counters >= 0
            valid1 = torch.sigmoid(counter1 * 5)
            valid2 = torch.sigmoid(counter2 * 5)
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
    string_correct = 0

    with torch.no_grad():
        for s, labels, is_valid in test_data:
            inp = [in2i[c] for c in s]
            targets = torch.tensor(labels)

            preds = model(inp).argmax(-1)
            correct += (preds == targets).sum().item()
            total += len(labels)

            if (preds == targets).all():
                string_correct += 1

    return correct / total, string_correct / len(test_data)


def main():
    print("=" * 60)
    print("Chomsky Level 1: Context-Sensitive (a^n b^n c^n)")
    print("=" * 60)

    in2i = {'a': 0, 'b': 1, 'c': 2}

    # Generate training data (small n)
    print("\nGenerating training data (max_n=5)...")
    random.seed(42)
    train_data = generate_anbncn_data(num_samples=300, max_n=5)
    valid_count = sum(1 for _, _, v in train_data if v)
    print(f"  {len(train_data)} samples, {valid_count} valid, {len(train_data)-valid_count} invalid")

    # ============ Train models ============

    print("\n" + "=" * 60)
    print("Training FSA (20 states)...")
    print("=" * 60)
    fsa = FSA(num_states=20, num_inputs=3)
    train_model(fsa, train_data, in2i, epochs=100)

    print("\n" + "=" * 60)
    print("Training Single Counter (CF level)...")
    print("=" * 60)
    single = SingleCounter(num_inputs=3)
    train_model(single, train_data, in2i, epochs=100)
    print(f"  Learned deltas: a→{single._delta[0].item():+.2f}, b→{single._delta[1].item():+.2f}, c→{single._delta[2].item():+.2f}")

    print("\n" + "=" * 60)
    print("Training Dual Counter (CS level)...")
    print("=" * 60)
    dual = DualCounter(num_inputs=3)
    train_model(dual, train_data, in2i, epochs=100)
    print(f"  Counter1 (a-b): a→{dual._delta1[0].item():+.2f}, b→{dual._delta1[1].item():+.2f}, c→{dual._delta1[2].item():+.2f}")
    print(f"  Counter2 (a-c): a→{dual._delta2[0].item():+.2f}, b→{dual._delta2[1].item():+.2f}, c→{dual._delta2[2].item():+.2f}")

    # ============ Test generalization ============
    print("\n" + "=" * 60)
    print("Testing Generalization (trained on n≤5)")
    print("=" * 60)

    print(f"\n{'Max n':<10} {'FSA':<15} {'Single Ctr':<15} {'Dual Ctr':<15}")
    print("-" * 55)

    for test_n in [3, 5, 8, 10, 15, 20]:
        random.seed(789 + test_n)
        test_data = generate_anbncn_data(num_samples=100, max_n=test_n)

        fsa_sym, fsa_str = evaluate_model(fsa, test_data, in2i)
        single_sym, single_str = evaluate_model(single, test_data, in2i)
        dual_sym, dual_str = evaluate_model(dual, test_data, in2i)

        note = "(in-dist)" if test_n <= 5 else "(OOD!)"
        print(f"{test_n:<10} {fsa_sym:<15.1%} {single_sym:<15.1%} {dual_sym:<15.1%} {note}")

    # ============ Specific examples ============
    print("\n" + "=" * 60)
    print("Specific Examples")
    print("=" * 60)

    examples = [
        ("aabbcc", "valid a^2 b^2 c^2"),
        ("aaabbbccc", "valid a^3 b^3 c^3"),
        ("aabbbcc", "INVALID: too many b"),
        ("aabbccc", "INVALID: too many c"),
        ("abcabc", "INVALID: wrong order"),
        ("aaaabbbbcccc", "valid a^4 b^4 c^4 (OOD)"),
    ]

    for s, desc in examples:
        inp = [in2i[c] for c in s]
        labels = compute_validity_labels(s)

        with torch.no_grad():
            fsa_pred = fsa(inp).argmax(-1).tolist()
            single_pred = single(inp).argmax(-1).tolist()
            dual_pred = dual(inp).argmax(-1).tolist()

        print(f"\n  '{s}' ({desc})")
        print(f"    Truth:  {labels}")
        print(f"    FSA:    {fsa_pred}")
        print(f"    Single: {single_pred}")
        print(f"    Dual:   {dual_pred}")


if __name__ == "__main__":
    main()
