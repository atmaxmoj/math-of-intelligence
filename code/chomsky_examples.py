"""
More Chomsky Hierarchy Examples

Level 3 (Regular):
- Binary divisibility by 3
- Strings ending with "ab"

Level 2 (Context-Free):
- Matching brackets: (), [], {}
- a^n b^n

Level 1 (Context-Sensitive):
- Copy language: ww (same string repeated)
- a^n b^n c^n d^n (4-way count match)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys

sys.stdout.reconfigure(line_buffering=True)


# ============ Models ============

class FSA(nn.Module):
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


class Counter(nn.Module):
    """Single counter for CF languages."""
    def __init__(self, num_inputs, init_deltas=None):
        super().__init__()
        if init_deltas is not None:
            self._delta = nn.Parameter(torch.tensor(init_deltas, dtype=torch.float))
        else:
            self._delta = nn.Parameter(torch.zeros(num_inputs))

    def forward(self, input_seq):
        counter = 0.0
        outputs = []
        for inp in input_seq:
            counter = counter + self._delta[inp]
            outputs.append(counter)
        return torch.stack(outputs)


class MultiCounter(nn.Module):
    """Multiple counters for CS languages."""
    def __init__(self, num_inputs, num_counters, init_deltas=None):
        super().__init__()
        self.num_counters = num_counters
        if init_deltas is not None:
            self._delta = nn.Parameter(torch.tensor(init_deltas, dtype=torch.float))
        else:
            self._delta = nn.Parameter(torch.zeros(num_counters, num_inputs))

    def forward(self, input_seq):
        counters = torch.zeros(self.num_counters)
        outputs = []
        for inp in input_seq:
            counters = counters + self._delta[:, inp]
            outputs.append(counters.clone())
        return torch.stack(outputs)


# ============ Training Utils ============

def train_regression(model, train_data, in2i, epochs=100, lr=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        random.shuffle(train_data)
        total_loss = 0
        for s, targets in train_data:
            inp = [in2i[c] for c in s]
            preds = model(inp)
            targets_t = torch.tensor(targets, dtype=torch.float)
            loss = F.mse_loss(preds.squeeze(), targets_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if ep % 25 == 0:
            print(f"    Epoch {ep}: loss={total_loss/len(train_data):.4f}")


def train_classification(model, train_data, in2i, epochs=100, lr=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        random.shuffle(train_data)
        correct = total = 0
        for s, labels in train_data:
            inp = [in2i[c] for c in s]
            probs = model(inp)
            targets = torch.tensor(labels)
            loss = F.cross_entropy(probs, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (probs.argmax(-1) == targets).sum().item()
            total += len(labels)
        if ep % 25 == 0:
            print(f"    Epoch {ep}: {correct/total:.1%}")


# ============ Example 1: Binary Divisibility by 3 (Regular) ============

def example_divisibility():
    print("\n" + "=" * 70)
    print("LEVEL 3 (Regular): Binary Divisibility by 3")
    print("  Input: binary string")
    print("  Output at each position: current value mod 3")
    print("=" * 70)

    def generate_data(num_samples, max_len):
        data = []
        for _ in range(num_samples):
            length = random.randint(1, max_len)
            s = ''.join(random.choice('01') for _ in range(length))
            # Compute value mod 3 at each position
            val = 0
            mods = []
            for c in s:
                val = (val * 2 + int(c)) % 3
                mods.append(val)
            data.append((s, mods))
        return data

    in2i = {'0': 0, '1': 1}

    print("\nGenerating data (max_len=8)...")
    random.seed(42)
    train_data = generate_data(200, 8)

    print("\nTraining FSA (3 states - one per mod value)...")
    fsa = FSA(num_states=3, num_inputs=2, num_outputs=3)
    train_classification(fsa, train_data, in2i, epochs=100)

    # Test generalization
    print("\nTesting on longer strings (max_len=20)...")
    random.seed(123)
    test_data = generate_data(100, 20)
    correct = total = 0
    with torch.no_grad():
        for s, mods in test_data:
            inp = [in2i[c] for c in s]
            preds = fsa(inp).argmax(-1).tolist()
            correct += sum(p == t for p, t in zip(preds, mods))
            total += len(mods)
    print(f"  Accuracy: {correct/total:.1%}")

    # Example
    s = "110101"
    inp = [in2i[c] for c in s]
    with torch.no_grad():
        preds = fsa(inp).argmax(-1).tolist()
    val = 0
    truth = []
    for c in s:
        val = (val * 2 + int(c)) % 3
        truth.append(val)
    print(f"\n  Example: '{s}' (decimal: {int(s, 2)}, mod 3 = {int(s,2)%3})")
    print(f"    Truth:  {truth}")
    print(f"    FSA:    {preds}")


# ============ Example 2: Matching Brackets (CF) ============

def example_brackets():
    print("\n" + "=" * 70)
    print("LEVEL 2 (Context-Free): Multi-type Bracket Matching")
    print("  Input: string of ()[]")
    print("  Output: depth at each position, -1 if invalid")
    print("=" * 70)

    def generate_balanced(max_depth, max_len):
        """Generate balanced bracket string."""
        seq = []
        stack = []
        open_b = ['(', '[']
        close_b = [')', ']']
        match = {'(': ')', '[': ']'}

        while len(seq) < max_len:
            if len(stack) == 0:
                b = random.choice(open_b)
                seq.append(b)
                stack.append(b)
            elif len(stack) >= max_depth:
                seq.append(match[stack.pop()])
            elif random.random() < 0.5:
                b = random.choice(open_b)
                seq.append(b)
                stack.append(b)
            else:
                seq.append(match[stack.pop()])

            if len(stack) == 0 and len(seq) >= 4 and random.random() < 0.3:
                break

        while stack:
            seq.append(match[stack.pop()])

        return ''.join(seq)

    def compute_depths(s):
        """Compute depth at each position, -1 if mismatched."""
        depths = []
        stack = []
        match = {')': '(', ']': '['}
        valid = True

        for c in s:
            if not valid:
                depths.append(-1)
                continue

            if c in '([':
                stack.append(c)
                depths.append(len(stack))
            else:
                if not stack or stack[-1] != match[c]:
                    valid = False
                    depths.append(-1)
                else:
                    depths.append(len(stack))
                    stack.pop()

        return depths

    def generate_data(num_samples, max_depth, max_len, include_invalid=True):
        data = []
        for _ in range(num_samples):
            if random.random() < 0.6 or not include_invalid:
                s = generate_balanced(max_depth, max_len)
            else:
                # Invalid: random string
                length = random.randint(2, max_len)
                s = ''.join(random.choice('()[]') for _ in range(length))
            depths = compute_depths(s)
            data.append((s, depths))
        return data

    in2i = {'(': 0, ')': 1, '[': 2, ']': 3}

    print("\nGenerating data (max_depth=5)...")
    random.seed(42)
    train_data = generate_data(200, 5, 15)

    print("\nTraining Counter (regression)...")

    class BracketCounter(nn.Module):
        def __init__(self):
            super().__init__()
            # ( and [ increment, ) and ] decrement
            self._delta = nn.Parameter(torch.tensor([1.0, -1.0, 1.0, -1.0]))

        def forward(self, input_seq):
            counter = 0.0
            outputs = []
            for inp in input_seq:
                counter = counter + self._delta[inp]
                outputs.append(F.leaky_relu(counter, 0.01))
            return torch.stack(outputs)

    counter = BracketCounter()
    train_regression(counter, train_data, in2i, epochs=100, lr=0.2)

    print(f"\n  Learned deltas: (→{counter._delta[0].item():+.2f}, )→{counter._delta[1].item():+.2f}, "
          f"[→{counter._delta[2].item():+.2f}, ]→{counter._delta[3].item():+.2f}")

    # Test
    print("\nTesting on deeper brackets (max_depth=10)...")
    random.seed(456)
    test_data = generate_data(100, 10, 25, include_invalid=False)
    correct = total = 0
    with torch.no_grad():
        for s, depths in test_data:
            inp = [in2i[c] for c in s]
            preds = counter(inp).round().long().tolist()
            correct += sum(p == t for p, t in zip(preds, depths))
            total += len(depths)
    print(f"  Accuracy: {correct/total:.1%}")

    # Example
    s = "([()][])"
    inp = [in2i[c] for c in s]
    depths = compute_depths(s)
    with torch.no_grad():
        preds = counter(inp).round().long().tolist()
    print(f"\n  Example: '{s}'")
    print(f"    Truth:  {depths}")
    print(f"    Pred:   {preds}")


# ============ Example 3: Copy Language ww (CS) ============

def example_copy_language():
    print("\n" + "=" * 70)
    print("LEVEL 1 (Context-Sensitive): Copy Language ww")
    print("  Input: string over {a,b}")
    print("  Task: Is it of form ww (same string repeated)?")
    print("  This requires matching symbols across distance!")
    print("=" * 70)

    def generate_ww(max_len):
        """Generate valid ww string."""
        w_len = random.randint(1, max_len // 2)
        w = ''.join(random.choice('ab') for _ in range(w_len))
        return w + w

    def is_ww(s):
        if len(s) % 2 != 0:
            return False
        half = len(s) // 2
        return s[:half] == s[half:]

    def generate_data(num_samples, max_len):
        data = []
        for _ in range(num_samples):
            if random.random() < 0.5:
                s = generate_ww(max_len)
                label = 1
            else:
                length = random.randint(2, max_len)
                s = ''.join(random.choice('ab') for _ in range(length))
                label = 1 if is_ww(s) else 0
            data.append((s, label))
        return data

    in2i = {'a': 0, 'b': 1}

    print("\nGenerating data (max_len=10)...")
    random.seed(42)
    train_data = generate_data(300, 10)
    valid_count = sum(1 for _, l in train_data if l == 1)
    print(f"  {len(train_data)} samples, {valid_count} valid ww")

    # FSA approach
    print("\nTraining FSA (20 states)...")

    class FSAClassifier(nn.Module):
        def __init__(self, num_states, num_inputs):
            super().__init__()
            self._T = nn.Parameter(torch.randn(num_states, num_inputs, num_states) * 0.1)
            self._init = nn.Parameter(torch.zeros(num_states))
            self._out = nn.Linear(num_states, 2)

        def forward(self, input_seq):
            T = F.softmax(self._T, dim=-1)
            state = F.softmax(self._init, dim=0)
            for inp in input_seq:
                state = (T[:, inp, :] * state.unsqueeze(1)).sum(0)
            return self._out(state)

    fsa = FSAClassifier(20, 2)
    opt = torch.optim.Adam(fsa.parameters(), lr=0.1)

    for ep in range(100):
        random.shuffle(train_data)
        correct = total = 0
        for s, label in train_data:
            inp = [in2i[c] for c in s]
            logits = fsa(inp)
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([label]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            pred = logits.argmax().item()
            correct += (pred == label)
            total += 1
        if ep % 25 == 0:
            print(f"    Epoch {ep}: {correct/total:.1%}")

    # Test
    print("\nTesting on longer strings (max_len=20)...")
    random.seed(789)
    test_data = generate_data(200, 20)
    correct = total = 0
    with torch.no_grad():
        for s, label in test_data:
            inp = [in2i[c] for c in s]
            pred = fsa(inp).argmax().item()
            correct += (pred == label)
            total += 1
    print(f"  FSA Accuracy: {correct/total:.1%}")

    # Examples
    print("\n  Examples:")
    for s in ["abab", "abba", "abaaba", "aabb", "aabaab"]:
        inp = [in2i[c] for c in s]
        with torch.no_grad():
            pred = fsa(inp).argmax().item()
        truth = 1 if is_ww(s) else 0
        status = "✓" if pred == truth else "✗"
        print(f"    '{s}': truth={truth}, pred={pred} {status}")

    print("\n  Note: FSA struggles with ww because it requires matching")
    print("        symbols at positions i and n/2+i (unbounded distance)")


# ============ Example 4: Four-way Count Match (CS) ============

def example_four_counts():
    print("\n" + "=" * 70)
    print("LEVEL 1 (Context-Sensitive): a^n b^n c^n d^n")
    print("  Even harder than a^n b^n c^n!")
    print("  Requires tracking 3 independent count constraints")
    print("=" * 70)

    def generate_valid(max_n):
        n = random.randint(1, max_n)
        return 'a' * n + 'b' * n + 'c' * n + 'd' * n

    def is_valid_prefix(s):
        """Check if s is a valid prefix of some a^n b^n c^n d^n."""
        counts = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
        phase_order = 'abcd'
        phase = 0

        for c in s:
            if c not in 'abcd':
                return False

            expected_phase = phase_order[phase]

            if c == expected_phase:
                counts[c] += 1
            elif phase < 3 and c == phase_order[phase + 1]:
                # Transition to next phase
                if phase > 0:
                    # Check previous count constraint
                    prev = phase_order[phase]
                    if counts[prev] != counts['a']:
                        return False
                phase += 1
                counts[c] = 1
            else:
                return False

            # Current count should not exceed a's count
            if counts[c] > counts['a']:
                return False

        return True

    def generate_data(num_samples, max_n):
        data = []
        for _ in range(num_samples):
            if random.random() < 0.5:
                s = generate_valid(max_n)
                # All positions valid
                labels = [1] * len(s)
            else:
                # Invalid
                na = random.randint(1, max_n)
                nb = random.randint(1, max_n)
                nc = random.randint(1, max_n)
                nd = random.randint(1, max_n)
                s = 'a' * na + 'b' * nb + 'c' * nc + 'd' * nd
                labels = []
                for i in range(len(s)):
                    labels.append(1 if is_valid_prefix(s[:i+1]) else 0)
            data.append((s, labels))
        return data

    in2i = {'a': 0, 'b': 1, 'c': 2, 'd': 3}

    print("\nGenerating data (max_n=4)...")
    random.seed(42)
    train_data = generate_data(300, 4)

    # Multi-counter model
    print("\nTraining Multi-Counter (3 counters)...")

    class ThreeCounter(nn.Module):
        def __init__(self):
            super().__init__()
            # Counter 1: a - b
            # Counter 2: a - c
            # Counter 3: a - d
            self._delta = nn.Parameter(torch.tensor([
                [1.0, -1.0, 0.0, 0.0],   # a-b: a+1, b-1
                [1.0, 0.0, -1.0, 0.0],   # a-c: a+1, c-1
                [1.0, 0.0, 0.0, -1.0],   # a-d: a+1, d-1
            ]))

        def forward(self, input_seq):
            counters = torch.zeros(3)
            outputs = []
            for inp in input_seq:
                counters = counters + self._delta[:, inp]
                # Valid if all counters >= 0
                valid_probs = torch.sigmoid(counters * 5)
                valid = valid_probs.prod()
                outputs.append(torch.stack([1 - valid, valid]))
            return torch.stack(outputs)

    counter = ThreeCounter()
    train_classification(counter, train_data, in2i, epochs=100, lr=0.1)

    # Test
    print("\nTesting on larger n (max_n=8)...")
    random.seed(456)
    test_data = generate_data(100, 8)
    correct = total = 0
    with torch.no_grad():
        for s, labels in test_data:
            inp = [in2i[c] for c in s]
            preds = counter(inp).argmax(-1).tolist()
            correct += sum(p == t for p, t in zip(preds, labels))
            total += len(labels)
    print(f"  Accuracy: {correct/total:.1%}")

    # Examples
    print("\n  Examples:")
    for s in ["aabbccdd", "aaabbbcccddd", "aabbccd", "aabbccdddd"]:
        inp = [in2i[c] for c in s]
        with torch.no_grad():
            preds = counter(inp).argmax(-1).tolist()
        labels = []
        for i in range(len(s)):
            labels.append(1 if is_valid_prefix(s[:i+1]) else 0)
        match = "✓" if preds == labels else "✗"
        print(f"    '{s}': truth={labels[-3:]}, pred={preds[-3:]} {match}")


# ============ Main ============

def main():
    print("=" * 70)
    print("Chomsky Hierarchy: More Examples")
    print("=" * 70)

    example_divisibility()    # Level 3: Regular
    example_brackets()        # Level 2: Context-Free
    example_copy_language()   # Level 1: Context-Sensitive
    example_four_counts()     # Level 1: Context-Sensitive (harder)


if __name__ == "__main__":
    main()
