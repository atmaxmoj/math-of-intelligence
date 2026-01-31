"""
Climb Chomsky Hierarchy: Level 2 - Context-Free

Learn a Pushdown Transducer using differentiable stack.

Task: Learn parenthesis matching
  Input:  ( ( ) ( ) )
  Output: 1 2 2 2 2 1   (depth at each position)

Or: bracket transformation
  Input:  ( ( ) )
  Output: [ [ ] ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys

sys.stdout.reconfigure(line_buffering=True)


class DifferentiableStack:
    """
    Soft stack using attention-like mechanism.
    Stack is a fixed-size buffer with soft pointer.
    """
    def __init__(self, stack_size, stack_dim):
        self.stack_size = stack_size
        self.stack_dim = stack_dim
        self.reset()

    def reset(self):
        # Stack content: [stack_size, stack_dim]
        self.content = torch.zeros(self.stack_size, self.stack_dim)
        # Soft pointer distribution over positions
        self.pointer = torch.zeros(self.stack_size)
        self.pointer[0] = 1.0  # Start at bottom

    def push(self, value, strength=1.0):
        """Push value onto stack with given strength."""
        # Move pointer up
        new_pointer = torch.zeros_like(self.pointer)
        new_pointer[1:] = self.pointer[:-1]  # Shift up
        new_pointer[0] = 0  # Can't go below bottom

        # Interpolate between old and new pointer
        self.pointer = strength * new_pointer + (1 - strength) * self.pointer

        # Write value at current position
        write_weights = self.pointer.unsqueeze(1)  # [stack_size, 1]
        self.content = self.content * (1 - write_weights) + value * write_weights

    def pop(self, strength=1.0):
        """Pop from stack, return value."""
        # Read at current position
        read_weights = self.pointer.unsqueeze(1)  # [stack_size, 1]
        value = (self.content * read_weights).sum(dim=0)

        # Move pointer down
        new_pointer = torch.zeros_like(self.pointer)
        new_pointer[:-1] = self.pointer[1:]  # Shift down
        new_pointer[-1] = 0

        self.pointer = strength * new_pointer + (1 - strength) * self.pointer

        return value

    def top(self):
        """Read top of stack without popping."""
        read_weights = self.pointer.unsqueeze(1)
        return (self.content * read_weights).sum(dim=0)


class FuzzyPushdownTransducer(nn.Module):
    """
    Differentiable Pushdown Transducer.

    At each step:
    1. Read input symbol
    2. Look at current state and stack top
    3. Decide: push/pop/noop and what to output
    4. Transition to next state
    """
    def __init__(self, num_states, num_inputs, num_outputs, stack_size=20, stack_dim=8):
        super().__init__()
        self.num_states = num_states
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.stack_size = stack_size
        self.stack_dim = stack_dim

        # State transition: (state, input, stack_top) → next_state
        self._T = nn.Parameter(torch.randn(num_states, num_inputs, stack_dim, num_states) * 0.1)

        # Output: (state, input, stack_top) → output
        self._O = nn.Parameter(torch.randn(num_states, num_inputs, stack_dim, num_outputs) * 0.1)

        # Stack action: (state, input) → (push_strength, pop_strength, push_value)
        self._push_gate = nn.Parameter(torch.randn(num_states, num_inputs, 1) * 0.1)
        self._pop_gate = nn.Parameter(torch.randn(num_states, num_inputs, 1) * 0.1)
        self._push_value = nn.Parameter(torch.randn(num_states, num_inputs, stack_dim) * 0.1)

        # Initial state
        self._init = nn.Parameter(torch.zeros(num_states))

    def forward(self, input_seq):
        """Process input sequence, return output probabilities."""
        # Initialize
        state = F.softmax(self._init, dim=0)
        stack = DifferentiableStack(self.stack_size, self.stack_dim)

        outputs = []

        for inp in input_seq:
            stack_top = stack.top()

            # Compute output: weighted combination over states and stack
            # O[state, input, stack_top] → output distribution
            O = F.softmax(self._O, dim=-1)
            out_prob = torch.zeros(self.num_outputs)
            for s in range(self.num_states):
                # Weight by state probability
                weight = state[s]
                # Stack-dependent output
                stack_weight = F.softmax(torch.matmul(stack_top, self._O[s, inp]), dim=-1)
                out_prob = out_prob + weight * stack_weight

            outputs.append(out_prob)

            # Stack operations
            push_strength = torch.sigmoid(self._push_gate[torch.arange(self.num_states), inp].squeeze() @ state)
            pop_strength = torch.sigmoid(self._pop_gate[torch.arange(self.num_states), inp].squeeze() @ state)
            push_value = torch.tanh(self._push_value[:, inp, :].T @ state)  # [stack_dim]

            # Execute stack ops
            if pop_strength > 0.01:
                stack.pop(pop_strength.item())
            if push_strength > 0.01:
                stack.push(push_value, push_strength.item())

            # State transition
            T = F.softmax(self._T, dim=-1)
            new_state = torch.zeros(self.num_states)
            for s in range(self.num_states):
                # Transition weighted by stack top
                trans_weight = F.softmax(torch.matmul(stack_top, self._T[s, inp]), dim=-1)
                new_state = new_state + state[s] * trans_weight

            state = new_state

        return torch.stack(outputs)


class SimplePushdownTransducer(nn.Module):
    """
    Simpler version: just count depth, output depth.
    No explicit stack, but learns depth-like behavior.
    """
    def __init__(self, num_states, num_inputs, num_outputs):
        super().__init__()
        self.num_states = num_states
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


def generate_balanced_parens(max_depth=5, max_length=20):
    """Generate balanced parentheses with depth labels."""
    seq = []
    depths = []
    depth = 0

    length = random.randint(2, max_length)

    for _ in range(length):
        if depth == 0:
            # Must open
            seq.append('(')
            depth += 1
        elif depth >= max_depth or (random.random() < 0.5 and len(seq) < length - depth):
            # Close
            seq.append(')')
            depth -= 1
        else:
            # Open
            seq.append('(')
            depth += 1
        depths.append(depth)

    # Close remaining
    while depth > 0:
        seq.append(')')
        depth -= 1
        depths.append(depth)

    return seq, depths


def generate_bracket_transform(max_depth=5, max_length=20):
    """
    Input: ( ( ) )
    Output: [ [ ] ]

    This requires memory to know depth, but FSA can't do it properly
    for arbitrary nesting. Let's see if our model can learn it.
    """
    parens, depths = generate_balanced_parens(max_depth, max_length)
    brackets = ['[' if p == '(' else ']' for p in parens]
    return parens, brackets


def main():
    print("=" * 60)
    print("Chomsky Level 2: Learning Context-Free Transductions")
    print("=" * 60)

    # Task 1: Parenthesis → Depth
    print("\n" + "=" * 60)
    print("Task 1: Parenthesis → Depth")
    print("  Input:  ( ( ) ( ) )")
    print("  Output: 1 2 1 2 1 0")
    print("=" * 60)

    # Generate training data
    random.seed(42)
    train_data = []
    for _ in range(200):
        seq, depths = generate_balanced_parens(max_depth=6, max_length=15)
        train_data.append((seq, depths))

    in2i = {'(': 0, ')': 1}
    out2i = {i: i for i in range(10)}  # depths 0-9

    # Try with FSA (should fail for deep nesting)
    print("\nTraining FSA (should struggle with deep nesting)...")
    fsa = SimplePushdownTransducer(num_states=10, num_inputs=2, num_outputs=10)
    opt = torch.optim.Adam(fsa.parameters(), lr=0.1)

    for ep in range(200):
        random.shuffle(train_data)
        correct = total = 0
        for seq, depths in train_data:
            inp = [in2i[s] for s in seq]
            out = [min(d, 9) for d in depths]  # cap at 9
            probs = fsa(inp)
            loss = F.cross_entropy(probs, torch.tensor(out))
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (probs.argmax(-1) == torch.tensor(out)).sum().item()
            total += len(out)
        if ep % 50 == 0:
            print(f"  Epoch {ep}: {correct/total:.1%}")

    # Test on different depths
    print("\nTesting by max depth:")
    for test_depth in [2, 4, 6, 8]:
        test_data = []
        random.seed(123 + test_depth)
        for _ in range(50):
            seq, depths = generate_balanced_parens(max_depth=test_depth, max_length=20)
            test_data.append((seq, depths))

        correct = total = 0
        with torch.no_grad():
            for seq, depths in test_data:
                inp = [in2i[s] for s in seq]
                out = [min(d, 9) for d in depths]
                preds = fsa(inp).argmax(-1).tolist()
                correct += sum(p == t for p, t in zip(preds, out))
                total += len(out)
        print(f"  max_depth={test_depth}: {correct/total:.1%}")

    # Task 2: Bracket transformation
    print("\n" + "=" * 60)
    print("Task 2: Parenthesis → Bracket (same structure)")
    print("  Input:  ( ( ) )")
    print("  Output: [ [ ] ]")
    print("=" * 60)

    random.seed(42)
    train_data2 = []
    for _ in range(200):
        parens, brackets = generate_bracket_transform(max_depth=6, max_length=15)
        train_data2.append((parens, brackets))

    in2i2 = {'(': 0, ')': 1}
    out2i2 = {'[': 0, ']': 1}

    # This is actually regular! ( → [, ) → ]
    # But let's verify FSA can learn it easily
    print("\nTraining FSA for bracket transform...")
    fsa2 = SimplePushdownTransducer(num_states=4, num_inputs=2, num_outputs=2)
    opt2 = torch.optim.Adam(fsa2.parameters(), lr=0.1)

    for ep in range(100):
        random.shuffle(train_data2)
        correct = total = 0
        for parens, brackets in train_data2:
            inp = [in2i2[s] for s in parens]
            out = [out2i2[s] for s in brackets]
            probs = fsa2(inp)
            loss = F.cross_entropy(probs, torch.tensor(out))
            opt2.zero_grad()
            loss.backward()
            opt2.step()
            correct += (probs.argmax(-1) == torch.tensor(out)).sum().item()
            total += len(out)
        if ep % 25 == 0:
            print(f"  Epoch {ep}: {correct/total:.1%}")

    print("\n  → This is trivially regular: just map ( → [, ) → ]")

    # Task 3: Something truly context-free
    print("\n" + "=" * 60)
    print("Task 3: Dyck language check (needs counting)")
    print("  Valid:   ( ( ) ) → 1 1 1 1")
    print("  Invalid: ( ) ) ( → 1 1 0 0")
    print("=" * 60)

    def generate_dyck_check(max_len=15):
        """Generate strings and whether they're valid at each position."""
        seq = []
        valid = []
        depth = 0
        length = random.randint(4, max_len)

        for _ in range(length):
            if random.random() < 0.5:
                seq.append('(')
                depth += 1
            else:
                seq.append(')')
                depth -= 1
            # Valid if depth >= 0
            valid.append(1 if depth >= 0 else 0)

        return seq, valid

    random.seed(42)
    train_data3 = [generate_dyck_check(15) for _ in range(300)]

    print("\nTraining FSA for Dyck validity check...")
    fsa3 = SimplePushdownTransducer(num_states=15, num_inputs=2, num_outputs=2)
    opt3 = torch.optim.Adam(fsa3.parameters(), lr=0.1)

    for ep in range(200):
        random.shuffle(train_data3)
        correct = total = 0
        for seq, valid in train_data3:
            inp = [in2i[s] for s in seq]
            probs = fsa3(inp)
            loss = F.cross_entropy(probs, torch.tensor(valid))
            opt3.zero_grad()
            loss.backward()
            opt3.step()
            correct += (probs.argmax(-1) == torch.tensor(valid)).sum().item()
            total += len(valid)
        if ep % 50 == 0:
            print(f"  Epoch {ep}: {correct/total:.1%}")

    # Test generalization
    print("\n  Testing on longer sequences...")
    for test_len in [10, 20, 30, 50]:
        random.seed(456 + test_len)
        test_data = [generate_dyck_check(test_len) for _ in range(100)]
        correct = total = 0
        with torch.no_grad():
            for seq, valid in test_data:
                inp = [in2i[s] for s in seq]
                preds = fsa3(inp).argmax(-1).tolist()
                correct += sum(p == t for p, t in zip(preds, valid))
                total += len(valid)
        print(f"    max_len={test_len}: {correct/total:.1%}")


if __name__ == "__main__":
    main()
