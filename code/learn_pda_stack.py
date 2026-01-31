"""
Differentiable Pushdown Automaton with Real Stack

Key idea: Stack as continuous operations
- Stack content: tensor of shape [max_depth, stack_dim]
- Stack pointer: soft distribution over positions
- Push: shift content up, write at bottom
- Pop: read from top, shift content down
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys

sys.stdout.reconfigure(line_buffering=True)


class NeuralStack(nn.Module):
    """
    Differentiable stack following Grefenstette et al. (2015)
    "Learning to Transduce with Unbounded Memory"
    """
    def __init__(self, stack_width, stack_depth=50):
        super().__init__()
        self.stack_width = stack_width
        self.stack_depth = stack_depth

    def forward(self, actions, values):
        """
        Process sequence of (action, value) pairs.

        actions: [seq_len, 3] - (push_prob, pop_prob, noop_prob) per step
        values: [seq_len, stack_width] - value to push per step

        Returns: [seq_len, stack_width] - stack top at each step
        """
        batch_size = 1  # Simplified: no batching
        device = actions.device

        # Stack: [depth, width]
        stack = torch.zeros(self.stack_depth, self.stack_width, device=device)
        # Strength at each position (how "real" is the content)
        strength = torch.zeros(self.stack_depth, device=device)

        outputs = []

        for t in range(len(actions)):
            push_prob = actions[t, 0]
            pop_prob = actions[t, 1]
            noop_prob = actions[t, 2]
            value = values[t]

            # Read current top (weighted by strength)
            if strength.sum() > 1e-6:
                # Find effective top
                read_weights = strength / (strength.sum() + 1e-6)
                top_value = (stack * read_weights.unsqueeze(1)).sum(dim=0)
            else:
                top_value = torch.zeros(self.stack_width, device=device)

            outputs.append(top_value)

            # Pop: reduce strength from top
            new_strength_pop = strength.clone()
            remaining = pop_prob
            for i in range(self.stack_depth - 1, -1, -1):
                if remaining <= 0:
                    break
                reduction = min(remaining, new_strength_pop[i])
                new_strength_pop[i] = new_strength_pop[i] - reduction
                remaining = remaining - reduction

            # Push: add new value at conceptual "top"
            # Shift everything down, add new at top
            new_stack_push = torch.zeros_like(stack)
            new_stack_push[1:] = stack[:-1]
            new_stack_push[0] = value

            new_strength_push = torch.zeros_like(strength)
            new_strength_push[1:] = strength[:-1]
            new_strength_push[0] = push_prob

            # Combine based on action probabilities
            # Noop: keep as is
            # Pop: use popped strength
            # Push: use pushed stack and strength

            stack = (noop_prob * stack +
                     pop_prob * stack +  # Stack content unchanged on pop
                     push_prob * new_stack_push)

            strength = (noop_prob * strength +
                        pop_prob * new_strength_pop +
                        push_prob * new_strength_push)

            # Normalize strength to prevent explosion
            if strength.sum() > self.stack_depth:
                strength = strength / strength.sum() * self.stack_depth

        return torch.stack(outputs)


class FuzzyPDA(nn.Module):
    """
    Fuzzy Pushdown Automaton with differentiable stack.
    """
    def __init__(self, num_states, num_inputs, num_outputs, stack_width=8, stack_depth=30):
        super().__init__()
        self.num_states = num_states
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.stack_width = stack_width

        # State transition
        self._T = nn.Parameter(torch.randn(num_states, num_inputs, num_states) * 0.1)

        # Output (depends on state, input, and stack top)
        self._O = nn.Linear(num_states + num_inputs + stack_width, num_outputs)

        # Stack actions (depends on state and input)
        self._action = nn.Linear(num_states + num_inputs, 3)  # push, pop, noop

        # Push value (depends on state and input)
        self._push_val = nn.Linear(num_states + num_inputs, stack_width)

        # Initial state
        self._init = nn.Parameter(torch.zeros(num_states))

        self.stack = NeuralStack(stack_width, stack_depth)

    def forward(self, input_seq):
        """Process input sequence."""
        device = self._init.device
        seq_len = len(input_seq)

        # Convert input to one-hot
        inp_onehot = torch.zeros(seq_len, self.num_inputs, device=device)
        for t, inp in enumerate(input_seq):
            inp_onehot[t, inp] = 1.0

        # Initial state distribution
        state = F.softmax(self._init, dim=0)

        # Collect actions and values for stack
        actions = []
        values = []
        states = []

        for t in range(seq_len):
            states.append(state.clone())

            # Concatenate state and input
            state_inp = torch.cat([state, inp_onehot[t]])

            # Compute stack action
            action_logits = self._action(state_inp)
            action = F.softmax(action_logits, dim=0)
            actions.append(action)

            # Compute push value
            push_val = torch.tanh(self._push_val(state_inp))
            values.append(push_val)

            # State transition
            T = F.softmax(self._T, dim=-1)
            inp_idx = input_seq[t]
            new_state = (T[:, inp_idx, :] * state.unsqueeze(1)).sum(0)
            state = new_state

        actions = torch.stack(actions)  # [seq_len, 3]
        values = torch.stack(values)    # [seq_len, stack_width]

        # Run stack
        stack_tops = self.stack(actions, values)  # [seq_len, stack_width]

        # Compute outputs
        outputs = []
        for t in range(seq_len):
            state_inp_stack = torch.cat([states[t], inp_onehot[t], stack_tops[t]])
            out_logits = self._O(state_inp_stack)
            out_prob = F.softmax(out_logits, dim=0)
            outputs.append(out_prob)

        return torch.stack(outputs)


def generate_depth_task(max_depth=10, max_length=30):
    """
    Generate balanced parentheses with depth at each position.
    """
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

        # Randomly stop
        if depth == 0 and len(seq) >= 4 and random.random() < 0.3:
            break

    return seq, depths


def main():
    print("=" * 60)
    print("Differentiable PDA with Neural Stack")
    print("=" * 60)

    in2i = {'(': 0, ')': 1}

    # Generate training data
    print("\nGenerating training data...")
    random.seed(42)
    train_data = []
    for _ in range(300):
        seq, depths = generate_depth_task(max_depth=15, max_length=25)
        train_data.append((seq, depths))

    max_depth_seen = max(max(d) for _, d in train_data)
    print(f"Max depth in training: {max_depth_seen}")

    num_outputs = max_depth_seen + 2  # 0 to max_depth + buffer

    # ============ Train FSA baseline ============
    print("\n" + "=" * 60)
    print("Training FSA (baseline, 15 states)...")
    print("=" * 60)

    class SimpleFSA(nn.Module):
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

    fsa = SimpleFSA(15, 2, num_outputs)
    opt_fsa = torch.optim.Adam(fsa.parameters(), lr=0.1)

    for ep in range(150):
        random.shuffle(train_data)
        correct = total = 0
        for seq, depths in train_data:
            inp = [in2i[s] for s in seq]
            out = [min(d, num_outputs-1) for d in depths]
            probs = fsa(inp)
            loss = F.cross_entropy(probs, torch.tensor(out))
            opt_fsa.zero_grad()
            loss.backward()
            opt_fsa.step()
            correct += (probs.argmax(-1) == torch.tensor(out)).sum().item()
            total += len(out)
        if ep % 50 == 0:
            print(f"  Epoch {ep}: {correct/total:.1%}")

    # ============ Train PDA ============
    print("\n" + "=" * 60)
    print("Training PDA with Neural Stack...")
    print("=" * 60)

    pda = FuzzyPDA(num_states=5, num_inputs=2, num_outputs=num_outputs,
                   stack_width=8, stack_depth=30)
    opt_pda = torch.optim.Adam(pda.parameters(), lr=0.05)

    for ep in range(150):
        random.shuffle(train_data)
        correct = total = 0
        for seq, depths in train_data:
            inp = [in2i[s] for s in seq]
            out = [min(d, num_outputs-1) for d in depths]
            probs = pda(inp)
            loss = F.cross_entropy(probs, torch.tensor(out))
            opt_pda.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pda.parameters(), 1.0)
            opt_pda.step()
            correct += (probs.argmax(-1) == torch.tensor(out)).sum().item()
            total += len(out)
        if ep % 50 == 0:
            print(f"  Epoch {ep}: {correct/total:.1%}")

    # ============ Test generalization ============
    print("\n" + "=" * 60)
    print("Testing Generalization by Max Depth")
    print("=" * 60)

    print(f"\n{'Max Depth':<12} {'FSA':<12} {'PDA':<12}")
    print("-" * 36)

    for test_depth in [5, 10, 15, 20, 25]:
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
                out = [min(d, num_outputs-1) for d in depths]

                fsa_preds = fsa(inp).argmax(-1).tolist()
                pda_preds = pda(inp).argmax(-1).tolist()

                fsa_correct += sum(p == t for p, t in zip(fsa_preds, out))
                pda_correct += sum(p == t for p, t in zip(pda_preds, out))
                fsa_total += len(out)
                pda_total += len(out)

        fsa_acc = fsa_correct / fsa_total
        pda_acc = pda_correct / pda_total
        winner = "← PDA wins!" if pda_acc > fsa_acc + 0.01 else ("← FSA wins!" if fsa_acc > pda_acc + 0.01 else "")
        print(f"{test_depth:<12} {fsa_acc:<12.1%} {pda_acc:<12.1%} {winner}")

    # Show example
    print("\n" + "=" * 60)
    print("Example Prediction")
    print("=" * 60)

    random.seed(999)
    test_seq, test_depths = generate_depth_task(max_depth=12, max_length=20)
    inp = [in2i[s] for s in test_seq]

    with torch.no_grad():
        fsa_pred = fsa(inp).argmax(-1).tolist()
        pda_pred = pda(inp).argmax(-1).tolist()

    print(f"\nInput:  {' '.join(test_seq)}")
    print(f"Truth:  {' '.join(map(str, test_depths))}")
    print(f"FSA:    {' '.join(map(str, fsa_pred))}")
    print(f"PDA:    {' '.join(map(str, pda_pred))}")


if __name__ == "__main__":
    main()
