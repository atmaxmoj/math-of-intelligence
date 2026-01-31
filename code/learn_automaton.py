"""
Learn Automaton from Traces

Benchmark: automata.cs.ru.nl Mealy machines

Pipeline:
1. Parse .dot file → ground truth automaton
2. Generate input-output traces
3. Learn fuzzy automaton via gradient descent
4. Discretize → compare with ground truth
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import random


# =============================================================================
# Parse DOT file
# =============================================================================

def parse_dot(filepath):
    """
    Parse Mealy machine from .dot file

    Returns:
        states: list of state names
        inputs: list of input symbols
        outputs: list of output symbols
        transitions: dict (state, input) -> (next_state, output)
        initial: initial state
    """
    with open(filepath) as f:
        content = f.read()

    states = set()
    transitions = {}
    inputs = set()
    outputs = set()
    initial = None

    # Find initial state
    init_match = re.search(r'__start\d*\s*->\s*(\w+)', content)
    if init_match:
        initial = init_match.group(1)

    # Find states
    for match in re.finditer(r'(\w+)\s*\[shape="circle"', content):
        states.add(match.group(1))

    # Find transitions: q0 -> q1 [label="a / A"]
    for match in re.finditer(r'(\w+)\s*->\s*(\w+)\s*\[label="(\w+)\s*/\s*(\w+)"\]', content):
        src, dst, inp, out = match.groups()
        states.add(src)
        states.add(dst)
        inputs.add(inp)
        outputs.add(out)
        transitions[(src, inp)] = (dst, out)

    return {
        'states': sorted(states),
        'inputs': sorted(inputs),
        'outputs': sorted(outputs),
        'transitions': transitions,
        'initial': initial
    }


def generate_traces(automaton, num_traces=100, max_length=10):
    """Generate input-output traces from automaton"""
    traces = []

    for _ in range(num_traces):
        state = automaton['initial']
        length = random.randint(1, max_length)

        input_seq = []
        output_seq = []

        for _ in range(length):
            inp = random.choice(automaton['inputs'])
            if (state, inp) in automaton['transitions']:
                next_state, out = automaton['transitions'][(state, inp)]
                input_seq.append(inp)
                output_seq.append(out)
                state = next_state
            else:
                break

        if input_seq:
            traces.append((input_seq, output_seq))

    return traces


# =============================================================================
# Fuzzy Mealy Machine
# =============================================================================

class FuzzyMealyMachine(nn.Module):
    """
    Fuzzy Mealy Machine with learnable transitions

    Mealy: output depends on transition (state, input) -> (next_state, output)

    Parameters:
        - T[s, i, s']: transition probability from s to s' on input i
        - O[s, i, o]: output probability o on transition from s with input i
    """

    def __init__(self, num_states, num_inputs, num_outputs):
        super().__init__()
        self.num_states = num_states
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # Transition logits: (num_states, num_inputs, num_states)
        self._T = nn.Parameter(torch.randn(num_states, num_inputs, num_states))

        # Output logits: (num_states, num_inputs, num_outputs)
        self._O = nn.Parameter(torch.randn(num_states, num_inputs, num_outputs))

    @property
    def T(self):
        """Transition probabilities (softmax over next states)"""
        return F.softmax(self._T, dim=-1)

    @property
    def O(self):
        """Output probabilities (softmax over outputs)"""
        return F.softmax(self._O, dim=-1)

    def forward(self, input_seq, initial_state=0):
        """
        Process input sequence, return output probabilities

        Args:
            input_seq: list of input indices
            initial_state: index of initial state

        Returns:
            output_probs: (seq_len, num_outputs) probabilities
        """
        batch_size = 1
        state_dist = torch.zeros(self.num_states)
        state_dist[initial_state] = 1.0

        output_probs = []

        for inp in input_seq:
            # Output distribution: sum over states weighted by state distribution
            # O[s, inp, o] * state_dist[s]
            out_prob = (self.O[:, inp, :] * state_dist.unsqueeze(1)).sum(dim=0)
            output_probs.append(out_prob)

            # State transition: T[s, inp, s'] * state_dist[s]
            new_state_dist = (self.T[:, inp, :] * state_dist.unsqueeze(1)).sum(dim=0)
            state_dist = new_state_dist

        return torch.stack(output_probs)  # (seq_len, num_outputs)

    def sparsity_loss(self):
        """Encourage discrete transitions (one-hot)"""
        T = self.T
        O = self.O
        # Entropy loss: lower entropy = more discrete
        T_entropy = -(T * (T + 1e-10).log()).sum()
        O_entropy = -(O * (O + 1e-10).log()).sum()
        return T_entropy + O_entropy

    def discretize(self):
        """Return discrete (argmax) transitions"""
        T_discrete = self.T.argmax(dim=-1)  # (num_states, num_inputs)
        O_discrete = self.O.argmax(dim=-1)  # (num_states, num_inputs)
        return T_discrete, O_discrete


# =============================================================================
# Training
# =============================================================================

def train_on_traces(model, traces, input_to_idx, output_to_idx, epochs=500, lr=0.1):
    """Train fuzzy Mealy machine on traces"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_count = 0

        for input_seq, output_seq in traces:
            # Convert to indices
            inp_indices = [input_to_idx[x] for x in input_seq]
            out_indices = [output_to_idx[x] for x in output_seq]

            # Forward
            out_probs = model(inp_indices)  # (seq_len, num_outputs)

            # Cross-entropy loss
            targets = torch.tensor(out_indices)
            loss = F.cross_entropy(out_probs, targets)

            # Sparsity loss
            loss = loss + model.sparsity_loss() * 0.001

            total_loss += loss.item()

            # Accuracy
            preds = out_probs.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += len(targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            acc = total_correct / total_count
            print(f"Epoch {epoch}: loss={total_loss/len(traces):.4f}, acc={acc:.2%}")

    return model


# =============================================================================
# Evaluation
# =============================================================================

def compare_with_ground_truth(model, automaton, input_to_idx, output_to_idx):
    """Compare learned automaton with ground truth"""
    idx_to_state = {i: s for i, s in enumerate(automaton['states'])}
    idx_to_input = {i: x for i, x in enumerate(automaton['inputs'])}
    idx_to_output = {i: x for i, x in enumerate(automaton['outputs'])}

    T_discrete, O_discrete = model.discretize()

    print("\nGround Truth Transitions:")
    for (src, inp), (dst, out) in sorted(automaton['transitions'].items()):
        print(f"  {src} --{inp}/{out}--> {dst}")

    print("\nLearned Transitions (discretized):")
    state_map = {automaton['initial']: 0}  # Map ground truth states to learned indices

    # Try to align states (this is a simplification)
    for s_idx in range(model.num_states):
        for i_idx in range(model.num_inputs):
            next_s = T_discrete[s_idx, i_idx].item()
            out_idx = O_discrete[s_idx, i_idx].item()
            inp = idx_to_input[i_idx]
            out = idx_to_output[out_idx]
            print(f"  s{s_idx} --{inp}/{out}--> s{next_s}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Learning Mealy Machine from Traces")
    print("=" * 60)

    # Load benchmark
    dot_path = Path(__file__).parent.parent / "datasets" / "automata" / "Mealy" / "principle" / "BenchmarkToyModels" / "cacm.dot"

    if not dot_path.exists():
        print(f"File not found: {dot_path}")
        return

    print(f"\nLoading: {dot_path.name}")
    automaton = parse_dot(dot_path)

    print(f"States: {automaton['states']}")
    print(f"Inputs: {automaton['inputs']}")
    print(f"Outputs: {automaton['outputs']}")
    print(f"Initial: {automaton['initial']}")
    print(f"Transitions: {len(automaton['transitions'])}")

    # Generate traces
    print("\nGenerating traces...")
    traces = generate_traces(automaton, num_traces=200, max_length=15)
    print(f"Generated {len(traces)} traces")
    print(f"Example: {traces[0]}")

    # Create model
    input_to_idx = {x: i for i, x in enumerate(automaton['inputs'])}
    output_to_idx = {x: i for i, x in enumerate(automaton['outputs'])}

    model = FuzzyMealyMachine(
        num_states=len(automaton['states']),
        num_inputs=len(automaton['inputs']),
        num_outputs=len(automaton['outputs'])
    )

    # Train
    print("\nTraining...")
    print("-" * 40)
    model = train_on_traces(model, traces, input_to_idx, output_to_idx, epochs=500)

    # Evaluate
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    # Test accuracy on new traces
    test_traces = generate_traces(automaton, num_traces=100, max_length=20)
    correct = 0
    total = 0

    for input_seq, output_seq in test_traces:
        inp_indices = [input_to_idx[x] for x in input_seq]
        out_indices = [output_to_idx[x] for x in output_seq]

        with torch.no_grad():
            out_probs = model(inp_indices)
            preds = out_probs.argmax(dim=-1).tolist()

        if preds == out_indices:
            correct += 1
        total += 1

    print(f"\nTest accuracy (full sequence match): {correct}/{total} = {correct/total:.2%}")

    # Compare structures
    compare_with_ground_truth(model, automaton, input_to_idx, output_to_idx)


if __name__ == "__main__":
    main()
