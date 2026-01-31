"""
ARC Prototype: Semiring-valued Coalgebra Learning

Core idea:
- ARC task = learn a transformation rule (program)
- Program = coalgebra (state machine)
- Learn via gradient semiring (differentiable)
- Converge to crisp (discrete) rule

This is a minimal prototype to test the concept.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


# =============================================================================
# Gradient Semiring (from DeepProbLog)
# =============================================================================

class GradientSemiring:
    """
    Element: (value, gradient)
    ⊗: (v1,g1) ⊗ (v2,g2) = (v1*v2, v1*g2 + v2*g1)  # product rule
    ⊕: (v1,g1) ⊕ (v2,g2) = (v1+v2, g1+g2)          # sum rule

    In PyTorch, autograd handles this automatically.
    This class is for conceptual clarity.
    """
    def __init__(self, value, gradient=None):
        self.value = value
        self.gradient = gradient if gradient is not None else torch.zeros_like(value)

    def __mul__(self, other):
        # Product rule
        new_value = self.value * other.value
        new_grad = self.value * other.gradient + other.value * self.gradient
        return GradientSemiring(new_value, new_grad)

    def __add__(self, other):
        # Sum rule
        return GradientSemiring(
            self.value + other.value,
            self.gradient + other.gradient
        )


# =============================================================================
# Fuzzy/Soft Operations (for differentiable logic)
# =============================================================================

def soft_eq(a, b, temperature=1.0):
    """Soft equality: 1 if a==b, 0 otherwise (differentiable)"""
    return torch.exp(-temperature * (a - b) ** 2)

def soft_and(a, b):
    """Product t-norm"""
    return a * b

def soft_or(a, b):
    """Probabilistic t-conorm"""
    return a + b - a * b

def soft_not(a):
    """Standard negation"""
    return 1 - a


# =============================================================================
# Simple Rule Learner for ARC
# =============================================================================

class ARCRuleLearner(nn.Module):
    """
    Learn a transformation rule for ARC tasks.

    Architecture (simple version):
    - Embed each cell (color) into a vector
    - Apply learned local rules (convolution-like)
    - Output new grid

    This is a minimal prototype - real ARC needs much more.
    """

    def __init__(self, num_colors=10, hidden_dim=32, kernel_size=3):
        super().__init__()
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim

        # Color embedding
        self.color_embed = nn.Embedding(num_colors, hidden_dim)

        # Local rule (like a convolution but learnable)
        self.rule = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.ReLU(),
        )

        # Output color prediction
        self.output = nn.Conv2d(hidden_dim, num_colors, 1)

        # Sparsity parameter (to push toward discrete)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, grid):
        """
        grid: (H, W) tensor of color indices
        returns: (H, W, num_colors) soft predictions
        """
        H, W = grid.shape

        # Embed colors: (H, W) -> (H, W, hidden_dim)
        x = self.color_embed(grid)

        # Reshape for conv: (1, hidden_dim, H, W)
        x = x.permute(2, 0, 1).unsqueeze(0)

        # Apply rule
        x = self.rule(x)

        # Output logits: (1, num_colors, H, W)
        logits = self.output(x)

        # Soft predictions with temperature
        probs = F.softmax(logits * self.temperature, dim=1)

        # (H, W, num_colors)
        return probs.squeeze(0).permute(1, 2, 0)

    def predict(self, grid):
        """Hard prediction (argmax)"""
        with torch.no_grad():
            probs = self.forward(grid)
            return probs.argmax(dim=-1)


# =============================================================================
# Training
# =============================================================================

def load_arc_task(task_path):
    """Load an ARC task from JSON"""
    with open(task_path) as f:
        task = json.load(f)
    return task

def grid_to_tensor(grid):
    """Convert list of lists to tensor"""
    return torch.tensor(grid, dtype=torch.long)

def train_on_task(model, task, epochs=1000, lr=0.01):
    """
    Train model on a single ARC task.

    Loss = cross-entropy between predicted and target output
         + sparsity term (push toward discrete)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_examples = task['train']

    for epoch in range(epochs):
        total_loss = 0

        for example in train_examples:
            input_grid = grid_to_tensor(example['input'])
            target_grid = grid_to_tensor(example['output'])

            # Handle size mismatch (ARC often changes grid size)
            # For now, skip if sizes don't match
            if input_grid.shape != target_grid.shape:
                # This is a limitation - real ARC needs to handle size changes
                continue

            # Forward
            probs = model(input_grid)  # (H, W, num_colors)

            # Cross-entropy loss
            H, W = target_grid.shape
            probs_flat = probs.view(-1, model.num_colors)
            target_flat = target_grid.view(-1)
            loss = F.cross_entropy(probs_flat, target_flat)

            # Sparsity: encourage predictions to be near 0 or 1
            # entropy of predictions (lower = more discrete)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
            sparsity_loss = entropy * 0.1

            total_loss = loss + sparsity_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {total_loss.item():.4f}")

    return model


def evaluate_on_task(model, task):
    """Evaluate model on test examples"""
    test_examples = task['test']

    correct = 0
    total = 0

    for example in test_examples:
        input_grid = grid_to_tensor(example['input'])
        target_grid = grid_to_tensor(example['output'])

        if input_grid.shape != target_grid.shape:
            continue

        pred_grid = model.predict(input_grid)

        if torch.equal(pred_grid, target_grid):
            correct += 1
        total += 1

        print(f"Input:\n{input_grid.numpy()}")
        print(f"Target:\n{target_grid.numpy()}")
        print(f"Predicted:\n{pred_grid.numpy()}")
        print(f"Match: {torch.equal(pred_grid, target_grid)}")
        print()

    return correct, total


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Find ARC data
    arc_path = Path(__file__).parent.parent / "datasets" / "ARC-AGI" / "data" / "training"

    if not arc_path.exists():
        print(f"ARC data not found at {arc_path}")
        exit(1)

    # Load a simple task (one where input/output sizes match)
    # Let's try a few and find one that works
    task_files = list(arc_path.glob("*.json"))

    print(f"Found {len(task_files)} tasks")
    print("Looking for a task with matching input/output sizes...")

    for task_file in task_files[:20]:  # Try first 20
        task = load_arc_task(task_file)

        # Check if all examples have matching sizes
        all_match = True
        for ex in task['train'] + task['test']:
            inp = ex['input']
            out = ex['output']
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                all_match = False
                break

        if all_match:
            print(f"\nFound matching task: {task_file.name}")
            print(f"Train examples: {len(task['train'])}")
            print(f"Test examples: {len(task['test'])}")

            # Show first example
            ex = task['train'][0]
            print(f"Example input shape: {len(ex['input'])}x{len(ex['input'][0])}")
            print(f"Example output shape: {len(ex['output'])}x{len(ex['output'][0])}")

            # Train
            print("\nTraining...")
            model = ARCRuleLearner()
            model = train_on_task(model, task, epochs=500)

            # Evaluate
            print("\nEvaluating on test:")
            correct, total = evaluate_on_task(model, task)
            print(f"Accuracy: {correct}/{total}")

            break
    else:
        print("No matching task found in first 20 files")
