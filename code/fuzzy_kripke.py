"""
Fuzzy Kripke Learning - 最简原型

目标：从观测数据学习一个 Kripke 结构

Kripke Model M = (W, R, V)
- W: worlds (states)
- R: accessibility relation (W × W → {0,1})
- V: valuation (W × Props → {0,1})

Fuzzy 版本：
- R: W × W → [0,1]  (soft accessibility)
- V: W × Props → [0,1]  (soft truth)

学习：
1. 初始化 R, V 为随机 [0,1] 值
2. 定义 loss（满足某些约束）
3. Gradient descent
4. 收敛后 R, V → {0,1}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FuzzyKripkeModel(nn.Module):
    """
    Fuzzy Kripke Model with learnable R and V
    """
    def __init__(self, num_worlds, num_props):
        super().__init__()
        self.num_worlds = num_worlds
        self.num_props = num_props

        # Accessibility relation R: W × W → [0,1]
        # 用 sigmoid 保证在 [0,1]
        self._R = nn.Parameter(torch.randn(num_worlds, num_worlds))

        # Valuation V: W × Props → [0,1]
        self._V = nn.Parameter(torch.randn(num_worlds, num_props))

    @property
    def R(self):
        """Accessibility relation in [0,1]"""
        return torch.sigmoid(self._R)

    @property
    def V(self):
        """Valuation in [0,1]"""
        return torch.sigmoid(self._V)

    def eval_prop(self, p, w):
        """Evaluate proposition p at world w"""
        return self.V[w, p]

    def eval_box(self, phi_values):
        """
        □φ at world w = min over accessible worlds

        Fuzzy version: □φ(w) = min_v { R(w,v) → φ(v) }
        where a → b = min(1, 1-a+b) (Łukasiewicz implication)

        Soft approximation: use weighted min
        """
        R = self.R  # (W, W)
        # φ(v) for all v
        phi = phi_values  # (W,)

        # R(w,v) → φ(v) = min(1, 1 - R(w,v) + φ(v))
        # 用 soft min: weighted by R
        implications = torch.clamp(1 - R + phi.unsqueeze(0), 0, 1)  # (W, W)

        # Soft min: use negative softmax trick
        # min ≈ -logsumexp(-x)
        temperature = 10.0
        soft_min = -torch.logsumexp(-temperature * implications, dim=1) / temperature

        return soft_min  # (W,)

    def eval_diamond(self, phi_values):
        """
        ◇φ at world w = max over accessible worlds

        Fuzzy version: ◇φ(w) = max_v { R(w,v) ⊗ φ(v) }
        where ⊗ is t-norm (product)
        """
        R = self.R  # (W, W)
        phi = phi_values  # (W,)

        # R(w,v) ⊗ φ(v) = R(w,v) * φ(v)
        products = R * phi.unsqueeze(0)  # (W, W)

        # Soft max
        temperature = 10.0
        soft_max = torch.logsumexp(temperature * products, dim=1) / temperature

        return soft_max  # (W,)

    def sparsity_loss(self):
        """
        Push R and V toward {0, 1}

        Loss = Σ r(1-r) + Σ v(1-v)
        Minimized when all values are 0 or 1
        """
        R = self.R
        V = self.V
        return (R * (1 - R)).sum() + (V * (1 - V)).sum()

    def discretize(self):
        """Round to {0, 1}"""
        return (self.R > 0.5).float(), (self.V > 0.5).float()


def demo_learn_simple():
    """
    Demo: 学习一个简单的 Kripke 结构

    Target structure (3 worlds, 2 props):

        w0 ---> w1 ---> w2
        p       q       p,q

    R = [[0,1,0],
         [0,0,1],
         [0,0,0]]

    V = [[1,0],   # w0: p
         [0,1],   # w1: q
         [1,1]]   # w2: p,q

    约束（用来学习）:
    - ◇p 在 w0 应该为 true（因为 w0→w1→w2，w2 有 p）
    - □q 在 w1 应该为 true（因为 w1 只能到 w2，w2 有 q）
    """
    print("=" * 50)
    print("Fuzzy Kripke Learning Demo")
    print("=" * 50)

    # Target
    target_R = torch.tensor([
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 0.]
    ])
    target_V = torch.tensor([
        [1., 0.],
        [0., 1.],
        [1., 1.]
    ])

    print("\nTarget R (accessibility):")
    print(target_R.numpy())
    print("\nTarget V (valuation):")
    print(target_V.numpy())

    # Create fuzzy model
    model = FuzzyKripkeModel(num_worlds=3, num_props=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Training constraints:
    # We'll use MSE to the target as supervision
    # In real scenario, we'd use modal formula satisfaction

    print("\n" + "-" * 50)
    print("Training...")
    print("-" * 50)

    for epoch in range(500):
        optimizer.zero_grad()

        # Loss 1: match target structure
        loss_R = F.mse_loss(model.R, target_R)
        loss_V = F.mse_loss(model.V, target_V)

        # Loss 2: sparsity (push to 0/1)
        loss_sparse = model.sparsity_loss() * 0.1

        loss = loss_R + loss_V + loss_sparse

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f} "
                  f"(R={loss_R.item():.4f}, V={loss_V.item():.4f}, sparse={loss_sparse.item():.4f})")

    # Results
    print("\n" + "-" * 50)
    print("Learned (fuzzy):")
    print("-" * 50)
    print(f"R:\n{model.R.detach().numpy().round(2)}")
    print(f"V:\n{model.V.detach().numpy().round(2)}")

    # Discretize
    R_discrete, V_discrete = model.discretize()
    print("\n" + "-" * 50)
    print("Discretized (crisp):")
    print("-" * 50)
    print(f"R:\n{R_discrete.numpy()}")
    print(f"V:\n{V_discrete.numpy()}")

    # Check correctness
    print("\n" + "-" * 50)
    print("Verification:")
    print("-" * 50)
    print(f"R matches target: {torch.equal(R_discrete, target_R)}")
    print(f"V matches target: {torch.equal(V_discrete, target_V)}")

    # Test modal formulas
    print("\n" + "-" * 50)
    print("Modal formula evaluation:")
    print("-" * 50)

    # p values at each world
    p_values = model.V[:, 0]
    q_values = model.V[:, 1]

    # ◇p at each world
    diamond_p = model.eval_diamond(p_values)
    print(f"◇p at each world: {diamond_p.detach().numpy().round(2)}")

    # □q at each world
    box_q = model.eval_box(q_values)
    print(f"□q at each world: {box_q.detach().numpy().round(2)}")


def demo_learn_from_formulas():
    """
    Demo: 只从 modal formula constraints 学习结构

    不给 target R, V，只给约束：
    - w0 ⊨ ◇◇p  (从 w0 可以两步到达 p)
    - w2 ⊨ p ∧ q
    - w0 ⊭ q
    """
    print("\n" + "=" * 50)
    print("Learning from Modal Formulas Only")
    print("=" * 50)

    model = FuzzyKripkeModel(num_worlds=3, num_props=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(1000):
        optimizer.zero_grad()

        p_values = model.V[:, 0]  # p
        q_values = model.V[:, 1]  # q

        # Constraint 1: w2 ⊨ p ∧ q
        loss1 = (1 - model.V[2, 0]) ** 2 + (1 - model.V[2, 1]) ** 2

        # Constraint 2: w0 ⊭ q (w0 doesn't have q)
        loss2 = model.V[0, 1] ** 2

        # Constraint 3: w0 ⊨ ◇◇p
        diamond_p = model.eval_diamond(p_values)
        diamond_diamond_p = model.eval_diamond(diamond_p)
        loss3 = (1 - diamond_diamond_p[0]) ** 2

        # Constraint 4: encourage some structure (not all zeros)
        loss4 = -model.R.sum() * 0.01  # encourage edges

        # Sparsity
        loss_sparse = model.sparsity_loss() * 0.05

        loss = loss1 + loss2 + loss3 + loss4 + loss_sparse

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f}")

    # Results
    R_discrete, V_discrete = model.discretize()

    print("\n" + "-" * 50)
    print("Learned structure:")
    print("-" * 50)
    print(f"R (fuzzy):\n{model.R.detach().numpy().round(2)}")
    print(f"V (fuzzy):\n{model.V.detach().numpy().round(2)}")
    print(f"\nR (crisp):\n{R_discrete.numpy()}")
    print(f"V (crisp):\n{V_discrete.numpy()}")

    # Verify constraints
    print("\n" + "-" * 50)
    print("Constraint verification:")
    print("-" * 50)
    print(f"w2 ⊨ p: {V_discrete[2, 0].item()}")
    print(f"w2 ⊨ q: {V_discrete[2, 1].item()}")
    print(f"w0 ⊭ q: {1 - V_discrete[0, 1].item()}")


if __name__ == "__main__":
    demo_learn_simple()
    demo_learn_from_formulas()
