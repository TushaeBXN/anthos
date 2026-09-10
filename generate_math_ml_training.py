#!/usr/bin/env python3
"""
generate_math_ml_training.py — Math for ML training data for Anthos

Source: The Palindrome "Roadmap of Mathematics for Machine Learning"
Covers: Linear Algebra, Calculus, Multivariable Calculus, Probability Theory

Two modes per topic:
  1. Explain it  — conceptual understanding, intuition, real meaning
  2. Build it    — complete working Python implementations

Output: data/math_ml_sft.jsonl
"""

import json
import random
from pathlib import Path

OUTPUT = Path("data/math_ml_sft.jsonl")

ANTHOS_MATH_SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You explain mathematics for machine learning clearly — with real intuition, "
    "not just formulas. You connect concepts to how they actually work in neural networks, "
    "optimization, and data. When asked to implement something, you write complete, "
    "runnable Python code with numpy. No pseudocode. No outlines. Working implementations."
)

def conv(q, a):
    return {
        "conversations": [
            {"from": "system",  "value": ANTHOS_MATH_SYSTEM},
            {"from": "human",   "value": q.strip()},
            {"from": "gpt",     "value": a.strip()},
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# LINEAR ALGEBRA — Explain it
# ─────────────────────────────────────────────────────────────────────────────

LA_EXPLAIN = [
    (
        "What is a vector space and why does it matter for machine learning?",
        """A vector space is a set of vectors where you can add them together and scale them by numbers, and the result stays in the set.

Formally: V is a vector space over ℝ if:
- v + w ∈ V for any v, w ∈ V  (closed under addition)
- α·v ∈ V for any v ∈ V, α ∈ ℝ  (closed under scalar multiplication)

Why it matters for ML:
- Every data point in a dataset lives in a vector space (ℝⁿ for n features)
- Neural network layers are linear transformations between vector spaces
- The "embedding space" of an LLM is a high-dimensional vector space — words become vectors you can do algebra on

The critical insight: when you train a model, you're finding a path through a vector space (the parameter space) that minimizes loss. All of optimization is geometry in a vector space."""
    ),
    (
        "Explain norms and distance — what are they actually measuring?",
        """A norm is a function that assigns a length to a vector. The Euclidean norm (L2) is what you picture as "distance":

‖v‖₂ = √(v₁² + v₂² + ... + vₙ²)

But there are others:
- L1 norm: ‖v‖₁ = |v₁| + |v₂| + ... + |vₙ|  (Manhattan distance)
- L∞ norm: max(|v₁|, |v₂|, ..., |vₙ|)  (largest component)

In ML these show up constantly:
- **L2 regularization** (weight decay): adds ‖w‖₂² to the loss — penalizes large weights
- **L1 regularization** (LASSO): adds ‖w‖₁ — drives weights to exactly zero (sparsity)
- **Gradient norm clipping**: if ‖∇‖₂ > threshold, scale it down — prevents exploding gradients

The choice of norm encodes a belief about what "far away" means. L1 treats all directions equally. L2 cares most about the largest component."""
    ),
    (
        "What is an inner product and how does dot product measure similarity?",
        """The inner product generalizes the dot product. For standard Euclidean space:

⟨u, v⟩ = u₁v₁ + u₂v₂ + ... + uₙvₙ = ‖u‖‖v‖cos(θ)

Where θ is the angle between the vectors.

This gives you three things at once:
1. **Same direction** (θ=0): ⟨u,v⟩ = ‖u‖‖v‖ (maximum, positive)
2. **Opposite direction** (θ=π): ⟨u,v⟩ = -‖u‖‖v‖ (maximum negative)
3. **Perpendicular** (θ=π/2): ⟨u,v⟩ = 0 (orthogonal, no similarity)

In ML this is everywhere:
- **Attention mechanisms**: Q·Kᵀ computes dot products between query and key vectors — similarity decides how much to attend
- **Cosine similarity**: ⟨u,v⟩ / (‖u‖‖v‖) — normalized dot product, measures direction not magnitude
- **Neural network forward pass**: each neuron computes a dot product (wᵀx + b) then applies nonlinearity"""
    ),
    (
        "What are linear transformations and how do matrices represent them?",
        """A linear transformation T: V → W maps vectors from one space to another while preserving structure:

- T(u + v) = T(u) + T(v)  (preserves addition)
- T(αv) = αT(v)  (preserves scaling)

Every linear transformation between finite-dimensional spaces can be represented as a matrix. If you apply T to the standard basis vectors and stack the results as columns, you get the matrix A.

Then: T(x) = Ax

This means:
- **Neural network layers** are linear transformations (weight matrix times input)
- **Matrix multiplication** is composing two linear transformations — (BA)x = B(Ax)
- **Stacking layers** in a neural net is composing transformations
- The final matrix product of all weight matrices represents the full linear portion of your network

The nonlinear activation functions (ReLU, sigmoid) break pure linearity — without them, any deep network collapses to a single matrix multiplication."""
    ),
    (
        "What are eigenvalues and eigenvectors in plain terms?",
        """An eigenvector is a special direction in space that a matrix transformation doesn't rotate — it only stretches or shrinks.

If A is a matrix:
  Av = λv

v is the eigenvector, λ (lambda) is the eigenvalue — the stretch factor.

Intuition: most vectors, when multiplied by A, point in a completely new direction. Eigenvectors are the exceptions. They're the "natural axes" of the transformation.

In ML:
- **PCA**: eigenvectors of the covariance matrix are the principal components — directions of maximum variance. Eigenvalues tell you how much variance each direction captures.
- **Optimization**: the Hessian's eigenvalues tell you the curvature of the loss surface. Large eigenvalue = steep direction. Near-zero eigenvalue = flat direction (saddle point risk).
- **Stable training**: if a weight matrix has eigenvalues > 1, gradients can explode through it. If < 1, they vanish. This is why RNNs are hard to train.
- **Graph neural networks**: graph Laplacian eigenvalues define the frequency structure of signals on the graph."""
    ),
    (
        "Explain SVD — what is it actually doing?",
        """SVD (Singular Value Decomposition) breaks any matrix into three pieces:

A = U Σ Vᵀ

Where:
- U: orthonormal matrix — "output directions" (left singular vectors)
- Σ: diagonal matrix of singular values (σ₁ ≥ σ₂ ≥ ... ≥ 0)
- Vᵀ: orthonormal matrix — "input directions" (right singular vectors)

What it's doing geometrically: any linear transformation can be decomposed into (1) a rotation/reflection, (2) a scaling along axes, (3) another rotation/reflection.

In ML:
- **PCA**: the principal components are the right singular vectors (columns of V). Singular values squared ∝ variance explained.
- **Low-rank approximation**: keep only the top-k singular values/vectors → Aₖ = UₖΣₖVₖᵀ. This is how image compression works, and how LoRA fine-tunes LLMs (approximates weight updates with low-rank matrices).
- **Recommendation systems**: SVD factorizes the user-item matrix to find latent features.
- **Stable training**: the largest singular value of a weight matrix bounds how much it can amplify signals."""
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# LINEAR ALGEBRA — Build it
# ─────────────────────────────────────────────────────────────────────────────

LA_BUILD = [
    (
        "Implement vector norms (L1, L2, L-infinity) from scratch in numpy and show the difference.",
        """import numpy as np

def l1_norm(v):
    return np.sum(np.abs(v))

def l2_norm(v):
    return np.sqrt(np.sum(v ** 2))

def linf_norm(v):
    return np.max(np.abs(v))

# Test
v = np.array([3.0, -4.0, 0.0, 2.0])

print(f"Vector: {v}")
print(f"L1 norm:  {l1_norm(v):.4f}")   # 3+4+0+2 = 9
print(f"L2 norm:  {l2_norm(v):.4f}")   # sqrt(9+16+0+4) = sqrt(29) ≈ 5.385
print(f"L∞ norm:  {linf_norm(v):.4f}") # max absolute = 4

# Verify against numpy
assert np.isclose(l1_norm(v), np.linalg.norm(v, 1))
assert np.isclose(l2_norm(v), np.linalg.norm(v, 2))
assert np.isclose(linf_norm(v), np.linalg.norm(v, np.inf))

print("\\nAll match numpy. ✓")

# ML use: compute cosine similarity
def cosine_similarity(u, v):
    return np.dot(u, v) / (l2_norm(u) * l2_norm(v))

a = np.array([1.0, 0.0, 1.0])
b = np.array([1.0, 0.0, 0.5])
c = np.array([-1.0, 0.0, -1.0])

print(f"\\nSimilarity(a, b): {cosine_similarity(a, b):.4f}")  # close, ~0.949
print(f"Similarity(a, c): {cosine_similarity(a, c):.4f}")  # opposite, -1.0"""
    ),
    (
        "Implement PCA from scratch using SVD. Apply it to reduce 4D data to 2D.",
        """import numpy as np

def pca(X, n_components):
    # Center the data
    X_centered = X - X.mean(axis=0)

    # SVD of the centered matrix
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Principal components = rows of Vt (right singular vectors)
    components = Vt[:n_components]

    # Project data onto principal components
    X_reduced = X_centered @ components.T

    # Variance explained by each component
    variance_explained = (S ** 2) / np.sum(S ** 2)

    return X_reduced, components, variance_explained[:n_components]


# Generate 4D data with correlation structure
np.random.seed(42)
n = 200
t = np.linspace(0, 2 * np.pi, n)

# 4 features, but really 2 underlying signals
X = np.column_stack([
    np.sin(t) + 0.1 * np.random.randn(n),
    np.cos(t) + 0.1 * np.random.randn(n),
    0.5 * np.sin(t) + 0.2 * np.random.randn(n),
    0.5 * np.cos(t) + 0.2 * np.random.randn(n),
])

print(f"Original shape: {X.shape}")  # (200, 4)

X_2d, components, var_explained = pca(X, n_components=2)

print(f"Reduced shape:  {X_2d.shape}")  # (200, 2)
print(f"\\nVariance explained:")
for i, v in enumerate(var_explained):
    print(f"  PC{i+1}: {v:.1%}")
print(f"  Total: {sum(var_explained):.1%}")"""
    ),
    (
        "Implement matrix multiplication and explain how it's used in a neural network forward pass.",
        """import numpy as np

def matmul(A, B):
    \"\"\"Manual matrix multiplication. A: (m,n), B: (n,p) → (m,p)\"\"\"
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, f"Shape mismatch: {A.shape} @ {B.shape}"
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

# Verify
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C_manual = matmul(A, B)
C_numpy  = A @ B
assert np.allclose(C_manual, C_numpy), "Mismatch"
print("Manual matmul matches numpy. ✓")

# Neural network forward pass
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    # Layer 1: linear + relu
    Z1 = X @ W1 + b1        # (batch, hidden)
    A1 = relu(Z1)
    # Layer 2: linear + softmax
    Z2 = A1 @ W2 + b2        # (batch, n_classes)
    A2 = softmax(Z2)
    return A2

# Example: 8 samples, 10 features → 16 hidden → 3 classes
batch, features, hidden, classes = 8, 10, 16, 3
X  = np.random.randn(batch, features)
W1 = np.random.randn(features, hidden) * 0.1
b1 = np.zeros(hidden)
W2 = np.random.randn(hidden, classes) * 0.1
b2 = np.zeros(classes)

probs = forward(X, W1, b1, W2, b2)
print(f"\\nInput:  {X.shape}")
print(f"Output: {probs.shape}")
print(f"Row sums (should be 1.0): {probs.sum(axis=1).round(6)}")"""
    ),
    (
        "Compute eigenvalues and eigenvectors from scratch and use them for PCA the long way.",
        """import numpy as np

# Use numpy's eig for correctness; show what the results mean
np.random.seed(0)

# Covariance matrix of 2D data
X = np.random.randn(100, 2)
X[:, 1] = 0.8 * X[:, 0] + 0.6 * np.random.randn(100)  # correlated

# Center and compute covariance
X_c = X - X.mean(axis=0)
C = (X_c.T @ X_c) / (len(X) - 1)  # (2, 2) covariance matrix

print(f"Covariance matrix:\\n{C.round(4)}")

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(C)

# Sort by eigenvalue descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"\\nEigenvalues: {eigenvalues.round(4)}")
print(f"Eigenvectors (columns):\\n{eigenvectors.round(4)}")

# Verify: C @ v = λv
for i in range(2):
    lhs = C @ eigenvectors[:, i]
    rhs = eigenvalues[i] * eigenvectors[:, i]
    assert np.allclose(lhs, rhs), "Eigenvector check failed"
print("\\nEigenvector check passed. ✓")

# Project onto first principal component
pc1 = eigenvectors[:, 0]
X_1d = X_c @ pc1

print(f"\\nVariance explained by PC1: {eigenvalues[0] / eigenvalues.sum():.1%}")
print(f"Original shape: {X.shape} → Reduced shape: {X_1d.shape}")"""
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# CALCULUS — Explain it
# ─────────────────────────────────────────────────────────────────────────────

CALC_EXPLAIN = [
    (
        "What is a derivative intuitively? How does backpropagation use it?",
        """A derivative measures the instantaneous rate of change. At any point x on a function f, the derivative f'(x) tells you: "if I move x a tiny bit, how much does f(x) move?"

Formally: f'(x) = lim(h→0) [f(x+h) - f(x)] / h

Intuition: the slope of the tangent line at x.

Why backprop works:
Backpropagation is just the chain rule applied recursively through a neural network.

The loss L is a composition of functions:
L = loss(softmax(linear(relu(linear(x)))))

The chain rule says: dL/dx = dL/da · da/db · db/dc · dc/dx

Each layer contributes a factor. Backprop computes these factors in reverse order, which is efficient because you can reuse intermediate results (stored during the forward pass).

The key insight: you don't need to know how the loss changes with respect to x directly — you just need to know how each layer's output changes with its input, then multiply them all together. That's backpropagation."""
    ),
    (
        "Explain the gradient and gradient descent — what is the gradient telling you?",
        """For a function of multiple variables f(x₁, x₂, ..., xₙ), the gradient is the vector of all partial derivatives:

∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

What it tells you: the direction of steepest ascent. If you stand at a point on a hilly landscape, the gradient points uphill — toward where f increases fastest.

**Gradient descent** goes the opposite direction:
  θ ← θ - α·∇L(θ)

Where α is the learning rate.

Three-step loop:
1. Compute gradient of loss at current weights
2. Step in the negative gradient direction (downhill)
3. Repeat until convergence

Why it works: the loss function is a landscape over all possible weight configurations. Gradient descent is rolling downhill on that landscape. You're guaranteed to reach a local minimum if the learning rate is small enough.

Why it's tricky:
- Local minima (not the global minimum)
- Saddle points (gradient is zero but not a minimum)
- Learning rate too large → oscillates; too small → takes forever
- Modern variants (Adam, AdaGrad) adapt the learning rate per parameter"""
    ),
    (
        "What is the chain rule and why is it the foundation of neural network training?",
        """The chain rule tells you how to differentiate composed functions.

If y = f(g(x)), then:
  dy/dx = (dy/du) · (du/dx)   where u = g(x)

For multiple layers (a neural network with 3 layers):
  L = l(a(b(c(x))))
  dL/dx = dL/da · da/db · db/dc · dc/dx

This is just a chain of multiplication. Each factor is a local gradient.

Why neural networks can be trained at all: without the chain rule, computing how the loss changes with the first layer's weights would require tracking every path from those weights to the output. The chain rule lets you compute it as a product of local, simple derivatives — each layer only needs to know its own gradient.

In code (PyTorch does this automatically with autograd):
- Forward pass: compute and store each layer's output
- Backward pass: starting from the loss, multiply gradients layer by layer in reverse

The stored intermediate values from the forward pass are used in the backward pass — this is why you keep them (gradient checkpointing trades memory for recomputation)."""
    ),
    (
        "What is the Hessian matrix and when does it matter in ML?",
        """The Hessian is the matrix of all second-order partial derivatives of a function:

H[i,j] = ∂²f / (∂xᵢ ∂xⱼ)

For a loss function L(θ) with n parameters, H is n×n.

What it tells you:
- **Curvature**: how the gradient itself is changing. High curvature → steep, curved bowl. Low curvature → flat plateau.
- **Critical points**: if ∇L = 0, look at H to classify:
  - All eigenvalues positive → local minimum
  - All eigenvalues negative → local maximum
  - Mixed signs → saddle point

Why it matters in ML:
1. **Second-order optimizers** (Newton's method, L-BFGS): use H⁻¹ to take a better step — accounts for curvature, converges faster, but computing H for millions of parameters is prohibitive
2. **Saddle points**: deep networks have exponentially many saddle points. The Hessian's eigenvalue spectrum reveals how flat/curved the loss surface is
3. **Learning rate tuning**: the largest eigenvalue of H bounds the maximum stable learning rate (Lipschitz constant)
4. **Sharpness-Aware Minimization (SAM)**: seeks flat minima (small Hessian eigenvalues) for better generalization"""
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# CALCULUS — Build it
# ─────────────────────────────────────────────────────────────────────────────

CALC_BUILD = [
    (
        "Implement gradient descent from scratch to minimize a simple quadratic function.",
        """import numpy as np

def f(x):
    \"\"\"f(x, y) = (x-3)² + (y+2)² — minimum at (3, -2)\"\"\"
    return (x[0] - 3)**2 + (x[1] + 2)**2

def grad_f(x):
    \"\"\"Analytical gradient: [2(x-3), 2(y+2)]\"\"\"
    return np.array([2*(x[0] - 3), 2*(x[1] + 2)])

def gradient_descent(f, grad_f, x0, lr=0.1, max_iter=100, tol=1e-6):
    x = x0.copy().astype(float)
    history = [x.copy()]

    for i in range(max_iter):
        g = grad_f(x)
        x = x - lr * g
        history.append(x.copy())

        if np.linalg.norm(g) < tol:
            print(f"Converged at iteration {i+1}")
            break

    return x, np.array(history)

x0 = np.array([0.0, 0.0])
x_min, history = gradient_descent(f, grad_f, x0, lr=0.2)

print(f"Starting point: {history[0]}")
print(f"Found minimum:  {x_min.round(6)}")
print(f"True minimum:   [3.0, -2.0]")
print(f"Loss at min:    {f(x_min):.8f}")
print(f"Iterations:     {len(history)-1}")"""
    ),
    (
        "Implement numerical gradient checking — verify your analytical gradient is correct.",
        """import numpy as np

def numerical_gradient(f, x, eps=1e-5):
    \"\"\"Estimate gradient via finite differences.\"\"\"
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus  = x.copy().astype(float); x_plus[i]  += eps
        x_minus = x.copy().astype(float); x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

def check_gradient(f, analytical_grad_fn, x, tol=1e-5):
    ag = analytical_grad_fn(x)
    ng = numerical_gradient(f, x)
    diff = np.linalg.norm(ag - ng) / (np.linalg.norm(ag) + np.linalg.norm(ng) + 1e-8)
    ok = diff < tol
    print(f"Analytical: {ag}")
    print(f"Numerical:  {ng.round(8)}")
    print(f"Relative diff: {diff:.2e}  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok

# Example: f(x) = sin(x0) * x1^2 + x2^3
def f(x):
    return np.sin(x[0]) * x[1]**2 + x[2]**3

def grad_f(x):
    return np.array([
        np.cos(x[0]) * x[1]**2,   # df/dx0
        2 * np.sin(x[0]) * x[1],  # df/dx1
        3 * x[2]**2,               # df/dx2
    ])

x = np.array([1.2, -0.5, 2.1])
check_gradient(f, grad_f, x)

# Now test with a neural network layer
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy(y_pred, y_true):
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

# One-layer network loss as a function of weights
np.random.seed(1)
X = np.random.randn(5, 3)
y = np.array([1., 0., 1., 0., 1.])

def loss_fn(w):
    return cross_entropy(sigmoid(X @ w), y)

def grad_loss(w):
    p = sigmoid(X @ w)
    return X.T @ (p - y) / len(y)

w = np.random.randn(3)
print("\\nNeural net layer gradient check:")
check_gradient(loss_fn, grad_loss, w)"""
    ),
    (
        "Implement SGD, SGD with momentum, and Adam optimizer from scratch.",
        """import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        return {k: params[k] - self.lr * grads[k] for k in params}


class MomentumSGD:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def step(self, params, grads):
        if not self.velocity:
            self.velocity = {k: np.zeros_like(v) for k, v in params.items()}
        updated = {}
        for k in params:
            self.velocity[k] = self.momentum * self.velocity[k] - self.lr * grads[k]
            updated[k] = params[k] + self.velocity[k]
        return updated


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # first moment
        self.v = {}  # second moment
        self.t = 0

    def step(self, params, grads):
        if not self.m:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t += 1
        updated = {}
        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grads[k]**2
            m_hat = self.m[k] / (1 - self.beta1**self.t)  # bias correction
            v_hat = self.v[k] / (1 - self.beta2**self.t)
            updated[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return updated


# Compare on Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
def rosenbrock(p):
    x, y = p['x'], p['y']
    return (1 - x)**2 + 100*(y - x**2)**2

def rosenbrock_grad(p):
    x, y = p['x'], p['y']
    return {
        'x': -2*(1 - x) - 400*x*(y - x**2),
        'y': 200*(y - x**2),
    }

def run(optimizer, n_steps=2000):
    p = {'x': np.float64(-1.0), 'y': np.float64(1.0)}
    for _ in range(n_steps):
        g = rosenbrock_grad(p)
        p = optimizer.step(p, g)
    loss = rosenbrock(p)
    return p, loss

p_sgd,  l_sgd  = run(SGD(lr=0.001))
p_mom,  l_mom  = run(MomentumSGD(lr=0.001))
p_adam, l_adam = run(Adam(lr=0.01))

print("Rosenbrock minimum at (1, 1):")
print(f"SGD:      x={p_sgd['x']:.4f}, y={p_sgd['y']:.4f}, loss={l_sgd:.6f}")
print(f"Momentum: x={p_mom['x']:.4f}, y={p_mom['y']:.4f}, loss={l_mom:.6f}")
print(f"Adam:     x={p_adam['x']:.4f}, y={p_adam['y']:.4f}, loss={l_adam:.6f}")"""
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY THEORY — Explain it
# ─────────────────────────────────────────────────────────────────────────────

PROB_EXPLAIN = [
    (
        "Explain Bayes' Theorem and how it connects to machine learning inference.",
        """Bayes' theorem relates conditional probabilities:

P(A|B) = P(B|A) · P(A) / P(B)

In ML terms:
- A = "model parameters θ"
- B = "observed data X"

P(θ|X) = P(X|θ) · P(θ) / P(X)

- **P(θ|X)**: posterior — what we believe about θ after seeing data
- **P(X|θ)**: likelihood — how probable is the data given these parameters
- **P(θ)**: prior — what we believed before seeing data
- **P(X)**: evidence — normalizing constant (often intractable)

Maximum Likelihood Estimation (MLE): maximize P(X|θ) — find parameters that make the data most probable. No prior.

Maximum A Posteriori (MAP): maximize P(θ|X) ∝ P(X|θ)·P(θ) — adds a prior. L2 regularization is MAP with a Gaussian prior on weights. L1 is MAP with a Laplace prior.

Bayesian inference: maintain a full posterior distribution P(θ|X) instead of a single point estimate. This gives uncertainty quantification — "I don't know" is a valid answer."""
    ),
    (
        "What is entropy in information theory and why is it the basis of the loss function?",
        """Entropy measures uncertainty — or equivalently, the expected amount of information in a message.

For a discrete distribution P:
  H(P) = -Σ p(x) log p(x)

High entropy = high uncertainty (uniform distribution, p(x) = 1/n for all x)
Low entropy  = low uncertainty (deterministic, one p(x) = 1)

**Cross-entropy loss** in classification:
  H(P, Q) = -Σ p(x) log q(x)

Where P is the true label distribution (one-hot) and Q is the model's predicted probabilities.

For one example with true class k:
  Loss = -log q(k)

This penalizes the model for assigning low probability to the correct class. Perfect prediction → loss = 0. Random prediction → loss = log(n_classes).

**KL Divergence** (relative entropy):
  KL(P‖Q) = Σ p(x) log [p(x)/q(x)] = H(P, Q) - H(P)

KL measures how much information is lost when using Q to approximate P. It's always ≥ 0. Cross-entropy = KL + H(P). When P is fixed (labels don't change), minimizing cross-entropy is the same as minimizing KL — you're making Q as close to P as possible."""
    ),
    (
        "What is the expected value and why are loss functions expected values?",
        """Expected value is the long-run average of a random variable:

Discrete:   E[X] = Σ x · P(X=x)
Continuous: E[X] = ∫ x · p(x) dx

The law of large numbers: sample means converge to E[X] as sample size → ∞.

**Why loss functions are expected values:**

You want to minimize the true risk:
  R(θ) = E[(x,y)~D] [L(f_θ(x), y)]

The expectation is over the true data distribution D — which you can't see in full.

What you can do: approximate it with a sample average (empirical risk):
  R̂(θ) = (1/n) Σᵢ L(f_θ(xᵢ), yᵢ)

This is just a Monte Carlo estimate of the expected loss. By the law of large numbers, with enough data, R̂ → R.

SGD samples mini-batches to estimate the gradient of R̂:
  ∇R̂ ≈ (1/B) Σᵢ∈batch ∇L(f_θ(xᵢ), yᵢ)

It's a noisy gradient estimate, but the noise actually helps escape sharp minima. This is the probabilistic foundation for why stochastic gradient descent works."""
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY THEORY — Build it
# ─────────────────────────────────────────────────────────────────────────────

PROB_BUILD = [
    (
        "Implement cross-entropy loss and KL divergence from scratch in numpy.",
        """import numpy as np

def cross_entropy(y_true, y_pred, eps=1e-8):
    \"\"\"
    Cross-entropy loss for classification.
    y_true: one-hot labels (n_samples, n_classes) or class indices (n_samples,)
    y_pred: model probabilities (n_samples, n_classes)
    \"\"\"
    y_pred = np.clip(y_pred, eps, 1 - eps)

    if y_true.ndim == 1:
        # Integer labels → pick log prob of correct class
        n = len(y_true)
        return -np.mean(np.log(y_pred[np.arange(n), y_true]))
    else:
        # One-hot labels
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def kl_divergence(p, q, eps=1e-8):
    \"\"\"KL(p‖q) — how much info is lost approximating p with q.\"\"\"
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))

def entropy(p, eps=1e-8):
    \"\"\"Entropy H(p) — uncertainty of distribution p.\"\"\"
    p = np.clip(p, eps, 1)
    return -np.sum(p * np.log(p))

# Demonstration
n_classes = 4
np.random.seed(42)

# True distribution (one-hot for class 2)
y_true_onehot = np.array([[0, 0, 1, 0]])

# Various model predictions
perfect_pred  = np.array([[0.001, 0.001, 0.997, 0.001]])
decent_pred   = np.array([[0.1,   0.1,   0.7,   0.1]])
random_pred   = np.array([[0.25,  0.25,  0.25,  0.25]])
wrong_pred    = np.array([[0.001, 0.001, 0.001, 0.997]])

print("Cross-entropy losses:")
print(f"  Perfect: {cross_entropy(y_true_onehot, perfect_pred):.4f}")
print(f"  Decent:  {cross_entropy(y_true_onehot, decent_pred):.4f}")
print(f"  Random:  {cross_entropy(y_true_onehot, random_pred):.4f}")
print(f"  Wrong:   {cross_entropy(y_true_onehot, wrong_pred):.4f}")

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.4, 0.4, 0.2])
print(f"\\nEntropy H(p):    {entropy(p):.4f}")
print(f"KL(p‖q):         {kl_divergence(p, q):.4f}")
print(f"Cross-entropy:   {cross_entropy(p.reshape(1,-1), q.reshape(1,-1)):.4f}")
print(f"H(p) + KL(p‖q) = {entropy(p) + kl_divergence(p, q):.4f}  (should equal cross-entropy)")"""
    ),
    (
        "Implement Naive Bayes classifier from scratch and apply it to text classification.",
        """import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    \"\"\"Multinomial Naive Bayes for text classification.\"\"\"

    def fit(self, X, y):
        \"\"\"
        X: list of documents (each a list of words)
        y: list of class labels
        \"\"\"
        self.classes = list(set(y))
        self.class_priors = {}
        self.word_likelihoods = {}

        n = len(y)
        for c in self.classes:
            # Prior P(c)
            docs_c = [X[i] for i in range(n) if y[i] == c]
            self.class_priors[c] = len(docs_c) / n

            # Word counts with Laplace smoothing
            word_counts = defaultdict(int)
            total_words = 0
            for doc in docs_c:
                for word in doc:
                    word_counts[word] += 1
                    total_words += 1

            self.word_likelihoods[c] = (word_counts, total_words)

        # Vocabulary size for smoothing
        vocab = set(w for doc in X for w in doc)
        self.vocab_size = len(vocab)
        return self

    def _log_likelihood(self, doc, c):
        word_counts, total_words = self.word_likelihoods[c]
        log_prob = 0
        for word in doc:
            # Laplace smoothing: (count + 1) / (total + vocab_size)
            count = word_counts.get(word, 0)
            log_prob += np.log((count + 1) / (total_words + self.vocab_size))
        return log_prob

    def predict(self, X):
        preds = []
        for doc in X:
            scores = {}
            for c in self.classes:
                scores[c] = np.log(self.class_priors[c]) + self._log_likelihood(doc, c)
            preds.append(max(scores, key=scores.get))
        return preds


# Simple sentiment example
train_X = [
    ["great", "movie", "loved", "it"],
    ["excellent", "film", "amazing"],
    ["best", "movie", "ever", "loved"],
    ["terrible", "movie", "hated", "it"],
    ["awful", "boring", "waste", "of", "time"],
    ["worst", "film", "terrible", "acting"],
]
train_y = ["pos", "pos", "pos", "neg", "neg", "neg"]

test_X = [
    ["amazing", "acting", "loved", "it"],
    ["boring", "terrible", "waste"],
    ["great", "film"],
]

clf = NaiveBayesClassifier().fit(train_X, train_y)
preds = clf.predict(test_X)

for doc, pred in zip(test_X, preds):
    print(f"  {' '.join(doc):40s} → {pred}")"""
    ),
    (
        "Implement Monte Carlo estimation and show why it works (law of large numbers).",
        """import numpy as np

# Monte Carlo: estimate π by random sampling
def estimate_pi(n_samples):
    # Sample random points in unit square
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)

    # Count points inside unit circle
    inside = (x**2 + y**2) <= 1.0
    return 4 * inside.mean()

np.random.seed(42)
print("Estimating π via Monte Carlo:")
for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
    est = estimate_pi(n)
    print(f"  n={n:>8,}:  π ≈ {est:.6f}  (error: {abs(est - np.pi):.6f})")

# Law of large numbers: sample means converge
print("\\nLaw of large numbers — die roll expected value (should → 3.5):")
np.random.seed(0)
rolls = np.random.randint(1, 7, size=1_000_000)
for n in [10, 100, 1_000, 10_000, 100_000, 1_000_000]:
    print(f"  n={n:>8,}:  mean = {rolls[:n].mean():.4f}")

# Expected value via sampling vs analytical
print("\\nEstimate E[X²] for X ~ Uniform(0,1):")
print(f"  Analytical: 1/3 = {1/3:.6f}")
samples = np.random.uniform(0, 1, 1_000_000)
print(f"  MC estimate: {(samples**2).mean():.6f}")"""
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# APPLIED ML MATH — Connect theory to practice
# ─────────────────────────────────────────────────────────────────────────────

APPLIED_MATH = [
    (
        "Build a complete linear regression model from scratch using only numpy — including gradient descent and analytical solution.",
        """import numpy as np

# Generate data: y = 2x + 3 + noise
np.random.seed(42)
n = 100
X = np.random.randn(n, 1)
y = 2 * X.squeeze() + 3 + 0.5 * np.random.randn(n)

# ─── Analytical solution: θ = (XᵀX)⁻¹Xᵀy ───
X_b = np.column_stack([np.ones(n), X])  # add bias column
theta_analytical = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"Analytical:       intercept={theta_analytical[0]:.4f}, slope={theta_analytical[1]:.4f}")

# ─── Gradient descent ───
def mse_loss(X_b, y, theta):
    residuals = X_b @ theta - y
    return np.mean(residuals**2)

def mse_grad(X_b, y, theta):
    residuals = X_b @ theta - y
    return 2 * X_b.T @ residuals / len(y)

theta = np.zeros(2)
lr = 0.1
for step in range(1000):
    g = mse_grad(X_b, y, theta)
    theta = theta - lr * g

print(f"Gradient descent: intercept={theta[0]:.4f}, slope={theta[1]:.4f}")
print(f"True:             intercept=3.0000, slope=2.0000")

# ─── Predict and evaluate ───
y_pred = X_b @ theta
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - y.mean())**2)
r2 = 1 - ss_res / ss_tot
rmse = np.sqrt(np.mean((y - y_pred)**2))
print(f"\\nR² = {r2:.4f}")
print(f"RMSE = {rmse:.4f}")"""
    ),
    (
        "Implement softmax regression (logistic regression for multiple classes) from scratch.",
        """import numpy as np

def softmax(z):
    # Numerically stable: subtract max before exp
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, y_true):
    n = len(y_true)
    log_probs = np.log(probs[np.arange(n), y_true] + 1e-8)
    return -log_probs.mean()

def accuracy(probs, y_true):
    return (probs.argmax(axis=1) == y_true).mean()

class SoftmaxRegression:
    def __init__(self, n_features, n_classes, lr=0.1):
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros(n_classes)
        self.lr = lr

    def forward(self, X):
        return softmax(X @ self.W + self.b)

    def step(self, X, y):
        n = len(y)
        probs = self.forward(X)

        # Gradient of cross-entropy loss w.r.t. logits: (probs - one_hot) / n
        delta = probs.copy()
        delta[np.arange(n), y] -= 1
        delta /= n

        # Gradients
        dW = X.T @ delta
        db = delta.sum(axis=0)

        self.W -= self.lr * dW
        self.b -= self.lr * db

        return cross_entropy_loss(probs, y)

# Generate 3-class data
np.random.seed(0)
n_per_class = 100
X = np.vstack([
    np.random.randn(n_per_class, 2) + [0, 0],
    np.random.randn(n_per_class, 2) + [4, 0],
    np.random.randn(n_per_class, 2) + [2, 4],
])
y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)

# Normalize
X = (X - X.mean(0)) / X.std(0)

model = SoftmaxRegression(n_features=2, n_classes=3, lr=0.5)

for epoch in range(200):
    loss = model.step(X, y)
    if (epoch + 1) % 50 == 0:
        acc = accuracy(model.forward(X), y)
        print(f"Epoch {epoch+1:3d}  loss={loss:.4f}  acc={acc:.2%}")"""
    ),
    (
        "Implement a two-layer neural network with backpropagation from scratch using numpy.",
        """import numpy as np

def relu(z):        return np.maximum(0, z)
def relu_back(z):   return (z > 0).astype(float)
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

class TwoLayerNet:
    def __init__(self, n_in, n_hidden, n_out, lr=0.01):
        self.W1 = np.random.randn(n_in, n_hidden) * np.sqrt(2/n_in)  # He init
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, n_out) * np.sqrt(2/n_hidden)
        self.b2 = np.zeros(n_out)
        self.lr = lr

    def forward(self, X):
        self.X  = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, y):
        n = len(y)
        # Output layer gradient (softmax + cross-entropy combined)
        dZ2 = self.A2.copy()
        dZ2[np.arange(n), y] -= 1
        dZ2 /= n

        dW2 = self.A1.T @ dZ2
        db2 = dZ2.sum(0)

        # Hidden layer gradient through relu
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_back(self.Z1)

        dW1 = self.X.T @ dZ1
        db1 = dZ1.sum(0)

        # Update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def loss(self, y):
        return -np.log(self.A2[np.arange(len(y)), y] + 1e-8).mean()

    def accuracy(self, y):
        return (self.A2.argmax(1) == y).mean()


# XOR problem (not linearly separable — needs hidden layer)
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0, 1, 1, 0])

net = TwoLayerNet(n_in=2, n_hidden=8, n_out=2, lr=0.5)

for epoch in range(2000):
    net.forward(X)
    net.backward(y)

print("XOR predictions:")
for xi, yi in zip(X, y):
    probs = net.forward(xi.reshape(1,-1))
    pred  = probs.argmax()
    print(f"  {xi} → pred={pred}  true={yi}  {'✓' if pred==yi else '✗'}")
print(f"\\nFinal accuracy: {net.accuracy(y):.0%}")"""
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Assemble all pairs
# ─────────────────────────────────────────────────────────────────────────────

def generate(n=600):
    base = []

    for q, a in LA_EXPLAIN:    base.append(conv(q, a))
    for q, a in LA_BUILD:      base.append(conv(q, a))
    for q, a in CALC_EXPLAIN:  base.append(conv(q, a))
    for q, a in CALC_BUILD:    base.append(conv(q, a))
    for q, a in PROB_EXPLAIN:  base.append(conv(q, a))
    for q, a in PROB_BUILD:    base.append(conv(q, a))
    for q, a in APPLIED_MATH:  base.append(conv(q, a))

    print(f"Unique pairs: {len(base)}")

    if n <= len(base):
        return random.sample(base, n)

    # Sample with replacement to hit n
    sampled = base.copy()
    while len(sampled) < n:
        sampled.append(random.choice(base))
    random.shuffle(sampled)
    return sampled[:n]


def main():
    random.seed(42)
    pairs = generate(n=600)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Output: {OUTPUT}  ({len(pairs)} pairs)")
    print(f"\nTo merge into sft_master.jsonl:")
    print(f"  cat data/sft_master.jsonl data/math_ml_sft.jsonl > /tmp/sft_merged.jsonl && mv /tmp/sft_merged.jsonl data/sft_master.jsonl")


if __name__ == "__main__":
    main()
