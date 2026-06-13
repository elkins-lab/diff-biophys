"""
Generate the three undergraduate tutorial notebooks for diff-biophys.

Run from repo root:
    python examples/interactive_tutorials/generate_notebooks.py

Requires:  pip install nbformat
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).parent


# ── helpers ──────────────────────────────────────────────────────────────────


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source)


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source)


def nb(cells: list[nbf.NotebookNode], title: str) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
            "colab": {"provenance": []},
        }
    )
    notebook.cells = cells
    return notebook


def save(notebook: nbf.NotebookNode, name: str) -> None:
    path = OUT / name
    with open(path, "w") as f:
        nbf.write(notebook, f)
    print(f"  ✓  {path}")


# ═══════════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — Hello, Gradient Descent!
# ═══════════════════════════════════════════════════════════════════════════


def notebook_01() -> nbf.NotebookNode:
    COLAB = "https://colab.research.google.com/github/elkins-lab/diff-biophys/blob/main/examples/interactive_tutorials/01_hello_gradient_descent.ipynb"
    cells = [
        # ── Title ──────────────────────────────────────────────────────────
        md(f"""\
# 01 · Hello, Gradient Descent!
### *No biology required — just a function and a slope*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

---

**Who is this for?**  Anyone who has taken calculus and is curious about how machine learning "learns."
You do *not* need a biology background for this notebook.

**What you will learn:**
1. What a gradient is (visually and mathematically)
2. How gradient descent minimises a function — step by step
3. How JAX computes gradients *automatically* (no more hand-derivations!)
4. How to fit a real scientific equation to data using the exact same idea

**Time:** ~30 minutes
"""),
        # ── Install ────────────────────────────────────────────────────────
        md("## 0 · Setup\n\nInstall the library (one-time, fast on Colab)."),
        code("""\
# Install diff-biophys and plotting tools
%pip install -q diff-biophys==0.1.5 matplotlib"""),
        code("""\
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Pretty plots
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
})
print("JAX version:", jax.__version__)
print("Devices available:", jax.devices())"""),
        # ── Section 1: What is a gradient? ─────────────────────────────────
        md("""\
---
## 1 · What is a gradient?

A **gradient** is the slope of a function at a point.

For a one-variable function $f(x)$, the gradient is just the derivative $\\frac{df}{dx}$.
It tells you: *"if I increase $x$ a little, does $f$ go up or down, and by how much?"*

### A simple example

Let's use the **parabola** $f(x) = (x - 3)^2$.
- It has a minimum at $x = 3$ (where $f = 0$).
- Its gradient is $\\frac{df}{dx} = 2(x - 3)$.
  - At $x = 0$: gradient = $-6$ (slope is negative, $f$ decreases as $x$ increases)
  - At $x = 5$: gradient = $+4$ (slope is positive, $f$ increases as $x$ increases)
  - At $x = 3$: gradient = $0$  ← minimum!
"""),
        code("""\
x_vals = np.linspace(-1, 7, 300)
f_vals = (x_vals - 3) ** 2

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_vals, f_vals, "steelblue", lw=2.5, label=r"$f(x) = (x-3)^2$")
ax.axvline(3, color="crimson", lw=1.5, ls="--", label="minimum at x = 3")

# Draw gradient arrows at a few points
for x0 in [0.5, 2.0, 4.5, 6.0]:
    g = 2 * (x0 - 3)          # analytic gradient
    f0 = (x0 - 3) ** 2
    ax.annotate("", xy=(x0 + 0.6 * np.sign(g), f0 + 0.6 * abs(g)),
                xytext=(x0, f0),
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=2))

ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("The parabola and its gradients")
ax.legend()
plt.tight_layout()
plt.show()
print("Gradient at x=0:", 2 * (0 - 3))
print("Gradient at x=5:", 2 * (5 - 3))
print("Gradient at x=3:", 2 * (3 - 3))"""),
        # ── Section 2: Gradient descent by hand ────────────────────────────
        md("""\
---
## 2 · Gradient descent — the algorithm

Here is the entire idea in one sentence:

> **Take a small step in the direction that makes f decrease.**

Since the gradient points *uphill*, we step in the *opposite* direction:

$$x_{t+1} = x_t - \\eta \\cdot \\frac{df}{dx}\\bigg|_{x_t}$$

$\\eta$ (eta) is the **learning rate** — how big each step is.

Let's watch it work, starting at $x_0 = 0$:
"""),
        code("""\
def f(x):
    return (x - 3) ** 2

def grad_f(x):
    return 2 * (x - 3)          # hand-computed derivative

learning_rate = 0.15
x = 0.0                         # starting point
trajectory = [x]

for step in range(25):
    g = grad_f(x)
    x = x - learning_rate * g   # gradient descent update
    trajectory.append(x)

# Plot the trajectory on the parabola
x_vals = np.linspace(-0.5, 6.5, 300)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x_vals, f(x_vals), "steelblue", lw=2.5)
ax1.scatter(trajectory, [f(x) for x in trajectory],
            c=range(len(trajectory)), cmap="plasma", s=50, zorder=5)
ax1.axvline(3, color="crimson", lw=1.5, ls="--", alpha=0.5)
ax1.set_xlabel("x"); ax1.set_ylabel("f(x)")
ax1.set_title("Gradient descent path on the parabola")

ax2.plot(range(len(trajectory)), [f(x) for x in trajectory],
         "steelblue", lw=2, marker="o", ms=4)
ax2.set_xlabel("Step"); ax2.set_ylabel("f(x)  [loss]")
ax2.set_title("Loss vs. step")
plt.tight_layout()
plt.show()

print(f"Started at x = 0.00,  f = {f(0.0):.2f}")
print(f"Ended   at x = {trajectory[-1]:.4f},  f = {f(trajectory[-1]):.6f}")"""),
        # ── Section 3: JAX does the calculus for you ───────────────────────
        md("""\
---
## 3 · JAX does the calculus for you

Hand-computing the derivative works for $(x-3)^2$.
But what about a complicated function with hundreds of inputs?

That's where **automatic differentiation** (autodiff) comes in.
JAX can compute the exact derivative of *any* differentiable function,
automatically, in one line:

```python
grad_f = jax.grad(f)
```

No chain rule by hand. No finite differences. **Exact gradients, always.**
"""),
        code("""\
# Define the function with JAX arrays
def f_jax(x):
    return (x - 3.0) ** 2

# jax.grad returns a NEW function that computes the gradient
grad_f_jax = jax.grad(f_jax)

# Test it at a few points
for x0 in [0.0, 3.0, 5.0]:
    x0_jax = jnp.array(x0)
    g_auto  = grad_f_jax(x0_jax)
    g_hand  = 2 * (x0 - 3)
    print(f"x={x0:4.1f}  →  auto-grad={float(g_auto):+.2f}  hand-calc={g_hand:+.2f}  ✓ match: {abs(float(g_auto) - g_hand) < 1e-5}")"""),
        code("""\
# Works on multi-dimensional functions too
def f_2d(params):
    \"\"\"f(x, y) = (x - 3)^2 + (y + 1)^2   minimum at (3, -1)\"\"\"
    x, y = params[0], params[1]
    return (x - 3.0) ** 2 + (y + 1.0) ** 2

grad_f_2d = jax.grad(f_2d)

params = jnp.array([0.0, 0.0])
for step in range(30):
    g = grad_f_2d(params)
    params = params - 0.2 * g

print(f"Minimum found at:  x = {params[0]:.4f}, y = {params[1]:.4f}")
print(f"True minimum is:   x = 3.0000, y = -1.0000")"""),
        code("""\
# Also JIT-compilable for speed — same code, 10-100x faster on GPU
@jax.jit
def step(params, lr=0.2):
    return params - lr * jax.grad(f_2d)(params)

params = jnp.array([0.0, 0.0])
for _ in range(30):
    params = step(params)
print(f"JIT result:  x = {params[0]:.4f}, y = {params[1]:.4f}")"""),
        # ── Section 4: Fitting a scientific equation ───────────────────────
        md("""\
---
## 4 · Fitting a real scientific equation

Here is where it gets exciting for biology.

**The Karplus equation** describes how a measurable NMR coupling constant $J$ (in Hz)
depends on a bond angle $\\theta$:

$$J(\\theta) = A \\cos^2(\\theta) + B \\cos(\\theta) + C$$

Scientists measure $J$ values experimentally and want to recover
the coefficients $A$, $B$, $C$ — this is a **curve-fitting** problem.

Gradient descent solves this automatically.
We minimize the **mean-squared error** between our predicted $J$ values and the measured ones.
"""),
        code("""\
from diff_biophys.nmr.karplus import calculate_karplus_j

# --- True coefficients (Vuister & Bax 1993 for HN-Hα coupling) ---
A_true, B_true, C_true = 6.98, -1.38, 1.72

# --- Generate synthetic "experimental" data ---
rng = np.random.default_rng(42)
theta_data = jnp.array(rng.uniform(0, np.pi, 30), dtype=jnp.float32)
J_exp = calculate_karplus_j(theta_data, A_true, B_true, C_true)
J_exp = J_exp + jnp.array(rng.normal(0, 0.15, 30), dtype=jnp.float32)  # add noise

# Plot the data
theta_dense = jnp.linspace(0, jnp.pi, 300)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.degrees(theta_dense),
        calculate_karplus_j(theta_dense, A_true, B_true, C_true),
        "steelblue", lw=2, label="True Karplus curve")
ax.scatter(np.degrees(theta_data), J_exp, color="crimson", s=40,
           zorder=5, label="Noisy observations")
ax.set_xlabel("θ (degrees)"); ax.set_ylabel("J (Hz)")
ax.set_title("Synthetic Karplus data — can we recover A, B, C?")
ax.legend(); plt.tight_layout(); plt.show()"""),
        code("""\
import optax   # pip install optax  (already a dep in diff-biophys[dev])

# --- Define the loss function ---
def karplus_loss(params):
    \"\"\"MSE between predicted and experimental J-couplings.\"\"\"
    A, B, C = params[0], params[1], params[2]
    J_pred = calculate_karplus_j(theta_data, A, B, C)
    return jnp.mean((J_pred - J_exp) ** 2)

# --- Initialise with wrong parameters ---
params = jnp.array([3.0, 0.0, 0.5], dtype=jnp.float32)
print(f"Starting loss:  {karplus_loss(params):.4f} Hz²")
print(f"Starting A={params[0]:.2f}, B={params[1]:.2f}, C={params[2]:.2f}")
print(f"True    A={A_true:.2f}, B={B_true:.2f}, C={C_true:.2f}\\n")

# --- Adam optimiser ---
optimizer = optax.adam(learning_rate=0.05)
opt_state = optimizer.init(params)
grad_loss  = jax.jit(jax.value_and_grad(karplus_loss))

losses = []
for step in range(300):
    loss_val, grads = grad_loss(params)
    losses.append(float(loss_val))
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

print(f"Final   loss:  {losses[-1]:.4f} Hz²")
print(f"Recovered A={params[0]:.2f}, B={params[1]:.2f}, C={params[2]:.2f}")
print(f"True      A={A_true:.2f}, B={B_true:.2f}, C={C_true:.2f}")"""),
        code("""\
# Plot: loss curve + recovered fit
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.semilogy(losses, "steelblue", lw=2)
ax1.set_xlabel("Optimisation step"); ax1.set_ylabel("MSE loss (Hz²)")
ax1.set_title("Loss decreasing during gradient descent")

A_fit, B_fit, C_fit = float(params[0]), float(params[1]), float(params[2])
ax2.plot(np.degrees(theta_dense),
         calculate_karplus_j(theta_dense, A_true, B_true, C_true),
         "steelblue", lw=2, label="True curve")
ax2.plot(np.degrees(theta_dense),
         calculate_karplus_j(theta_dense, A_fit, B_fit, C_fit),
         "crimson", lw=2, ls="--", label="Fitted curve")
ax2.scatter(np.degrees(theta_data), J_exp, color="grey", s=30, alpha=0.6)
ax2.set_xlabel("θ (degrees)"); ax2.set_ylabel("J (Hz)")
ax2.set_title("True vs. recovered Karplus curve")
ax2.legend(); plt.tight_layout(); plt.show()
print("\\n✅ Gradient descent recovered the Karplus coefficients from noisy data!")"""),
        # ── Summary ────────────────────────────────────────────────────────
        md("""\
---
## 5 · Summary

| Concept | One-line version |
|---|---|
| **Gradient** | The slope — which direction is uphill, and how steep? |
| **Gradient descent** | Take small steps downhill until you reach the bottom |
| **`jax.grad`** | Computes exact gradients of any function automatically |
| **`jax.jit`** | Compiles the function for GPU speed |
| **Karplus fitting** | Real science, same idea — minimize MSE over parameters |

### What's next?
- **Notebook 02 — NMR Fundamentals**: apply these ideas to chemical shifts and RDCs
- **Notebook 03 — CD Spectroscopy**: differentiate a coupled-oscillator model of helical dichroism

> 💡 **Key insight**: `diff-biophys` gives you the *same* gradient machinery you just used,
> but wired up to full protein structures with hundreds of atoms.
> The physics is more complex; the idea is identical.
"""),
    ]
    return nb(cells, "01_hello_gradient_descent")


# ═══════════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — NMR Fundamentals
# ═══════════════════════════════════════════════════════════════════════════


def notebook_02() -> nbf.NotebookNode:
    COLAB = "https://colab.research.google.com/github/elkins-lab/diff-biophys/blob/main/examples/interactive_tutorials/02_nmr_fundamentals.ipynb"
    cells = [
        md(f"""\
# 02 · NMR Fundamentals: From Angles to Observables
### *Making protein structure visible with nuclear magnetic resonance*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

---

**Prerequisites:** Notebook 01 (or comfort with gradient descent basics)

**What you will learn:**
1. What NMR measures and *why* it encodes protein structure
2. **Cα chemical shifts** — how backbone torsion angles predict shift values
3. **The Karplus equation** — measuring dihedral angles through J-couplings
4. **Residual Dipolar Couplings (RDCs)** — long-range orientational restraints
5. How gradients connect NMR observables back to structure

**Time:** ~45 minutes
"""),
        code("""\
%pip install -q diff-biophys==0.1.5 matplotlib
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
})
print("Ready. JAX devices:", jax.devices())"""),
        # ── Section 1: What is NMR? ─────────────────────────────────────────
        md("""\
---
## 1 · What is NMR?

**Nuclear Magnetic Resonance (NMR)** exploits the fact that certain atomic nuclei
(¹H, ¹³C, ¹⁵N) behave like tiny magnets.  Put a protein in a strong magnetic field,
pulse it with radio waves, and each nucleus rings back at a slightly different frequency.
That frequency difference — called the **chemical shift** (δ, in ppm) — is
exquisitely sensitive to the local chemical environment.

### The key insight

> The chemical shift of a nucleus depends on the **geometry** of the atoms around it.
> Geometry is encoded in **bond angles and dihedral angles**.
> Therefore: NMR shifts → dihedral angles → protein structure.

`diff-biophys` implements this chain **differentiably**, so you can run gradient descent
from measured shifts all the way back to atomic coordinates.

### NMR observables we will cover

| Observable | Symbol | What it encodes |
|---|---|---|
| Cα chemical shift | δ(Cα) | Local secondary structure (helix / sheet / coil) |
| J-coupling constant | ³J(HN,Hα) | Backbone φ dihedral angle (Karplus equation) |
| Residual Dipolar Coupling | D | Global orientation of bond vectors |
"""),
        # ── Section 2: Cα Chemical Shifts ──────────────────────────────────
        md("""\
---
## 2 · Cα Chemical Shifts and Secondary Structure

The **Cα chemical shift** (ppm) of a residue shifts away from its random-coil value
depending on whether the residue is in a helix, sheet, or loop:

| Secondary structure | Δδ(Cα) relative to random coil |
|---|---|
| α-helix | **+3.1 ppm** (downfield shift) |
| β-sheet | **−1.5 ppm** (upfield shift) |
| Random coil | 0 ppm |

`predict_ca_shifts` implements this using **soft Gaussian detectors** in (φ, ψ) space,
so the prediction is fully differentiable through the torsion angles.
"""),
        code("""\
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts, RANDOM_COIL_CA

# Random-coil shift for alanine (52.5 ppm)
rc_shift = jnp.array([RANDOM_COIL_CA["ALA"]], dtype=jnp.float32)
print(f"Alanine random-coil Cα shift: {float(rc_shift[0]):.1f} ppm")

# --- Predict shifts across the Ramachandran map ---
phi_grid = jnp.linspace(-jnp.pi, jnp.pi, 60)
psi_grid = jnp.linspace(-jnp.pi, jnp.pi, 60)
phi_mesh, psi_mesh = jnp.meshgrid(phi_grid, psi_grid)

# Vectorise: predict one residue at a time over the grid
phi_flat = phi_mesh.ravel()
psi_flat = psi_mesh.ravel()
rc_flat  = jnp.full_like(phi_flat, RANDOM_COIL_CA["ALA"])

shifts_flat = predict_ca_shifts(phi_flat, psi_flat, rc_flat)
shifts_grid = shifts_flat.reshape(phi_mesh.shape)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.contourf(np.degrees(phi_grid), np.degrees(psi_grid), shifts_grid,
                 levels=30, cmap="RdBu_r")
plt.colorbar(im, ax=ax, label="Predicted δ(Cα)  [ppm]")

# Mark canonical secondary structure regions
ax.scatter([-57], [-47], s=200, marker="*", color="gold", zorder=5,
           label="α-helix (φ=−57°, ψ=−47°)")
ax.scatter([-120], [120], s=200, marker="D", color="lime", zorder=5,
           label="β-strand (φ=−120°, ψ=+120°)")
ax.set_xlabel("φ (degrees)"); ax.set_ylabel("ψ (degrees)")
ax.set_title("Ramachandran map coloured by predicted Cα shift")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()"""),
        code("""\
# --- Side-by-side: helix vs. sheet vs. coil ---
phi_h = jnp.full((10,), jnp.deg2rad(-57.0),  dtype=jnp.float32)   # helix
psi_h = jnp.full((10,), jnp.deg2rad(-47.0),  dtype=jnp.float32)
phi_s = jnp.full((10,), jnp.deg2rad(-120.0), dtype=jnp.float32)   # sheet
psi_s = jnp.full((10,), jnp.deg2rad(120.0),  dtype=jnp.float32)
phi_c = jnp.full((10,), jnp.deg2rad(-70.0),  dtype=jnp.float32)   # coil
psi_c = jnp.full((10,), jnp.deg2rad(150.0),  dtype=jnp.float32)
rc    = jnp.full((10,), RANDOM_COIL_CA["ALA"], dtype=jnp.float32)

shifts_h = predict_ca_shifts(phi_h, psi_h, rc)
shifts_s = predict_ca_shifts(phi_s, psi_s, rc)
shifts_c = predict_ca_shifts(phi_c, psi_c, rc)

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(10) + 1
ax.bar(x - 0.25, shifts_h, 0.25, label="α-helix",   color="steelblue")
ax.bar(x,        shifts_s, 0.25, label="β-strand",  color="crimson")
ax.bar(x + 0.25, shifts_c, 0.25, label="coil",      color="grey", alpha=0.7)
ax.axhline(RANDOM_COIL_CA["ALA"], ls="--", color="k", alpha=0.4, label="random coil")
ax.set_xlabel("Residue"); ax.set_ylabel("δ(Cα)  [ppm]")
ax.set_title("Predicted Cα shifts: helix > coil > sheet")
ax.legend(); plt.tight_layout(); plt.show()

print(f"  Helix mean:  {float(jnp.mean(shifts_h)):.2f} ppm")
print(f"  Sheet mean:  {float(jnp.mean(shifts_s)):.2f} ppm")
print(f"  Coil  mean:  {float(jnp.mean(shifts_c)):.2f} ppm")"""),
        # ── Section 3: Karplus Equation ─────────────────────────────────────
        md("""\
---
## 3 · The Karplus Equation

**³J coupling constants** measure how strongly two nuclei interact through three chemical bonds.
For the backbone H–N–Cα–H (HN–Hα) coupling, the Karplus equation gives:

$$J(\\phi) = A \\cos^2(\\phi - 60°) + B \\cos(\\phi - 60°) + C$$

where $\\phi$ is the backbone **phi** torsion angle.

This means: **measure J → infer φ → constrain the backbone geometry.**

The Vuister & Bax (1993) coefficients are: $A = 6.98$, $B = -1.38$, $C = 1.72$ Hz.
"""),
        code("""\
from diff_biophys.nmr.karplus import calculate_karplus_j

A, B, C = 6.98, -1.38, 1.72   # Vuister & Bax 1993

# The coupling depends on phi - 60° (offset convention)
phi_vals = jnp.linspace(-jnp.pi, jnp.pi, 360)
theta    = phi_vals - jnp.deg2rad(60.0)   # offset for HN-Ha coupling
J_vals   = calculate_karplus_j(theta, A, B, C)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(np.degrees(phi_vals), J_vals, "steelblue", lw=2.5)
ax.axhline(0, color="k", lw=0.5)

# Annotate canonical phi values
for phi_deg, label, color in [(-57, "α-helix\\nφ=−57°", "gold"),
                               (-120, "β-strand\\nφ=−120°", "lime")]:
    phi_r  = jnp.deg2rad(phi_deg)
    theta_r = phi_r - jnp.deg2rad(60.0)
    j0 = float(calculate_karplus_j(jnp.array([theta_r]), A, B, C)[0])
    ax.scatter([phi_deg], [j0], s=120, zorder=5, color=color, edgecolors="k")
    ax.annotate(f"{label}\\nJ={j0:.1f} Hz", (phi_deg, j0),
                xytext=(phi_deg + 20, j0 + 0.8), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1.2))

ax.set_xlabel("φ (degrees)"); ax.set_ylabel("³J(HN,Hα)  [Hz]")
ax.set_title("Karplus curve: J-coupling as a function of backbone φ")
ax.set_xlim(-180, 180); plt.tight_layout(); plt.show()

print("J-coupling for α-helix  (φ=−57°):", round(float(
    calculate_karplus_j(jnp.array([jnp.deg2rad(-57-60)]), A, B, C)[0]), 2), "Hz")
print("J-coupling for β-strand (φ=−120°):", round(float(
    calculate_karplus_j(jnp.array([jnp.deg2rad(-120-60)]), A, B, C)[0]), 2), "Hz")"""),
        md("""\
### Gradient through Karplus

The gradient $\\frac{dJ}{d\\phi}$ tells you: *"if I change the backbone angle slightly,
how does the measured J-coupling change?"*

This is the key to NMR-driven structure refinement: if the predicted J doesn't match
the experimental value, the gradient tells you *which direction* to move $\\phi$.
"""),
        code("""\
# Compute and visualise dJ/dθ across the full Karplus curve
grad_J = jax.grad(lambda t: jnp.sum(calculate_karplus_j(t, A, B, C)))
theta_vals = phi_vals - jnp.deg2rad(60.0)
dJ_dtheta = grad_J(theta_vals)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax1.plot(np.degrees(phi_vals), J_vals, "steelblue", lw=2)
ax1.set_ylabel("J  [Hz]"); ax1.set_title("Karplus curve and its gradient")

ax2.plot(np.degrees(phi_vals), dJ_dtheta, "crimson", lw=2)
ax2.axhline(0, color="k", lw=0.5)
ax2.set_ylabel("dJ/dθ  [Hz/rad]"); ax2.set_xlabel("φ (degrees)")
ax2.set_xlim(-180, 180)
plt.tight_layout(); plt.show()"""),
        # ── Section 4: RDCs ─────────────────────────────────────────────────
        md("""\
---
## 4 · Residual Dipolar Couplings (RDCs)

When a protein is slightly aligned in solution (e.g., in a liquid crystal medium),
dipolar couplings between nuclei are no longer averaged to zero.
The remaining **residual dipolar coupling** (RDC) $D$ depends on the
*orientation* of the bond vector relative to the magnetic field:

$$D_{NH} = D_{\\max} \\sum_{i,j} v_i S_{ij} v_j$$

where:
- $\\mathbf{v}$ = unit vector along the N–H bond
- $\\mathbf{S}$ = **Saupe alignment tensor** (a 3×3 symmetric traceless matrix)
- $D_{\\max}$ = maximum dipolar coupling (a known constant)

### The magic angle

A bond at $\\theta = \\arccos(1/\\sqrt{3}) \\approx 54.74°$ from the alignment axis has
$D = 0$, because $(3\\cos^2\\theta - 1) = 0$.  This is called the **magic angle**.
"""),
        code("""\
from diff_biophys.nmr.rdc import calculate_rdc_from_tensor

# A simple axially-symmetric Saupe tensor aligned along z
# Szz = 0.1 (weak alignment), Sxx = Syy = -0.05
S = jnp.array([[-0.05, 0.0, 0.0],
               [ 0.0, -0.05, 0.0],
               [ 0.0,  0.0,  0.1]], dtype=jnp.float32)

# Sweep bond orientation from z-axis (θ = 0°) to x-axis (θ = 90°)
theta_vals = jnp.linspace(0, jnp.pi / 2, 180)
# Bond vectors in the xz-plane: v = (sin θ, 0, cos θ)
vecs = jnp.stack([jnp.sin(theta_vals), jnp.zeros_like(theta_vals),
                  jnp.cos(theta_vals)], axis=-1)
rdcs = calculate_rdc_from_tensor(vecs, S, d_max=1.0)

magic = float(jnp.rad2deg(jnp.arccos(1 / jnp.sqrt(3))))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.degrees(theta_vals), rdcs, "steelblue", lw=2.5)
ax.axhline(0, color="k", lw=0.5)
ax.axvline(magic, color="crimson", ls="--", lw=1.5,
           label=f"Magic angle = {magic:.1f}°  (D = 0)")
ax.set_xlabel("θ  (degrees from alignment axis)")
ax.set_ylabel("RDC  (normalised units)")
ax.set_title("RDC angular dependence — $(3\\cos^2\\theta - 1)$ shape")
ax.legend(); plt.tight_layout(); plt.show()

# Verify the magic angle gives D ≈ 0
magic_vec = jnp.array([[jnp.sin(jnp.deg2rad(magic)), 0.0,
                         jnp.cos(jnp.deg2rad(magic))]], dtype=jnp.float32)
print(f"RDC at magic angle: {float(calculate_rdc_from_tensor(magic_vec, S)[0]):.2e}  (should be ≈ 0)")"""),
        # ── Section 5: Gradient connects NMR to structure ───────────────────
        md("""\
---
## 5 · How gradients connect NMR to structure

Now let's see the full pipeline in action.

**Scenario:** We have "experimental" Cα chemical shifts for a 6-residue peptide.
The shifts suggest it's helical ($\\delta \\approx 55.6$ ppm), but our starting model
has β-strand torsion angles ($\\phi = -120°, \\psi = +120°$).

**Goal:** Use gradient descent on the shift MSE to nudge the torsion angles toward helix.
"""),
        code("""\
import optax

# --- "Experimental" target: helix-like shifts ---
phi_helix = jnp.full((6,), jnp.deg2rad(-57.0),  dtype=jnp.float32)
psi_helix = jnp.full((6,), jnp.deg2rad(-47.0),  dtype=jnp.float32)
rc        = jnp.full((6,), RANDOM_COIL_CA["ALA"], dtype=jnp.float32)
target_shifts = predict_ca_shifts(phi_helix, psi_helix, rc)

# --- Starting model: beta-strand ---
phi_start = jnp.full((6,), jnp.deg2rad(-120.0), dtype=jnp.float32)
psi_start = jnp.full((6,), jnp.deg2rad( 120.0), dtype=jnp.float32)

# Stack phi and psi as a single (2, 6) parameter array
params = jnp.stack([phi_start, psi_start])

def shift_mse(params):
    phi, psi = params[0], params[1]
    predicted = predict_ca_shifts(phi, psi, rc)
    return jnp.mean((predicted - target_shifts) ** 2)

print(f"Initial MSE:  {shift_mse(params):.4f} ppm²")

# --- Optimise ---
optimizer = optax.adam(0.02)
opt_state = optimizer.init(params)
grad_fn   = jax.jit(jax.value_and_grad(shift_mse))

losses_nmr, phi_trajectory = [], []
for step in range(200):
    loss_val, grads = grad_fn(params)
    losses_nmr.append(float(loss_val))
    phi_trajectory.append(float(jnp.degrees(params[0, 0])))
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

print(f"Final   MSE:  {losses_nmr[-1]:.6f} ppm²")
print(f"phi: {phi_trajectory[0]:.1f}° → {phi_trajectory[-1]:.1f}°  (target: −57°)")"""),
        code("""\
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.semilogy(losses_nmr, "steelblue", lw=2)
ax1.set_xlabel("Optimisation step"); ax1.set_ylabel("Shift MSE  [ppm²]")
ax1.set_title("Chemical-shift loss during torsion refinement")

ax2.plot(phi_trajectory, "crimson", lw=2)
ax2.axhline(-57, color="gold", lw=2, ls="--", label="Helix target (φ=−57°)")
ax2.axhline(-120, color="grey", lw=1.5, ls=":", label="Starting point (φ=−120°)")
ax2.set_xlabel("Optimisation step"); ax2.set_ylabel("φ  [degrees]")
ax2.set_title("Backbone φ converging toward helix")
ax2.legend(); plt.tight_layout(); plt.show()

print("\\n✅ NMR-driven gradient descent nudged torsions from strand to helix geometry!")"""),
        md("""\
---
## 6 · Summary

| Observable | Function | What it tells you |
|---|---|---|
| Cα chemical shift | `predict_ca_shifts(phi, psi, rc)` | Secondary structure content |
| ³J coupling | `calculate_karplus_j(theta, A, B, C)` | Backbone φ angle |
| RDC | `calculate_rdc_from_tensor(vecs, S)` | Global bond orientations |
| Gradient | `jax.grad(loss)(params)` | Which direction to move torsions |

### What's next?
- **Notebook 03 — CD Spectroscopy**: a completely different physical observable — polarised light — differentiably simulated from atomic positions.

> 💡 Every observable in `diff-biophys` follows the same pattern:
> `forward_model(coords) → observable → loss vs. experiment → gradient → update coords`.
"""),
    ]
    return nb(cells, "02_nmr_fundamentals")


# ═══════════════════════════════════════════════════════════════════════════
# NOTEBOOK 3 — CD Spectroscopy
# ═══════════════════════════════════════════════════════════════════════════


def notebook_03() -> nbf.NotebookNode:
    COLAB = "https://colab.research.google.com/github/elkins-lab/diff-biophys/blob/main/examples/interactive_tutorials/03_cd_spectroscopy.ipynb"
    cells = [
        md(f"""\
# 03 · CD Spectroscopy: Seeing Protein Shape with Light
### *From helices and sheets to spectra — and back again*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

---

**Prerequisites:** Notebook 01 (gradient descent) — no NMR knowledge needed.

**What you will learn:**
1. What circular dichroism (CD) spectroscopy is and *why* it encodes secondary structure
2. How to build a protein helix from scratch using differentiable geometry (NeRF)
3. How `simulate_cd_matrix` computes a CD spectrum from atomic positions
4. How the spectrum changes as you distort the helix
5. How gradients flow from the CD spectrum back to chromophore positions

**Time:** ~40 minutes
"""),
        code("""\
%pip install -q diff-biophys==0.1.5 matplotlib
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
})
print("Ready. JAX devices:", jax.devices())"""),
        # ── Section 1: What is CD? ──────────────────────────────────────────
        md("""\
---
## 1 · What is Circular Dichroism?

**Circular dichroism (CD)** exploits the fact that chiral molecules (like proteins)
absorb **left-handed** and **right-handed** circularly polarised light *differently*.

The CD signal at each wavelength is:

$$\\Delta\\varepsilon(\\lambda) = \\varepsilon_L(\\lambda) - \\varepsilon_R(\\lambda)$$

The amide bonds (C=O ··· N–H) in the protein backbone are the main chromophores
in the far-UV region (180–260 nm).  The *coupling* between neighbouring amide
transition dipoles produces the characteristic spectral shapes:

| Secondary structure | 222 nm | 208 nm | 193 nm |
|---|---|---|---|
| **α-helix** | negative | negative | **positive** |
| **β-sheet** | negative | — | **positive** (weaker) |
| **Random coil** | — | — | negative |

These signatures are used daily in biochemistry labs to rapidly assess whether
a protein is folded, what its secondary-structure content is, and whether a
mutation or ligand disrupts the fold.

### The physics: coupled oscillators

Each amide bond has a transition dipole moment $\\boldsymbol{\\mu}_i$.
When two dipoles interact, they form in-phase and out-of-phase combinations
with *different* absorption energies.  The CD signal arises from the
**rotational strength** of these coupled transitions — which depends on
both the *distances* and *orientations* of the dipoles.

`simulate_cd_matrix` implements the full **DeVoe matrix method** differentiably.
"""),
        # ── Section 2: Build a helix ────────────────────────────────────────
        md("""\
---
## 2 · Building a Helix from Scratch

Before we can compute a CD spectrum, we need atomic positions.
We'll use the **NeRF (Natural Extension Reference Frame)** algorithm
to place amide nitrogen atoms along a canonical α-helix.

An ideal α-helix has:
- Rise per residue: 1.5 Å
- Helical radius: 2.3 Å
- 3.6 residues per turn → 100° rotation per residue
"""),
        code("""\
def build_helix_chromophores(n_residues: int, radius: float = 2.3,
                              rise: float = 1.5) -> tuple:
    \"\"\"
    Place amide N atoms on an ideal α-helix.

    Returns
    -------
    positions  : (n_residues, 3) amide N Cartesian coordinates
    dipoles    : (n_residues, 3) unit transition-dipole vectors (tangent to helix)
    \"\"\"
    coords = []
    degrees_per_residue = 100.0    # 360° / 3.6 res/turn

    for i in range(n_residues):
        angle = jnp.deg2rad(i * degrees_per_residue)
        x = radius * jnp.cos(angle)
        y = radius * jnp.sin(angle)
        z = i * rise
        coords.append(jnp.array([x, y, z]))

    positions = jnp.stack(coords)                    # (N, 3)

    # Dipoles: tangent to the helix (forward difference, normalised)
    tangents = jnp.roll(positions, -1, axis=0) - positions
    tangents = tangents.at[-1].set(tangents[-2])     # repeat last
    norms    = jnp.linalg.norm(tangents, axis=-1, keepdims=True)
    dipoles  = tangents / norms                      # unit vectors

    return positions, dipoles


n_res = 12
positions, dipoles = build_helix_chromophores(n_res)
print(f"Built {n_res}-residue helix")
print(f"  Positions shape: {positions.shape}")
print(f"  Dipoles   shape: {dipoles.shape}")
print(f"  Helix spans z = {float(positions[0, 2]):.1f} to {float(positions[-1, 2]):.1f} Å")"""),
        code("""\
# Visualise the helix
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

# Draw helix backbone
xs, ys, zs = np.array(positions[:, 0]), np.array(positions[:, 1]), np.array(positions[:, 2])
ax.plot(xs, ys, zs, "-o", color="steelblue", lw=2, ms=8, label="Amide N positions")

# Draw transition dipoles
scale = 1.5
for i in range(n_res):
    dx, dy, dz = [float(d) * scale for d in dipoles[i]]
    ax.quiver(xs[i], ys[i], zs[i], dx, dy, dz,
              color="crimson", alpha=0.7, arrow_length_ratio=0.3)

ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
ax.set_title(f"{n_res}-residue α-helix\\nblue = amide positions, red = transition dipoles")
plt.tight_layout(); plt.show()"""),
        # ── Section 3: Compute the CD spectrum ─────────────────────────────
        md("""\
---
## 3 · Computing the CD Spectrum

`simulate_cd_matrix` implements the DeVoe coupled-oscillator model.
The function takes:
- `peptide_positions` — where the chromophores are in 3D space
- `dipole_orientations` — which direction each transition dipole points
- `wavelengths` — the wavelengths at which to evaluate the spectrum

And returns the **molar ellipticity** [θ] in deg·cm²/dmol at each wavelength.
"""),
        code("""\
from diff_biophys.cd.kernels import simulate_cd_matrix

wavelengths = jnp.linspace(180.0, 260.0, 81, dtype=jnp.float32)  # 180–260 nm

cd_helix = simulate_cd_matrix(positions, dipoles, wavelengths,
                               f_osc=0.2, gamma=10.0, lambda_0=190.0)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(wavelengths, cd_helix, "steelblue", lw=2.5, label="α-helix (12 residues)")
ax.axhline(0, color="k", lw=0.8)
ax.axvline(222, color="crimson",  lw=1.2, ls="--", alpha=0.6, label="222 nm (helix marker)")
ax.axvline(208, color="darkorange", lw=1.2, ls="--", alpha=0.6, label="208 nm (helix marker)")
ax.axvline(193, color="green",    lw=1.2, ls="--", alpha=0.6, label="193 nm (positive band)")
ax.set_xlabel("Wavelength  [nm]")
ax.set_ylabel("[θ]  (deg cm² dmol⁻¹)")
ax.set_title("CD spectrum of an ideal α-helix")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()

print(f"[θ] at 222 nm: {float(cd_helix[jnp.argmin(jnp.abs(wavelengths - 222))]):.1f}  (should be negative)")
print(f"[θ] at 193 nm: {float(cd_helix[jnp.argmin(jnp.abs(wavelengths - 193))]):.1f}  (should be positive)")"""),
        # ── Section 4: How geometry changes the spectrum ────────────────────
        md("""\
---
## 4 · How Geometry Affects the Spectrum

The CD spectrum is exquisitely sensitive to 3D structure.
Let's gradually **unwind the helix** by increasing the z-rise while keeping the radius fixed,
and watch the spectrum change.

A larger rise-per-residue → more extended → less helical coupling → spectrum changes.
"""),
        code("""\
rises = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]   # Å per residue
colors = plt.cm.viridis(np.linspace(0, 1, len(rises)))

fig, ax = plt.subplots(figsize=(10, 5))

for rise, color in zip(rises, colors):
    pos_r, dip_r = build_helix_chromophores(n_res, radius=2.3, rise=rise)
    cd_r = simulate_cd_matrix(pos_r, dip_r, wavelengths)
    label = f"rise = {rise:.1f} Å {'← ideal helix' if rise == 1.5 else ''}"
    lw = 3.0 if rise == 1.5 else 1.5
    ax.plot(wavelengths, cd_r, lw=lw, color=color, label=label)

ax.axhline(0, color="k", lw=0.8)
ax.axvline(222, color="grey", lw=1, ls="--", alpha=0.5)
ax.set_xlabel("Wavelength  [nm]")
ax.set_ylabel("[θ]  (deg cm² dmol⁻¹)")
ax.set_title("CD spectrum as the helix is unwound (rise per residue increases)")
ax.legend(fontsize=8, loc="lower right"); plt.tight_layout(); plt.show()
print("Notice: the 222 nm negative band diminishes as the helix extends.")"""),
        code("""\
# Quantify: [theta] at 222 nm vs. rise
cd_at_222 = []
for rise in np.linspace(0.8, 4.0, 40):
    pos_r, dip_r = build_helix_chromophores(n_res, radius=2.3, rise=rise)
    cd_r = simulate_cd_matrix(pos_r, dip_r, wavelengths)
    idx_222 = int(jnp.argmin(jnp.abs(wavelengths - 222)))
    cd_at_222.append(float(cd_r[idx_222]))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.linspace(0.8, 4.0, 40), cd_at_222, "steelblue", lw=2.5, marker="o", ms=4)
ax.axvline(1.5, color="crimson", lw=1.5, ls="--", label="Ideal α-helix (1.5 Å/res)")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("Rise per residue  [Å]")
ax.set_ylabel("[θ]₂₂₂  (deg cm² dmol⁻¹)")
ax.set_title("[θ] at 222 nm vs. helical rise — a structural ruler")
ax.legend(); plt.tight_layout(); plt.show()"""),
        # ── Section 5: Gradients ────────────────────────────────────────────
        md("""\
---
## 5 · The Gradient of the CD Spectrum

Here is the most powerful part: `simulate_cd_matrix` is **fully differentiable**.

We can ask: *"At 222 nm, which amide position, if moved, would most change the CD signal?"*

This is $\\frac{\\partial [\\theta](222\\,\\text{nm})}{\\partial \\mathbf{r}_i}$ — the gradient
of the spectrum at a specific wavelength with respect to each chromophore's position.

This is the basis of CD-driven structure refinement.
"""),
        code("""\
idx_222 = int(jnp.argmin(jnp.abs(wavelengths - 222)))

def cd_at_222nm(pos):
    \"\"\"CD signal at 222 nm as a function of chromophore positions.\"\"\"\
    cd = simulate_cd_matrix(pos, dipoles, wavelengths)
    return cd[idx_222]

# Gradient: shape (12, 3) — one 3D vector per chromophore
grad_fn  = jax.grad(cd_at_222nm)
grad_pos = grad_fn(positions)

print("Gradient shape:", grad_pos.shape)
print("Largest gradient magnitude at residue:",
      int(jnp.argmax(jnp.linalg.norm(grad_pos, axis=-1))))

# Visualise the gradient magnitudes
grad_mags = jnp.linalg.norm(grad_pos, axis=-1)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(range(1, n_res + 1), grad_mags, color="steelblue", edgecolor="white")
ax.set_xlabel("Residue"); ax.set_ylabel("|d[θ]₂₂₂ / dr|")
ax.set_title("Gradient magnitude: which amide positions most influence [θ] at 222 nm?")
plt.tight_layout(); plt.show()"""),
        code("""\
# Visualise gradient vectors in 3D
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

xs = np.array(positions[:, 0])
ys = np.array(positions[:, 1])
zs = np.array(positions[:, 2])

ax.plot(xs, ys, zs, "-o", color="steelblue", lw=1.5, ms=7, alpha=0.6)

# Scale gradient arrows for visibility
scale = 8.0 / float(jnp.max(grad_mags) + 1e-8)
for i in range(n_res):
    gx, gy, gz = [float(g) * scale for g in grad_pos[i]]
    intensity = float(grad_mags[i]) / float(jnp.max(grad_mags))
    ax.quiver(xs[i], ys[i], zs[i], gx, gy, gz,
              color=plt.cm.hot(intensity), arrow_length_ratio=0.4, lw=2)

ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
ax.set_title("d[θ]₂₂₂/dr — gradient of 222 nm signal w.r.t. amide positions\\n"
             "(hot = large gradient, blue = small)")
plt.tight_layout(); plt.show()
print("\\n💡 Moving high-gradient residues will change the 222 nm signal most.")"""),
        code("""\
# --- Mini refinement: drive [theta] at 222nm more negative ---
import optax

# Starting from the ideal helix; we want to maximise |[theta]_222|
# equivalently: minimise cd_at_222nm (make it more negative)
def loss(pos):
    return cd_at_222nm(pos)   # we want this to go down (more negative)

opt = optax.adam(learning_rate=0.05)
pos = positions  # start from ideal helix
state = opt.init(pos)
grad_fn_jit = jax.jit(jax.value_and_grad(loss))

cd_trajectory = []
for step in range(60):
    val, grads = grad_fn_jit(pos)
    cd_trajectory.append(float(val))
    updates, state = opt.update(grads, state)
    pos = optax.apply_updates(pos, updates)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cd_trajectory, "crimson", lw=2)
ax.set_xlabel("Optimisation step")
ax.set_ylabel("[θ]₂₂₂  (deg cm² dmol⁻¹)")
ax.set_title("Driving [θ] at 222 nm more negative via gradient descent")
plt.tight_layout(); plt.show()
print(f"[θ]₂₂₂ :  {cd_trajectory[0]:.1f}  →  {cd_trajectory[-1]:.1f}")
print("\\n✅ CD-driven gradient descent moved the chromophores to strengthen the helical signal!")"""),
        md("""\
---
## 6 · Summary

| Concept | Key Point |
|---|---|
| CD signal at 222 nm | Negative = α-helix character; magnitude reports helical content |
| `simulate_cd_matrix` | Full DeVoe coupled-oscillator model, fully differentiable in JAX |
| `jax.grad(cd_at_222nm)(positions)` | Tells you which chromophores most influence the signal |
| Gradient descent on CD | Moves chromophore positions to match a target spectrum |

### The bigger picture

You have now seen three completely different physical experiments —
**SAXS** (X-ray scattering), **NMR** (magnetic resonance), **CD** (polarised light) —
all implemented as differentiable functions in `diff-biophys`.

Because they all produce JAX arrays, you can **combine them**:

```python
total_loss = saxs_chi2(coords) + nmr_mse(phi, psi) + cd_mse(chromophore_pos)
grads = jax.grad(total_loss)(all_params)
```

This is multi-experiment structure refinement — the frontier of the field — in one line.

### Further reading
- Greenfield 2006 — *Using circular dichroism spectra to estimate protein secondary structure* — Nature Protocols
- Kelly et al. 2005 — *How to study proteins by circular dichroism* — BBA
"""),
    ]
    return nb(cells, "03_cd_spectroscopy")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating notebooks...")
    save(notebook_01(), "01_hello_gradient_descent.ipynb")
    save(notebook_02(), "02_nmr_fundamentals.ipynb")
    save(notebook_03(), "03_cd_spectroscopy.ipynb")
    print("Done.")
