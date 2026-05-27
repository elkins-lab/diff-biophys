import jax
import jax.numpy as jnp

from diff_biophys.geometry.nerf import position_atom_3d


def demonstrate_nerf_gradient():
    print("--- NeRF Gradient Demonstration ---")

    # Target position we want to reach
    target_pos = jnp.array([2.5, 1.5, 0.5])

    # Fixed reference atoms
    p1 = jnp.array([0.0, 0.0, 0.0])
    p2 = jnp.array([1.46, 0.0, 0.0])
    p3 = jnp.array([2.01, 1.34, 0.0])

    # Parameters we want to optimize (bond_len, angle, dihedral)
    params = jnp.array([1.33, jnp.radians(116.0), jnp.radians(180.0)])

    def loss_fn(p):
        p4 = position_atom_3d(p1, p2, p3, p[0], p[1], p[2])
        return jnp.sum((p4 - target_pos) ** 2)

    # Calculate gradient
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params)

    print(f"Initial Parameters: {params}")
    print(f"Loss: {loss_fn(params):.4f}")
    print(f"Gradients: {grads}")

    # Simple Gradient Descent
    lr = 0.01
    p_opt = params
    for i in range(100):
        p_opt -= lr * grad_fn(p_opt)
        if i % 20 == 0:
            print(f"Step {i}, Loss: {loss_fn(p_opt):.4f}")

    print(f"Optimized Parameters: {p_opt}")
    print(f"Final Loss: {loss_fn(p_opt):.4f}")
    print("--- Success! Structure refined via autodiff ---\n")


if __name__ == "__main__":
    demonstrate_nerf_gradient()
