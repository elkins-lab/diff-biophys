import pytest

try:
    import jax
    import jax.numpy as jnp
    import torch

    from diff_biophys.torch_interop import HAS_TORCH, jax_to_torch
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_jax_to_torch_forward_backward() -> None:
    # Define a simple JAX function
    @jax.jit
    def simple_jax_fn(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.sum(x**2) + jnp.sum(y**3)

    # Wrap it
    torch_fn = jax_to_torch(simple_jax_fn)

    # Create PyTorch tensors that require gradients
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = torch.tensor([1.0, 3.0], requires_grad=True)

    # Forward pass
    out = torch_fn(x, y)

    # Expected output: (1^2 + 2^2) + (1^3 + 3^3) = 5 + 28 = 33
    assert torch.isclose(out, torch.tensor(33.0))

    # Backward pass
    out.backward()

    # Expected gradients:
    # dx = 2x -> [2.0, 4.0]
    # dy = 3y^2 -> [3.0, 27.0]
    assert torch.allclose(x.grad, torch.tensor([2.0, 4.0]))
    assert torch.allclose(y.grad, torch.tensor([3.0, 27.0]))
