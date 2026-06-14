import pytest

try:
    import jax.numpy as jnp
    import torch

    from diff_biophys.torch_interop import HAS_TORCH, jax_to_torch
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_jax_to_torch_biophysics_kernel() -> None:
    """Verify gradient flow through a real biophysics kernel (SAXS)."""
    from diff_biophys.saxs.kernels import debye_saxs

    # 1. Define JAX kernel call
    q_values = jnp.array([0.1, 0.2], dtype=jnp.float32)
    ff = jnp.ones((2, 2), dtype=jnp.float32)

    def saxs_loss(coords: jnp.ndarray) -> jnp.ndarray:
        iq = debye_saxs(coords, q_values, ff)
        return jnp.sum(iq)

    # 2. Wrap for Torch
    torch_saxs = jax_to_torch(saxs_loss)

    # 3. Use in Torch
    # Atoms at (0,0,0) and (1,0,0).
    coords_torch = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True)
    loss = torch_saxs(coords_torch)

    assert loss.dim() == 0
    loss.backward()

    assert coords_torch.grad is not None
    assert torch.all(torch.isfinite(coords_torch.grad))
    assert not torch.all(coords_torch.grad == 0)
