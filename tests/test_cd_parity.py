import pytest
from diff_biophys.cd.kernels import simulate_cd_matrix
import jax.numpy as jnp


def test_cd_not_implemented():
    """
    Verify that the CD matrix-method kernel raises NotImplementedError.

    The full DeVoe / matrix-method implementation is not yet available.
    This test documents that fact explicitly so any future implementation
    is automatically detected and can be validated against real CD data.
    """
    n_residues = 10
    wavelengths = jnp.linspace(190, 250, 61)
    positions = jnp.zeros((n_residues, 3))
    orientations = jnp.zeros((n_residues, 3))

    with pytest.raises(NotImplementedError):
        simulate_cd_matrix(positions, orientations, wavelengths)
