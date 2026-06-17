import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diff_biophys.nmr.chemical_shifts import make_ca_shift_loss
from diff_biophys.nmr.rdc import make_rdc_refinement_fns


def test_make_ca_shift_loss_parity() -> None:
    """Verify that make_ca_shift_loss builds a functioning RMSD loss."""
    exp_ids = np.array([1, 2, 3])
    exp_shifts = np.array([55.0, 56.0, 57.0])

    struct_ids = np.array([1, 2, 3, 4])
    struct_names = ["ALA", "ARG", "ASN", "ASP"]

    loss_fn, n_matched = make_ca_shift_loss(exp_ids, exp_shifts, struct_ids, struct_names)
    assert n_matched == 3

    # phi/psi for 4 residues
    phi = jnp.zeros(4)
    psi = jnp.zeros(4)

    loss = loss_fn(phi, psi)
    assert jnp.isfinite(loss)
    assert loss.shape == ()

    # Check differentiability
    grad = jax.grad(loss_fn, argnums=0)(phi, psi)
    assert grad.shape == (4,)
    assert jnp.all(jnp.isfinite(grad))


def test_make_ca_shift_loss_no_overlap() -> None:
    """ValueError is raised when no residues match."""
    with pytest.raises(ValueError, match="No residues overlap"):
        make_ca_shift_loss(np.array([100]), np.array([50.0]), np.array([1, 2]), ["ALA", "ALA"])


def test_make_rdc_refinement_fns_parity() -> None:
    """Verify that make_rdc_refinement_fns builds functioning loss and monitoring closures."""
    exp_ids = np.array([1, 2])
    exp_rdcs = np.array([10.0, -5.0])
    struct_ids = np.array([1, 2, 3])

    loss_fn, q_eval_fn, make_tensor_fn, n_matched = make_rdc_refinement_fns(
        exp_ids, exp_rdcs, struct_ids
    )
    assert n_matched == 2

    # Coords for 3 residues (3 atoms each = 9 atoms)
    coords = jnp.zeros((9, 3))
    # Add some variation so NH vectors are not all zeros
    coords = coords.at[1].set(jnp.array([1.0, 0.0, 0.0]))  # CA0
    coords = coords.at[3].set(jnp.array([0.0, 1.0, 0.0]))  # N1

    fixed_tensor = jnp.eye(3) - (1.0 / 3.0) * jnp.eye(3)  # Traceless-ish

    loss = loss_fn(coords, fixed_tensor)
    assert jnp.isfinite(loss)

    q = q_eval_fn(coords)
    assert jnp.isfinite(q)

    tensor = make_tensor_fn(coords)
    assert tensor.shape == (3, 3)
    assert jnp.all(jnp.isfinite(tensor))


def test_make_rdc_refinement_fns_no_overlap() -> None:
    """ValueError is raised when no residues match."""
    with pytest.raises(ValueError, match="No residues overlap"):
        make_rdc_refinement_fns(np.array([100]), np.array([10.0]), np.array([1, 2]))
