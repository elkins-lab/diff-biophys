from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit


@jit
def calculate_rdc_from_tensor(
    bond_vectors: jnp.ndarray, saupe_tensor: jnp.ndarray, d_max: float = 1.0
) -> jnp.ndarray:
    """
    Calculate RDCs from a full 3x3 Saupe alignment tensor.
    D = d_max * sum_ij (v_i * S_ij * v_j)

    Args:
        bond_vectors: (N, 3) unit vectors
        saupe_tensor: (3, 3) symmetric traceless Saupe tensor
        d_max: Maximum dipolar coupling constant (Hz)

    Returns:
        jnp.ndarray: Calculated RDCs (N,)
    """
    # Vectorized computation of v^T S v
    return d_max * jnp.einsum("ni,ij,nj->n", bond_vectors, saupe_tensor, bond_vectors)


@jit
def fit_saupe_tensor(
    bond_vectors: jnp.ndarray, experimental_rdcs: jnp.ndarray, d_max: float = 1.0
) -> jnp.ndarray:
    """
    Fit a Saupe alignment tensor to experimental RDCs using SVD (least squares).

    The RDC formula can be rewritten as D = A * s
    where s = [Sxx, Syy, Sxy, Sxz, Syz] (5 independent components)

    Args:
        bond_vectors: (N, 3) unit vectors
        experimental_rdcs: (N,) measured RDCs in Hz
        d_max: Maximum dipolar coupling constant (Hz)

    Returns:
        jnp.ndarray: (3, 3) Fitted Saupe tensor
    """
    x = bond_vectors[:, 0]
    y = bond_vectors[:, 1]
    z = bond_vectors[:, 2]

    # Basis functions for the 5 independent components
    # Using the identity Szz = -Sxx - Syy
    # D = d_max * [ Sxx*x^2 + Syy*y^2 + Szz*z^2 + 2Sxy*xy + 2Sxz*xz + 2Syz*yz ]
    # D = d_max * [ Sxx(x^2 - z^2) + Syy(y^2 - z^2) + 2Sxy*xy + 2Sxz*xz + 2Syz*yz ]

    A = d_max * jnp.stack([x**2 - z**2, y**2 - z**2, 2 * x * y, 2 * x * z, 2 * y * z], axis=1)

    # Solve A * s = experimental_rdcs
    s, _, _, _ = jnp.linalg.lstsq(A, experimental_rdcs)

    sxx, syy, sxy, sxz, syz = s
    szz = -(sxx + syy)

    tensor = jnp.array([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])

    return tensor


@jit
def calculate_q_factor(calculated_rdcs: jnp.ndarray, experimental_rdcs: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the RDC Q-factor (Cornilescu et al., 1998).
    Q = sqrt( sum((D_calc - D_exp)^2) / sum(D_exp^2) )

    Returns 0.0 when all experimental RDCs are zero (perfect trivial match).

    Args:
        calculated_rdcs: (N,) calculated couplings.
        experimental_rdcs: (N,) measured couplings.

    Returns:
        jnp.ndarray: Scalar Q-factor.
    """
    diff_sq = jnp.sum((calculated_rdcs - experimental_rdcs) ** 2)
    exp_sq = jnp.sum(experimental_rdcs**2)

    # Robust Q-factor calculation to avoid NaN gradients at zero experimental RDCs.
    # We use a safe denominator for the division and then mask the result.
    q = jnp.sqrt(diff_sq / jnp.maximum(exp_sq, 1e-10))
    return jnp.where(exp_sq > 0.0, q, 0.0)


@jit
def calculate_rdc(bond_vectors: jnp.ndarray, da: float, r: float) -> jnp.ndarray:
    """
    Differentiable RDC calculation in the principal axis frame (PAF).

    .. important::
        ``bond_vectors`` **must be expressed in the principal axis frame**
        of the alignment tensor, i.e. the frame where the Saupe tensor is
        diagonal.  Passing lab-frame vectors will give incorrect results
        without any error.

    The formula used is the standard Clore/Bax convention
    (Clore et al. 1998, *J. Magn. Reson.* **133**, 216–221)::

        D = Da · [(3 cos²θ − 1) + (3/2) R sin²θ cos 2φ]
          = Da · [(3z² − 1) + (3/2) R (x² − y²)]

    where ``Da`` is the axial component and ``R = (Axx − Ayy) / Azz`` is
    the rhombicity (0 ≤ R ≤ 2/3).

    Args:
        bond_vectors: (N, 3) unit vectors in the tensor's principal axis frame.
        da: Axial component Da in Hz.
        r: Rhombicity R (0 ≤ R ≤ 2/3).

    Returns:
        jnp.ndarray: (N,) Calculated RDCs in Hz.
    """
    x, y, z = bond_vectors[:, 0], bond_vectors[:, 1], bond_vectors[:, 2]

    axial = 3.0 * z**2 - 1.0
    rhombic = 1.5 * r * (x**2 - y**2)

    return da * (axial + rhombic)


# ---------------------------------------------------------------------------
# NH bond vector reconstruction
# ---------------------------------------------------------------------------


@jit
def nh_bond_vectors(coords: jnp.ndarray) -> jnp.ndarray:
    """Reconstruct amide N–H unit vectors from N–CA–C backbone coordinates.

    The amide H lies in the peptide plane defined by C(i−1), N(i), CA(i).
    Its direction is approximated as anti-parallel to the bisector of the
    N→CA and N→C(i−1) unit vectors, placing H at ~119° from each bond —
    consistent with standard peptide-plane geometry.

    Coordinate layout::

        coords = [N₀, CA₀, C₀,  N₁, CA₁, C₁,  …,  Nₙ, CAₙ, Cₙ]
        N(i)   = coords[3i]
        CA(i)  = coords[3i+1]
        C(i-1) = coords[3i-1]  (for i ≥ 1)

    Residue 0 has no preceding C; its NH vector falls back to −(N→CA).

    Args:
        coords: ``(3N, 3)`` backbone atom coordinates.

    Returns:
        jnp.ndarray: ``(N, 3)`` unit vectors pointing N→H.
    """
    n_atoms = coords[0::3]  # (N_res, 3)
    ca_atoms = coords[1::3]  # (N_res, 3)
    c_atoms = coords[2::3]  # (N_res, 3)

    # Unit vector N→CA
    n_to_ca = ca_atoms - n_atoms
    n_to_ca = n_to_ca / jnp.maximum(jnp.linalg.norm(n_to_ca, axis=-1, keepdims=True), 1e-8)

    # Unit vector N→C(i-1):  use C(i-1) = c_atoms[i-1], with dummy for i=0
    c_prev = jnp.concatenate([c_atoms[:1], c_atoms[:-1]], axis=0)
    n_to_cprev = c_prev - n_atoms
    n_to_cprev = n_to_cprev / jnp.maximum(jnp.linalg.norm(n_to_cprev, axis=-1, keepdims=True), 1e-8)

    # Bisector of the two bonds emanating from N; H is anti-parallel
    bisector = n_to_ca + n_to_cprev
    bisector = bisector / jnp.maximum(jnp.linalg.norm(bisector, axis=-1, keepdims=True), 1e-8)
    nh = -bisector

    # Residue 0 fallback: no C(i-1) available, use −(N→CA)
    nh = jnp.concatenate([-n_to_ca[:1], nh[1:]], axis=0)
    return nh  # (N_res, 3), already unit vectors


# ---------------------------------------------------------------------------
# Fixed-tensor RDC refinement factory
# ---------------------------------------------------------------------------


def make_rdc_refinement_fns(
    exp_res_ids: np.ndarray,
    exp_rdcs: np.ndarray,
    struct_res_ids: np.ndarray,
    d_max: float = 21.7,
) -> tuple[
    Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    Callable[[jnp.ndarray], jnp.ndarray],
    Callable[[jnp.ndarray], jnp.ndarray],
    int,
]:
    """Build three callables for fixed-tensor RDC-based structure refinement.

    The Saupe alignment tensor has 5 free parameters.  If it is fitted inside
    the gradient computation the optimiser trivially drives Q→0 by finding
    backbone orientations that any tensor can fit — the system is severely
    underdetermined.  The standard solution (X-PLOR/CNS/PALES) is to hold the
    tensor fixed during each gradient cycle and re-fit it periodically.

    This factory returns:

    * **loss_fn** ``(coords, fixed_tensor) → scalar MSE``
      Gradient flows through ``coords`` only; ``fixed_tensor`` is wrapped in
      :func:`jax.lax.stop_gradient`.  Use this inside ``jax.grad``.

    * **q_eval_fn** ``(coords) → scalar Q-factor``
      Re-fits the tensor from scratch and returns the best-achievable Q.
      Never call inside a gradient — use this for monitoring only.

    * **make_tensor_fn** ``(coords) → (3, 3) Saupe tensor``
      Fits and returns the alignment tensor for periodic updates outside the
      gradient.

    Args:
        exp_res_ids: ``(M,)`` residue IDs where RDCs were measured.
        exp_rdcs: ``(M,)`` experimental RDC values in Hz.
        struct_res_ids: ``(N,)`` residue IDs present in the structure.
        d_max: Maximum dipolar coupling constant in Hz (default 21.7 Hz for
            ¹⁵N–¹H).

    Returns:
        Tuple ``(loss_fn, q_eval_fn, make_tensor_fn, n_matched)`` where
        ``n_matched`` is the number of residues found in both datasets.

    Raises:
        ValueError: If no residues overlap between ``exp_res_ids`` and
            ``struct_res_ids``.

    Example::

        loss_fn, q_eval_fn, make_tensor_fn, n = make_rdc_refinement_fns(
            rdc_data["res_id"], rdc_data["rdc"], res_ids
        )
        tensor = make_tensor_fn(initial_coords)

        def total_loss(params, tensor):
            phi, psi = params
            coords = build(phi, psi)
            return loss_fn(coords, tensor)

        # Optimization loop
        for i in range(n_steps):
            if i % update_interval == 0:
                tensor = make_tensor_fn(build(*params))
            params = adam_step(total_loss, params, tensor)
    """
    res_id_to_idx = {int(rid): i for i, rid in enumerate(struct_res_ids)}

    matched_struct_idx: list[int] = []
    matched_rdcs: list[float] = []
    for rid, rdc in zip(exp_res_ids, exp_rdcs, strict=False):
        if int(rid) in res_id_to_idx:
            matched_struct_idx.append(res_id_to_idx[int(rid)])
            matched_rdcs.append(float(rdc))

    if not matched_struct_idx:
        raise ValueError(
            "No residues overlap between exp_res_ids and struct_res_ids. "
            "Check that residue numbering conventions match."
        )

    n_matched = len(matched_struct_idx)
    exp_jax = jnp.array(matched_rdcs, dtype=jnp.float32)
    idx_jax = jnp.array(matched_struct_idx, dtype=jnp.int32)

    def _matched_nh(coords: jnp.ndarray) -> jnp.ndarray:
        return cast(jnp.ndarray, nh_bond_vectors(coords)[idx_jax])  # (n_matched, 3)

    def loss_fn(coords: jnp.ndarray, fixed_tensor: jnp.ndarray) -> jnp.ndarray:
        """Fixed-tensor MSE loss; gradient flows through coords only."""
        tensor = jax.lax.stop_gradient(fixed_tensor)
        nh = _matched_nh(coords)
        calc = calculate_rdc_from_tensor(nh, tensor, d_max=d_max)
        return cast(jnp.ndarray, jnp.mean((calc - exp_jax) ** 2))

    def q_eval_fn(coords: jnp.ndarray) -> jnp.ndarray:
        """Re-fit tensor and return honest Q-factor (monitoring only)."""
        nh = _matched_nh(coords)
        tensor = fit_saupe_tensor(nh, exp_jax, d_max=d_max)
        calc = calculate_rdc_from_tensor(nh, tensor, d_max=d_max)
        return cast(jnp.ndarray, calculate_q_factor(calc, exp_jax))

    def make_tensor_fn(coords: jnp.ndarray) -> jnp.ndarray:
        """Fit and return the Saupe tensor for periodic updates."""
        nh = _matched_nh(coords)
        return cast(jnp.ndarray, fit_saupe_tensor(nh, exp_jax, d_max=d_max))

    return loss_fn, q_eval_fn, make_tensor_fn, n_matched
