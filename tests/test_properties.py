import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from diff_biophys.geometry.nerf import position_atom_3d
from diff_biophys.geometry.superposition import kabsch_alignment

# Fixed structure for Kabsch property test
n_atoms = 10
P_GLOBAL = np.random.randn(n_atoms, 3).astype(np.float32)
P_CENTERED_GLOBAL = P_GLOBAL - np.mean(P_GLOBAL, axis=0)


@settings(deadline=None)
@given(
    angle=st.floats(0, 2 * np.pi),
    tx=st.floats(-100, 100),
    ty=st.floats(-100, 100),
    tz=st.floats(-100, 100),
)
def test_kabsch_property(angle: float, tx: float, ty: float, tz: float) -> None:
    """
    Kabsch alignment should result in zero RMSD for identical,
    rotated and translated structures.
    """
    # 2. Rotate around Z axis
    R_true = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        dtype=np.float32,
    )

    Q = (R_true @ P_CENTERED_GLOBAL.T).T + np.array([tx, ty, tz])

    # 3. Align
    R_fit, t_fit = kabsch_alignment(jnp.array(P_GLOBAL), jnp.array(Q))

    # 4. Transform P
    P_fit = (np.array(R_fit) @ P_GLOBAL.T).T + np.array(t_fit)

    # 5. Check RMSD
    np.testing.assert_allclose(P_fit, Q, atol=1e-3)


@settings(deadline=None)
@given(
    length=st.floats(0.1, 10.0),
    angle=st.floats(0.1, jnp.pi - 0.1),
    dihedral=st.floats(-jnp.pi, jnp.pi),
)
def test_nerf_property(length: float, angle: float, dihedral: float) -> None:
    """
    NeRF should place p4 at the correct distance from p3.
    """
    p1 = jnp.array([1.0, 0.0, 0.0])
    p2 = jnp.array([0.0, 0.0, 0.0])
    p3 = jnp.array([0.0, 1.0, 0.0])

    p4 = position_atom_3d(p1, p2, p3, length, angle, dihedral)
    dist = jnp.linalg.norm(p4 - p3)
    np.testing.assert_allclose(float(dist), length, atol=1e-4)


if __name__ == "__main__":
    # Note: when running via pytest, hypothesis handles the loop.
    # When running manually, we'd need to call .hypothesis_test()
    pass
