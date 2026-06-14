"""
Issue 4: Quantitative validation of the excluded-volume Gaussian decay formula
against the Fraser et al. (1978) analytical expectation.

Fraser, R.D.B., MacRae, T.P. & Suzuki, E. (1978) J. Appl. Cryst. 11, 693-694.

The standard form (used in CRYSOL/ATSAS) is:
    f_eff(q) = f_vac(q) - rho_sol * V * exp(-(q * r_eff)^2 / (4*pi))

where r_eff = (3*V / (4*pi))^(1/3) is the hard-sphere equivalent radius.

Key diagnostics for a single isolated atom:
  - q -> 0 limit:   f_eff -> f_vac - rho_sol * V  (maximum contrast)
  - q -> inf limit: f_eff -> f_vac                (Gaussian decays fully)
  - 1/e decay point: q* = sqrt(4*pi) / r_eff
"""

import jax.numpy as jnp
import numpy as np

from diff_biophys.saxs.kernels import debye_saxs


def test_excluded_volume_q0_limit() -> None:
    """
    At q -> 0 the excluded-volume Gaussian = 1 (exp(0) = 1), so:
        f_eff(0) = f_vac - rho_sol * V
        I(0) = f_eff(0)^2

    This is the maximum contrast reduction from the solvent.
    Uses carbon: f_vac = 6 electrons, V = 16.44 A^3.
    """
    f_vac_val = 6.0  # Carbon electron count
    V = 16.44  # Carbon volume (Pavlov & Svergun 1997)
    rho_sol = 0.334  # water, e/A^3

    coords = jnp.array([[0.0, 0.0, 0.0]])
    q_values = jnp.array([1e-4])  # very small q ~ 0
    f_vac = jnp.array([[f_vac_val]])
    volumes = jnp.array([V])

    I_sol = float(debye_saxs(coords, q_values, f_vac, volumes=volumes, solvent_density=rho_sol)[0])
    f_eff_zero = f_vac_val - rho_sol * V  # ~ 6.0 - 5.49 = 0.51

    np.testing.assert_allclose(
        I_sol,
        f_eff_zero**2,
        rtol=0.01,
        err_msg=(
            f"I_sol(q~0) = {I_sol:.5f}, expected f_eff(0)^2 = {f_eff_zero**2:.5f}. "
            f"f_eff(0) = {f_vac_val:.2f} - {rho_sol:.3f}*{V:.2f} = {f_eff_zero:.4f}"
        ),
    )
    print(
        f"OK EV q->0 limit: f_eff(0) = {f_eff_zero:.4f} e, "
        f"I(0) = {I_sol:.5f} (expected {f_eff_zero**2:.5f})"
    )


def test_excluded_volume_high_q_limit() -> None:
    """
    At high q, the excluded-volume Gaussian -> 0, so f_eff -> f_vac.
    The scattering intensity should approach the vacuum value.

    q_star = sqrt(4*pi) / r_eff ~ 2.25 A^-1 for carbon.
    At q = 5 * q_star the Gaussian term is exp(-25) ~ 1.4e-11 (negligible).
    """
    f_vac_val = 6.0
    V = 16.44
    rho_sol = 0.334
    r_eff = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
    q_star = np.sqrt(4.0 * np.pi) / r_eff
    q_high = 5.0 * q_star  # Gaussian is exp(-25) ~ machine-zero here

    coords = jnp.array([[0.0, 0.0, 0.0]])
    f_vac = jnp.array([[f_vac_val]])
    volumes = jnp.array([V])

    I_vac = float(debye_saxs(coords, jnp.array([q_high]), f_vac, volumes=None)[0])
    I_sol = float(
        debye_saxs(coords, jnp.array([q_high]), f_vac, volumes=volumes, solvent_density=rho_sol)[0]
    )

    np.testing.assert_allclose(
        I_sol,
        I_vac,
        rtol=0.001,
        err_msg=(
            f"At q={q_high:.3f} A^-1 (5*q_star), I_sol={I_sol:.5f} should "
            f"equal I_vac={I_vac:.5f} — excluded-volume Gaussian should have decayed to ~0"
        ),
    )
    print(f"OK EV high-q limit: I_sol({q_high:.2f})={I_sol:.4f} ~ I_vac={I_vac:.4f}")


def test_excluded_volume_1e_decay_point() -> None:
    """
    Verify the analytical 1/e decay point of the excluded-volume Gaussian.

    The Fraser formula exp(-(q*r_eff)^2 / (4*pi)) equals 1/e when:
        q* = sqrt(4*pi) / r_eff

    At q*, the EV correction is rho_sol * V / e, so:
        f_eff(q*) = f_vac - rho_sol * V * exp(-1)

    This pins the constant 4*pi in the denominator against the Fraser formula.
    """
    f_vac_val = 6.0  # Carbon
    V = 16.44
    rho_sol = 0.334
    r_eff = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
    q_star = np.sqrt(4.0 * np.pi) / r_eff

    coords = jnp.array([[0.0, 0.0, 0.0]])
    f_vac = jnp.array([[f_vac_val]])
    volumes = jnp.array([V])

    I_sol = float(
        debye_saxs(coords, jnp.array([q_star]), f_vac, volumes=volumes, solvent_density=rho_sol)[0]
    )
    f_eff_at_qstar = np.sqrt(I_sol)

    # Fraser: f_eff(q*) = f_vac - rho_sol * V * exp(-1)
    expected_feff = f_vac_val - rho_sol * V * np.exp(-1.0)

    np.testing.assert_allclose(
        f_eff_at_qstar,
        expected_feff,
        rtol=0.005,
        err_msg=(
            f"At q*={q_star:.4f} A^-1 (1/e point): "
            f"f_eff={f_eff_at_qstar:.5f}, expected {expected_feff:.5f}. "
            f"If this fails, the 4*pi denominator in the Gaussian may be wrong."
        ),
    )
    print(
        f"OK Fraser 1/e point: q*={q_star:.4f} A^-1, "
        f"f_eff={f_eff_at_qstar:.5f} (expected {expected_feff:.5f})"
    )


def test_excluded_volume_decay_is_gaussian() -> None:
    """
    Verify that the excluded-volume correction decays as a Gaussian in q,
    i.e. that ln(ev_term) vs q^2 is linear with slope = -(r_eff^2 / (4*pi)).

    For a single atom at the origin the Debye self-term gives:
        I_sol(q) = f_eff(q)^2
        f_eff(q) = f_vac - rho_sol * V * exp(-(q*r_eff)^2 / (4*pi))
        ev_term(q) = f_vac - f_eff(q) = rho_sol * V * exp(-(q*r_eff)^2 / (4*pi))
        ln(ev_term / (rho_sol * V)) = -(r_eff^2 / (4*pi)) * q^2
    """
    f_vac_val = 6.0  # Carbon (large enough that f_eff stays positive)
    V = 16.44
    rho_sol = 0.334
    r_eff = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)

    q_values = jnp.linspace(0.01, 1.5, 150)
    coords = jnp.array([[0.0, 0.0, 0.0]])
    f_vac = jnp.ones((1, len(q_values))) * f_vac_val
    volumes = jnp.array([V])

    I_sol = np.array(debye_saxs(coords, q_values, f_vac, volumes=volumes, solvent_density=rho_sol))
    f_eff_arr = np.sqrt(np.maximum(I_sol, 0.0))
    q_arr = np.array(q_values)

    # Recover the EV term: ev(q) = f_vac - f_eff(q)
    ev_term = f_vac_val - f_eff_arr

    # Fit ln(ev / (rho_sol * V)) vs q^2 in the range where the signal is clean
    mask = (q_arr > 0.1) & (q_arr < 1.2) & (ev_term > 1e-4)
    q2_fit = q_arr[mask] ** 2
    ln_ev_fit = np.log(ev_term[mask] / (rho_sol * V))

    slope, intercept = np.polyfit(q2_fit, ln_ev_fit, 1)

    expected_slope = -(r_eff**2) / (4.0 * np.pi)
    np.testing.assert_allclose(
        slope,
        expected_slope,
        rtol=0.02,
        err_msg=(
            f"Fraser Gaussian slope: measured {slope:.5f}, "
            f"expected {expected_slope:.5f}  "
            f"(r_eff={r_eff:.3f} A, denominator=4*pi={4 * np.pi:.4f}). "
            "A wrong denominator (e.g. 2 or 4*pi^2) would shift the slope."
        ),
    )
    np.testing.assert_allclose(
        intercept,
        0.0,
        atol=0.05,
        err_msg=f"Gaussian intercept = {intercept:.4f}, expected ~0",
    )
    print(
        f"OK Fraser EV Gaussian: slope={slope:.5f} (expected {expected_slope:.5f}), "
        f"r_eff={r_eff:.3f} A, intercept={intercept:.4f}"
    )


if __name__ == "__main__":
    test_excluded_volume_q0_limit()
    test_excluded_volume_high_q_limit()
    test_excluded_volume_1e_decay_point()
    test_excluded_volume_decay_is_gaussian()
