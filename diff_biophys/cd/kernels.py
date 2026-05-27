def simulate_cd_matrix(peptide_positions, dipole_orientations, wavelengths):
    """
    Matrix-Method CD Simulation (DeVoe Theory).

    .. warning::
        **NOT IMPLEMENTED.** This function currently raises
        ``NotImplementedError``.  A full implementation would require:

        1. Building the interaction matrix V_ij of dipole–dipole coupling
           energies between amide chromophores.
        2. Computing the frequency-dependent polarizability tensor for each
           chromophore.
        3. Solving the coupled-oscillator equations for the complex optical
           response and converting to molar ellipticity [θ].

    Args:
        peptide_positions: (N, 3) positions of amide chromophores.
        dipole_orientations: (N, 3) unit vectors for transition dipoles.
        wavelengths: (M,) wavelengths in nm.

    Raises:
        NotImplementedError: Always, until this module is implemented.
    """
    raise NotImplementedError(
        "diff_biophys.cd.simulate_cd_matrix is not yet implemented. "
        "Contributions welcome — see the DeVoe (1964) matrix method as a "
        "starting point."
    )
