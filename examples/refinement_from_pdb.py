"""
Refinement from PDB: Hybrid Refinement Workflow
===============================================

This example demonstrates how to use the 'synth' ecosystem together:
1. Generate a physically realistic (energy-minimized) starting structure with synth-pdb.
2. Use synth-nmr parameters (automatically pulled into diff-biophys).
3. Refine the structure using JAX-based gradients to match experimental observables.

Requirements:
    pip install diff-biophys[examples] biotite synth-pdb synth-nmr
"""

from typing import cast

import biotite.structure as stripe
import biotite.structure.io.pdb as pdb
import jax
import jax.numpy as jnp
import numpy as np
import optax

from diff_biophys.geometry.nerf import chain_nerf
from diff_biophys.geometry.torsions import compute_dihedrals
from diff_biophys.nmr.rdc import calculate_q_factor, calculate_rdc


# Helper to extract phi/psi from N-CA-C backbone coordinates
def compute_phi_psi(coords: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # For a chain of N-CA-C:
    # Phi is C(i-1)-N(i)-CA(i)-C(i)
    # Psi is N(i)-CA(i)-C(i)-N(i+1)
    d = compute_dihedrals(coords)
    # d[0] is N0-CA0-C0-N1 (Psi 0)
    # d[1] is CA0-C0-N1-CA1 (Omega 0)
    # d[2] is C0-N1-CA1-C1 (Phi 1)

    # We want:
    # phi: [nan, d[2], d[5], ...]
    # psi: [d[0], d[3], d[6], ...]

    # Actually, let's just use the NeRF interleave pattern in reverse
    psi = d[0::3]
    phi = d[2::3]

    # Pad to N_RES
    psi = jnp.concatenate([psi, jnp.array([0.0])])
    phi = jnp.concatenate([jnp.array([0.0]), phi])

    return phi, psi


# 1. Load the PDB files (generated previously via synth-pdb)
# Target: The "true" native state we want to recover
# Initial: A physically plausible random coil, energy-minimized with OpenMM
target_pdb = pdb.PDBFile.read("examples/target_refinement.pdb")
initial_pdb = pdb.PDBFile.read("examples/initial_minimized.pdb")

target_struct = target_pdb.get_structure(model=1)
initial_struct = initial_pdb.get_structure(model=1)


# Select backbone atoms (N, CA, C) in the correct order
def get_backbone_coords(struct: stripe.AtomArray) -> jnp.ndarray:
    # Filter for backbone atoms
    mask = (struct.atom_name == "N") | (struct.atom_name == "CA") | (struct.atom_name == "C")
    backbone = struct[mask]
    # Ensure standard N-CA-C order per residue
    return cast(jnp.ndarray, jnp.array(backbone.coord, dtype=jnp.float32))


target_coords = get_backbone_coords(target_struct)
initial_coords = get_backbone_coords(initial_struct)

n_residues = (
    len(target_struct) // target_struct.stack_depth
    if hasattr(target_struct, "stack_depth")
    else len(np.unique(target_struct.res_id))
)
# For this simple Ala-10 case:
n_residues = 10

print(f"Loaded structures with {n_residues} residues.")


# 2. Setup Target Observables
# We'll use RDCs as our experimental "pull"
def get_peptide_bond_vectors(coords: jnp.ndarray) -> jnp.ndarray:
    # Extract C-N vectors (indices 2->3, 5->6, etc.)
    # In N-CA-C ordering: 0=N, 1=CA, 2=C
    c_atoms = coords[2::3][:-1]
    n_atoms = coords[3::3]
    vecs = n_atoms - c_atoms
    return cast(jnp.ndarray, vecs / jnp.linalg.norm(vecs, axis=-1, keepdims=True))


target_vectors = get_peptide_bond_vectors(target_coords)
# Synthetic RDCs (Da=10Hz, R=0.1)
target_rdcs = calculate_rdc(target_vectors, da=10.0, r=0.1)

# 3. Refinement Model (Torsional Space)
# We refine the phi/psi angles. Bond lengths and angles are kept at ideal values.
CA_C_LENGTH = 1.525
C_N_LENGTH = 1.329
N_CA_LENGTH = 1.459
CA_C_N_ANGLE = jnp.radians(116.2)
C_N_CA_ANGLE = jnp.radians(121.7)
N_CA_C_ANGLE = jnp.radians(111.2)

bond_lengths = jnp.array([C_N_LENGTH, N_CA_LENGTH, CA_C_LENGTH] * (n_residues - 1))
bond_angles = jnp.array([CA_C_N_ANGLE, C_N_CA_ANGLE, N_CA_C_ANGLE] * (n_residues - 1))

# Initial seed atoms (N0, CA0, C0) from the minimized structure
# This anchors the refinement in the same global orientation
seed_coords = initial_coords[:3]


def build_structure(phi: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
    # Trans peptide bonds
    omega = jnp.full((n_residues - 1,), jnp.pi)
    # Interleave: psi[0], omega[0], phi[1], psi[1], omega[1]...
    d = jnp.stack([psi[:-1], omega, phi[1:]], axis=1).ravel()
    return cast(jnp.ndarray, chain_nerf(seed_coords, bond_lengths, bond_angles, d))


# Calculate starting angles from the minimized PDB
# (Even though it's minimized, the global fold is random)
init_phi, init_psi = compute_phi_psi(initial_coords)


@jax.jit
def loss_fn(params: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    phi, psi = params
    coords = build_structure(phi, psi)
    vectors = get_peptide_bond_vectors(coords)
    rdcs = calculate_rdc(vectors, da=10.0, r=0.1)
    return cast(jnp.ndarray, calculate_q_factor(rdcs, target_rdcs))


# 4. Optimization
optimizer = optax.adam(learning_rate=0.02)
params = (init_phi, init_psi)
opt_state = optimizer.init(params)


@jax.jit
def step(
    params: tuple[jnp.ndarray, jnp.ndarray], opt_state: optax.OptState
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], optax.OptState, jnp.ndarray]:
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


print("Refining minimized decoy against RDC gradients...")
for i in range(151):
    params, opt_state, loss = step(params, opt_state)
    if i % 50 == 0:
        print(f"Step {i:03d} | RDC Q-factor: {loss:.4f}")

# 5. Save the Refined Structure
final_phi, final_psi = params
final_coords = build_structure(final_phi, final_psi)

# Create a new biotite structure for output
# We'll just update the coordinates of the initial backbone
final_struct = initial_struct.copy()
mask = (
    (final_struct.atom_name == "N")
    | (final_struct.atom_name == "CA")
    | (final_struct.atom_name == "C")
)
# Note: NeRF only builds the backbone. For a full-atom structure,
# you would typically use synth-pdb to re-pack sidechains.
final_struct.coord[mask] = np.array(final_coords)

# Save to PDB
out_pdb = pdb.PDBFile()
out_pdb.set_structure(final_struct)
out_pdb.write("examples/refined_output.pdb")

print("\nRefinement complete!")
print("Initial Q-factor: ", loss_fn((init_phi, init_psi)))
print("Final Q-factor:   ", loss_fn(params))
print("Saved refined structure to examples/refined_output.pdb")
print("\nTo visualize the improvement, run:")
print("synth-pdb --visualize --input-pdb examples/refined_output.pdb")
