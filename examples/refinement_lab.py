import jax
import jax.numpy as jnp
import numpy as np
from diff_biophys.geometry import chain_nerf, compute_bond_lengths, compute_bond_angles, compute_dihedrals
from diff_biophys.nmr import calculate_rdc, calculate_karplus_j, calculate_ring_current_shift, calculate_q_factor
from diff_biophys.saxs import debye_saxs

def run_refinement():
    print("🚀 Starting Refinement Lab: Multi-Objective Structure Optimization")
    
    # 1. SETUP TARGET STATE (The "Truth")
    n_atoms = 6
    true_lengths = jnp.full(n_atoms - 1, 1.5)
    true_angles = jnp.full(n_atoms - 2, jnp.radians(110.0))
    true_dihedrals = jnp.array([1.0, -1.0, 1.0])
    
    init_coords = jnp.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.0, 1.4, 0.0]
    ])
    
    true_coords = chain_nerf(init_coords, true_lengths[2:], true_angles[1:], true_dihedrals)
    
    # 2. GENERATE SYNTHETIC DATA
    # SAXS
    q_values = jnp.linspace(0.01, 0.5, 20)
    form_factors = jnp.ones((n_atoms, 20))
    target_saxs = debye_saxs(true_coords, q_values, form_factors)
    
    # RDC
    da, r = 10.0, 0.2
    true_vectors = true_coords[1:] - true_coords[:-1]
    true_vectors /= jnp.linalg.norm(true_vectors, axis=-1, keepdims=True)
    target_rdcs = calculate_rdc(true_vectors, da, r)
    
    # Karplus (on the 3 dihedrals)
    k_params = (6.5, -1.2, 1.6)
    target_j = calculate_karplus_j(true_dihedrals, *k_params)
    
    # Ring Current (shift of last atom due to a mock ring at the origin)
    rc_center = jnp.array([0.0, 0.0, 0.0])
    rc_normal = jnp.array([0.0, 0.0, 1.0])
    rc_intensity = 1.0
    target_rc = calculate_ring_current_shift(true_coords[-1:], rc_center, rc_normal, rc_intensity)
    
    # 3. SETUP STARTING STATE (Distorted)
    starting_dihedrals = jnp.array([0.5, -0.5, 0.5])
    
    # 4. DEFINE LOSS FUNCTION
    def loss_fn(dihedrals):
        # Reconstruct
        coords = chain_nerf(init_coords, true_lengths[2:], true_angles[1:], dihedrals)
        
        # SAXS
        s_loss = jnp.mean((debye_saxs(coords, q_values, form_factors) - target_saxs)**2)
        
        # RDC
        vecs = coords[1:] - coords[:-1]
        vecs /= (jnp.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-10)
        r_loss = jnp.mean((calculate_rdc(vecs, da, r) - target_rdcs)**2)
        
        # Karplus
        k_loss = jnp.mean((calculate_karplus_j(dihedrals, *k_params) - target_j)**2)
        
        # Ring Current
        rc_loss = jnp.mean((calculate_ring_current_shift(coords[-1:], rc_center, rc_normal, rc_intensity) - target_rc)**2)
        
        total_loss = 0.1*s_loss + 1.0*r_loss + 1.0*k_loss + 1.0*rc_loss
        return total_loss, (s_loss, r_loss, k_loss, rc_loss)

    # 5. OPTIMIZATION LOOP
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    
    params = starting_dihedrals
    velocity = jnp.zeros_like(params)
    lr = 0.005
    momentum = 0.9
    n_steps = 400
    
    print(f"{'Step':>4} | {'Total Loss':>12} | {'RMSD (Å)':>12} | {'RDC Q-fact':>12}")
    print("-" * 55)
    
    for i in range(n_steps + 1):
        (total_loss, aux), grads = grad_fn(params)
        
        current_coords = chain_nerf(init_coords, true_lengths[2:], true_angles[1:], params)
        rmsd = jnp.sqrt(jnp.mean(jnp.sum((current_coords - true_coords)**2, axis=-1)))
        
        # Calculate Q-factor for monitoring
        current_vectors = current_coords[1:] - current_coords[:-1]
        current_vectors /= (jnp.linalg.norm(current_vectors, axis=-1, keepdims=True) + 1e-10)
        q_factor = calculate_q_factor(calculate_rdc(current_vectors, da, r), target_rdcs)
        
        if i % 100 == 0:
            print(f"{i:4d} | {total_loss:12.6f} | {rmsd:12.4f} | {q_factor:12.4f}")
            
        velocity = momentum * velocity - lr * grads
        params = params + velocity

    # 6. RESULTS
    final_coords = chain_nerf(init_coords, true_lengths[2:], true_angles[1:], params)
    rmsd = jnp.sqrt(jnp.mean(jnp.sum((final_coords - true_coords)**2, axis=-1)))
    
    print("-" * 55)
    print(f"✅ Optimization Complete!")
    print(f"Final RMSD to Truth: {rmsd:.4f} Å")
    
    if rmsd < 0.1:
        print("🎉 Success! The structure converged to the target.")
    else:
        print("⚠️ Partial convergence. Consider adjusting learning rate or weights.")

if __name__ == "__main__":
    run_refinement()
