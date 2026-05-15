import jax
import jax.numpy as jnp
import numpy as np
import time
from diff_biophys.saxs.kernels import debye_saxs

def benchmark_saxs():
    print('--- SAXS Kernel Performance Benchmark ---')
    
    q_values = jnp.linspace(0.01, 0.5, 51)
    
    for n_atoms in [100, 500, 2000, 5000]:
        # Generate random coordinates
        coords = jax.random.normal(jax.random.PRNGKey(0), (n_atoms, 3))
        # Form factors (all 1.0 for benchmark)
        f_q = jnp.ones((n_atoms, len(q_values)))
        
        # Warmup
        _ = debye_saxs(coords, q_values, f_q).block_until_ready()
        
        start = time.time()
        n_runs = 5
        for _ in range(n_runs):
            _ = debye_saxs(coords, q_values, f_q).block_until_ready()
        
        avg_time = (time.time() - start) / n_runs
        print(f'N = {n_atoms:5d} atoms: {avg_time:.4f} seconds')

if __name__ == '__main__':
    benchmark_saxs()
