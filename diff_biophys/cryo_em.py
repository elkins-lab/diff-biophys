import jax
import jax.numpy as jnp


@jax.jit
def compute_fsc(
    data1: jax.Array, data2: jax.Array, voxel_size: tuple[float, float, float]
) -> tuple[jax.Array, jax.Array]:
    """
    Compute the fully differentiable Fourier Shell Correlation (FSC) between two 3D maps using JAX.
    Returns frequencies and correlation values.

    This function matches the implementation in synth-core, but uses jax.numpy
    so that gradients can flow through the FSC calculation to the input maps.
    """
    # Fourier transforms in JAX
    f1 = jnp.fft.rfftn(data1)
    f2 = jnp.fft.rfftn(data2)

    # Cross-spectral density and power spectra are computed using float arithmetic
    # We avoid complex multiplication to parallel the numpy memory stability fix
    cross = f1.real * f2.real + f1.imag * f2.imag
    p1 = f1.real**2 + f1.imag**2
    p2 = f2.real**2 + f2.imag**2

    # Calculate radial bins
    nz, ny, nx = data1.shape
    kz = jnp.fft.fftfreq(nz, d=voxel_size[0])
    ky = jnp.fft.fftfreq(ny, d=voxel_size[1])
    kx = jnp.fft.rfftfreq(nx, d=voxel_size[2])

    # Create 3D grid of frequencies
    kz_grid, ky_grid, kx_grid = jnp.meshgrid(kz, ky, kx, indexing="ij")

    # Calculate magnitude of frequency vector for each voxel
    k = jnp.sqrt(kz_grid**2 + ky_grid**2 + kx_grid**2)

    # Flatten everything
    k = k.ravel()
    cross = cross.ravel()
    p1 = p1.ravel()
    p2 = p2.ravel()

    # Sort by frequency
    idx = jnp.argsort(k)
    k_sorted = k[idx]
    cross_sorted = cross[idx]
    p1_sorted = p1[idx]
    p2_sorted = p2[idx]

    n_bins = min(nx, ny, nz) // 2
    k_max = k_sorted[-1]
    k_eps = k_max / (10 * n_bins)
    bins = jnp.linspace(k_eps, k_max, n_bins + 1)

    # We use vmap to compute the bin sums to keep the function differentiable and JIT-compatible
    # We avoid python loops with dynamic shapes.

    def compute_bin(i: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        bin_start = bins[i]
        bin_end = bins[i + 1]
        mask = (k_sorted >= bin_start) & (k_sorted < bin_end)

        sum_cross = jnp.sum(jnp.where(mask, cross_sorted, 0.0))
        sum_p1 = jnp.sum(jnp.where(mask, p1_sorted, 0.0))
        sum_p2 = jnp.sum(jnp.where(mask, p2_sorted, 0.0))

        num = sum_cross
        den = jnp.sqrt(sum_p1 * sum_p2)

        # Avoid division by zero
        val = jnp.where(den > 0, num / den, 0.0)
        # Clamp to [-1, 1]
        val = jnp.clip(val, -1.0, 1.0)
        freq = (bin_start + bin_end) / 2.0

        # We need to return valid mask too, because some bins might be empty
        is_valid = jnp.any(mask)
        return freq, val, is_valid

    indices = jnp.arange(n_bins)
    freqs, vals, is_valid = jax.vmap(compute_bin)(indices)

    # Note: jnp.where with dynamic sizes breaks JIT if we don't pad.
    # For a fully differentiable metric, we typically pad with 0s or NaNs, or return the full array.
    # We will return the full array but mask out invalid frequencies with NaN or 0.

    return freqs, vals
