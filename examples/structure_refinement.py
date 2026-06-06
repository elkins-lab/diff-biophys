# %% [markdown]
# # Gradient Descent Structure Refinement
# This example demonstrates how to use `diff-biophys` and `optax` to refine a structure
# directly against experimental data using gradient descent.
#
# We will simulate a target "experimental" Cryo-EM map and try to optimize a random
# starting map to maximize the Fourier Shell Correlation (FSC) with the target.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from diff_biophys.cryo_em import compute_fsc

# %% [markdown]
# ## 1. Setup the Target Data
# Let's create a dummy target density map. In a real scenario, this would be your
# experimental Cryo-EM map loaded via `mrcfile`.

# %%
shape = (16, 16, 16)
voxel_size = (1.0, 1.0, 1.0)
key = jax.random.PRNGKey(42)

# Generate a smooth target map
target_map = jax.random.normal(key, shape)
target_map = jax.scipy.ndimage.map_coordinates(target_map, tuple(jnp.indices(shape) * 0.5), order=1)

# %% [markdown]
# ## 2. Initialize the Model
# We'll initialize our "predicted" map with random noise.
# Our goal is to update this map to match the target.

# %%
key, subkey = jax.random.split(key)
params = jax.random.normal(subkey, shape)

# %% [markdown]
# ## 3. Define the Loss Function
# We want to maximize the FSC. Since `optax` minimizes the loss, we return the negative
# sum of the FSC curve across all valid frequency bins.


# %%
@jax.jit
def loss_fn(pred_map: jax.Array, target: jax.Array) -> jax.Array:
    freqs, fsc = compute_fsc(pred_map, target, voxel_size)

    # We only want to sum over valid frequencies (where freqs > 0)
    # The jax FSC function pads invalid bins with 0s for frequencies.
    valid_mask = freqs > 0.0

    # Negative FSC sum for minimization
    return -jnp.sum(jnp.where(valid_mask, fsc, 0.0))


# %% [markdown]
# ## 4. Setup the Optimizer
# We use `optax.adam` with a learning rate of 0.1.

# %%
optimizer = optax.adam(learning_rate=0.1)
opt_state = optimizer.init(params)

# %% [markdown]
# ## 5. Optimization Loop
# We compute the value and gradient of the loss function, and update the parameters.


# %%
@jax.jit
def step(
    params: jax.Array, opt_state: optax.OptState, target: jax.Array
) -> tuple[jax.Array, optax.OptState, jax.Array]:
    loss_value, grads = jax.value_and_grad(loss_fn)(params, target)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


print("Starting Refinement...")
losses = []
for i in range(50):
    params, opt_state, loss_value = step(params, opt_state, target_map)
    losses.append(loss_value)
    if i % 10 == 0:
        print(f"Step {i:03d} | Loss: {loss_value:.4f}")

# %% [markdown]
# ## 6. Results
# As you can see, the negative FSC loss decreases rapidly as the predicted map
# converges to the target map!

# %%
plt.plot(losses)
plt.title("FSC Optimization Loss")
plt.xlabel("Step")
plt.ylabel("Negative FSC Sum")
plt.savefig("fsc_optimization.png")
print("Optimization plot saved to fsc_optimization.png")
