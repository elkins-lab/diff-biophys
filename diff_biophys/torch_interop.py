from collections.abc import Callable
from typing import Any, cast

try:
    import jax
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def jax_to_torch(jax_fn: Callable) -> Callable:
    """
    Wrap a JAX function in a PyTorch autograd.Function to allow seamless interoperability
    between PyTorch deep learning models and JAX biophysics kernels.

    The wrapper uses DLPack to perform zero-copy memory transfers between the
    PyTorch and JAX tensor representations on the GPU.
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch must be installed to use jax_to_torch. Run `pip install diff-biophys[torch]`"
        )

    class JaxToTorchWrapper(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *args: torch.Tensor) -> torch.Tensor:
            # Convert PyTorch tensors to JAX arrays via DLPack natively
            jax_args = [jax.dlpack.from_dlpack(x) for x in args]

            # Compute the forward pass and its VJP (Vector-Jacobian Product) function
            y, vjp_fn = jax.vjp(jax_fn, *jax_args)
            ctx.vjp_fn = vjp_fn

            # Convert JAX array back to PyTorch tensor
            return torch.from_dlpack(y)

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
            # Convert PyTorch gradient to JAX array
            jax_grad_output = jax.dlpack.from_dlpack(grad_output)

            # Compute the backward pass
            jax_grads = ctx.vjp_fn(jax_grad_output)

            # Convert JAX gradients back to PyTorch tensors
            torch_grads = [torch.from_dlpack(g) for g in jax_grads]
            return tuple(torch_grads)

    def wrapper(*args: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, JaxToTorchWrapper.apply(*args))

    return wrapper
