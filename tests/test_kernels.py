import ifnt
from jax import numpy as jnp
from sentinel import kernels
import pytest
from typing import Callable


@pytest.mark.parametrize(
    "evaluate, evaluate_rfft",
    [
        (kernels.evaluate_matern32, kernels.evaluate_matern32_rfft),
    ],
)
@pytest.mark.parametrize("n", [1000, 1001])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("batch_shape", [(), (3, 4)])
def test_kernels(
    evaluate: Callable[..., jnp.ndarray],
    evaluate_rfft: Callable[..., jnp.ndarray],
    n: int,
    p: int,
    batch_shape: tuple,
) -> None:
    if p > 1 or batch_shape:
        pytest.skip("Non-trivial shapes not yet implemented.")
    sigma = jnp.broadcast_to(1.7, batch_shape)
    domain = 3.2
    length_scale = jnp.broadcast_to(0.05 * domain, batch_shape)
    # We use a large-ish n here because the approximation is only accurate for large n.
    x = jnp.linspace(0, domain, n, endpoint=False)[:, None]
    kernel = evaluate(sigma, length_scale, x, domain=domain)
    ifnt.testing.assert_circulant(kernel)

    expected_kernel_rfft = evaluate_rfft(sigma, length_scale, n, domain=domain)
    actual_kernel_rfft = jnp.fft.rfft(kernel[0], axis=-1)
    ifnt.testing.assert_allclose(actual_kernel_rfft.imag, 0, atol=1e-6)
    actual_kernel_rfft = actual_kernel_rfft.real
    ifnt.testing.assert_allclose(actual_kernel_rfft, expected_kernel_rfft, atol=1e-4)
