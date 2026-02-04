from jax import numpy as jnp
from jax.scipy.special import gamma


def _evaluate_distance(
    length_scale: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray = None,
    domain: jnp.ndarray = None,
) -> jnp.ndarray:
    if y is None:
        y = x[..., None, :]
    residuals = jnp.abs(x - y)
    if domain is not None:
        residuals = jnp.minimum(residuals, domain - residuals)
    assert residuals.shape[-1] == 1, "https://github.com/jax-ml/jax/issues/26248"
    # return jnp.linalg.norm(residuals / length_scale, axis=-1)
    return jnp.abs(residuals / length_scale)[..., 0]


def evaluate_matern32(
    sigma: jnp.ndarray,
    length_scale: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray = None,
    domain: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    Evaluate the Matern-3/2 kernel.

    Args:
        sigma: Scale parameter.
        length_scale: Length scale parameter.
        x: First input.
        y: Second input (defaults to outer product with :code:`x` along the last
            dimension).
        domain: Size of the domain with periodic boundary conditions.

    Returns:
        Matern-3/2 kernel evaluated at :code:`x` and :code:`y`.
    """
    d = jnp.sqrt(3) * _evaluate_distance(length_scale, x, y, domain)
    return sigma**2 * (1 + d) * jnp.exp(-d)


def _evaluate_matern_rfft(
    nu: jnp.ndarray,
    sigma: jnp.ndarray,
    length_scale: jnp.ndarray,
    n: jnp.ndarray,
    domain: jnp.ndarray,
) -> jnp.ndarray:
    nrfft = n // 2 + 1
    k = jnp.arange(nrfft)
    return (
        sigma**2
        * n
        * jnp.sqrt(2 * jnp.pi / nu)
        * gamma(nu + 0.5)
        / gamma(nu)
        * (1 + 2 / nu * (jnp.pi * length_scale / domain * k) ** 2) ** -(nu + 0.5)
        * length_scale
        / domain
    )


def evaluate_matern32_rfft(
    sigma: jnp.ndarray, length_scale: jnp.ndarray, n: jnp.ndarray, domain: jnp.ndarray
) -> jnp.ndarray:
    """
    Evaluate the power spectrum of the Matern-3/2 kernel.

    Args:
        sigma: Scale parameter.
        length_scale: Length scale parameter.
        n: Number of points.
        domain: Size of the domain with periodic boundary conditions.

    Returns:
        Power spectrum of the Matern-3/2 kernel.
    """
    return _evaluate_matern_rfft(1.5, sigma, length_scale, n, domain)
