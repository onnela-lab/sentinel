from jax import numpy as jnp
import logging
import numpyro
from numpyro import distributions as dists
from numpyro.distributions import CirculantNormal
from scipy.fft import next_fast_len
from typing import Literal, Optional
from .base import Model
from .. import kernels
from ..util import get_distribution_from_quantiles


LOGGER = logging.getLogger(__name__)


class CollapsedModel(Model):
    """
    Model with collapsed intercept parameters :code:`mu`, :code:`a`, and :code:`b`.
    """


class GaussianProcessCollapsedModel(CollapsedModel):
    """
    :cls:`~.CollapsedModel` with Gaussian process priors for the temporal sequences.
    """

    def __init__(
        self,
        *,
        coefficient_prior: Literal[None, "independent", "sum_to_zero"],
        kernel: Literal["matern32"],
        method: Literal["fft"],
        dof: Optional[float],
    ) -> None:
        self.kernel = kernel
        self.method = method
        super().__init__(coefficient_prior=coefficient_prior, dof=dof)

    def sample_effects(
        self,
        *,
        intercept_name: str,
        sequence_name: str,
        sigma_intercept: jnp.ndarray,
        n_weeks: int,
        n_weeks_padded: int,
        length_scale_prior: dists.Distribution,
        **kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        length_scale = numpyro.sample(
            f"length_scale_{sequence_name}", length_scale_prior
        )
        kappa = numpyro.deterministic(f"kappa_{sequence_name}", 3)
        sigma = numpyro.sample(f"sigma_{sequence_name}", dists.HalfNormal(kappa))

        # Expected shapes.
        batch_shape = sigma.shape

        # Evaluate the kernels.
        eps = 1e-6
        if self.kernel == "matern32":
            evaluate_kernel_rfft = kernels.evaluate_matern32_rfft
        else:
            raise ValueError(self.kernel)

        kernel_rfft = (
            evaluate_kernel_rfft(
                sigma[..., None],
                length_scale[..., None],
                n_weeks_padded,
                n_weeks_padded,
            )
            .at[..., 0]
            .add(n_weeks_padded * sigma_intercept**2)
            + eps
        )
        dist = CirculantNormal(jnp.zeros(n_weeks_padded), covariance_rfft=kernel_rfft)
        event_shape = (n_weeks_padded,)

        assert dist.batch_shape == batch_shape
        assert dist.event_shape == event_shape
        sequence = numpyro.sample(sequence_name, dist)
        assert sequence.shape == dist.shape()
        return None, sequence

    def get_expected_shapes(
        self, *, n_locs, n_types, n_weeks, n_weeks_padded, n_obs, **kwargs
    ):
        shapes = super().get_expected_shapes(
            n_locs=n_locs, n_types=n_types, n_weeks=n_weeks, n_obs=n_obs, **kwargs
        )
        shapes.update(
            {
                "length_scale_z": ((), ()),
                "length_scale_A": ((n_locs,), ()),
                "length_scale_B": ((n_types,), ()),
                "z": ((), (n_weeks_padded,)),
                "A": ((n_locs,), (n_weeks_padded,)),
                "B": ((n_types,), (n_weeks_padded,)),
            }
        )
        return shapes

    def get_static_args(
        self,
        *,
        n_locs: int,
        n_types: int,
        n_weeks: int,
        n_obs: int,
        n_features: int = None,
        **kwargs,
    ):
        length_scale_prior = get_distribution_from_quantiles(
            dists.Gamma, 2, n_weeks / 2, 0.01, 0.99
        )
        LOGGER.info(
            "Using length scale prior: %s with concentration %f and rate %f.",
            length_scale_prior,
            length_scale_prior.concentration,
            length_scale_prior.rate,
        )
        # Pad the number of weeks to overcome the periodic boundary conditions and
        # expand to the next fast FFT length.
        n_weeks_padded = next_fast_len(int(4 * n_weeks / 3))
        return super().get_static_args(
            n_locs=n_locs,
            n_types=n_types,
            n_weeks=n_weeks,
            n_obs=n_obs,
            n_features=n_features,
            **kwargs,
        ) | {"length_scale_prior": length_scale_prior, "n_weeks_padded": n_weeks_padded}
