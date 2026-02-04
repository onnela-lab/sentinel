import ifnt
from jax import numpy as jnp
import numpyro
from numpyro import distributions as dists
from typing import Any, Optional, Literal
from ..util import merge_dicts


class BaseModel:
    """
    Base class for models as a collection of jit-able, (almost) pure class methods. The
    methods are effectively pure provided the instance is not modified after
    declaration. We prevent modification post-declaration by overriding
    :meth:`__setattr_` and :meth:`__delattr__`.
    """

    def __init__(self) -> None:
        self._locked = True

    def __setattr__(self, name, value):
        if getattr(self, "_locked", False):
            raise RuntimeError(
                f"Cannot set attribute `{name}` on locked class `{self}`."
            )
        else:
            self.__dict__[name] = value

    def __delattr__(self, name):
        if getattr(self, "_locked", False):
            raise RuntimeError(
                f"Cannot delete attribute `{name}` on locked class `{self}`."
            )
        else:
            del self.__dict__[name]

    def get_expected_shapes(self, **kwargs) -> dict[str, tuple]:
        """
        Get the expected shapes of stochastic sites as
        :code:`(batch_shape, event_shape)` and deterministic sites as
        :code:`tensor_shape`.

        Args:
            kwargs: Keyword arguments from :meth:`get_static_kwargs`.
        """
        raise NotImplementedError

    def get_static_args(self, **kwargs) -> dict[str, Any]:
        """
        Get static arguments for :meth:`model`, such as tensor shapes.
        """
        raise NotImplementedError

    def get_traced_args(self, **kwargs) -> dict[str, jnp.ndarray]:
        """
        Get traced arguments for :meth:`__call__`, such as data to be mini-batched.
        """
        raise NotImplementedError

    def get_args(self, **kwargs) -> dict[str, Any]:
        """
        Get the union of static and traced arguments.
        """
        return merge_dicts(
            self.get_static_args(**kwargs), self.get_traced_args(**kwargs)
        )

    def __call__(self, **kwargs) -> None:
        """
        Numpyro model as a pure function (except for relying on :code:`self` for
        inheritance).
        """
        raise NotImplementedError


class Model(BaseModel):
    """
    Tensor decomposition model.

    The naming convention is :code:`sigma_*` for scale parameters of effects and
    coefficients, and :code:`kappa_*` for scale parameters of distributions for
    :code:`sigma_*` scale parameters.

    Attr:
        coefficient_prior: Prior for regression coefficients. :code:`None` to exclude
            features. :code:`"independent"` for standard hierarchical priors.
            :code:`"sum_to_zero"` for zero-sum priors on location and type coefficients.
        dof: Degree of freedoms of Student-T distribution for observations. :code:`None`
            represents a normal distribution.
    """

    def __init__(
        self,
        *,
        coefficient_prior: Literal[None, "independent", "sum_to_zero"],
        dof: Optional[float],
    ) -> None:
        self.coefficient_prior = coefficient_prior
        self.dof = dof
        super().__init__()

    def sample_effects(
        self,
        *,
        intercept_name: str,
        sequence_name: str,
        sigma_intercept: jnp.ndarray,
        n_weeks: int,
        **kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample the intercept and temporal sequence.

        Args:
            intercept_name: Name of the intercept parameter.
            sequence_name: Name of the temporal sequence parameter.
            sigma_intercept: Scale of the intercept parameter.
            n_weeks: Number of weeks.
            **kwargs: Other parameters passed to the model function.

        Returns:
            Tuple comprising intercept and sequence parameters. The intercept is ignored
            if :code:`None` is returned, but the sequence is required.
        """
        raise NotImplementedError

    def __call__(
        self,
        *,
        n_locs: int,
        n_types: int,
        n_weeks: int,
        n_obs: int,
        n_features: int = None,
        loc_id: jnp.ndarray,
        type_id: jnp.ndarray,
        week_id: jnp.ndarray,
        target: jnp.ndarray,
        features: jnp.ndarray = None,
        **kwargs,
    ) -> None:
        # Sample overall effects.
        sigma_mu = numpyro.deterministic("sigma_mu", jnp.array(3.0))
        mu, z = self.sample_effects(
            intercept_name="mu",
            sequence_name="z",
            sigma_intercept=sigma_mu,
            n_weeks=n_weeks,
            **kwargs,
        )

        # Sample location-specific effects.
        kappa_a = numpyro.deterministic("kappa_a", 3.0)
        sigma_a = numpyro.sample("sigma_a", dists.HalfNormal(kappa_a))
        locs_plate = numpyro.plate("locs", n_locs)
        with locs_plate:
            a, A = self.sample_effects(
                intercept_name="a",
                sequence_name="A",
                sigma_intercept=sigma_a,
                n_weeks=n_weeks,
                **kwargs,
            )

        # Sample type-specific effects.
        kappa_b = numpyro.deterministic("kappa_b", 3.0)
        sigma_b = numpyro.sample("sigma_b", dists.HalfNormal(kappa_b))
        types_plate = numpyro.plate("types", n_types)
        with types_plate:
            b, B = self.sample_effects(
                intercept_name="b",
                sequence_name="B",
                sigma_intercept=sigma_b,
                n_weeks=n_weeks,
                **kwargs,
            )

        # Sample location-type interaction effects.
        kappa_C = numpyro.deterministic("kappa_C", 3.0)
        sigma_C = numpyro.sample("sigma_C", dists.HalfNormal(kappa_C))
        C = numpyro.sample(
            "C", dists.Normal(0, sigma_C).expand((n_locs, n_types)).to_event()
        )

        # Sample parameters for fixed effects.
        if self.coefficient_prior:
            # Validate data.
            assert n_features, "At least one feature is required."
            assert features is not None, "Features are required."
            assert features.shape == (loc_id.size, n_features), (
                f"Features must match (n_targets={loc_id.size}, "
                f"n_features={n_features}), got {features.shape}."
            )

            # Global regression coefficients.
            sigma_coef = numpyro.deterministic("sigma_coef", 3.0)
            coef = numpyro.sample(
                "coef", dists.Normal(0, sigma_coef).expand((n_features,)).to_event()
            )

            # Prior scales for location and type coefficients.
            kappa_coef_loc = numpyro.deterministic("kappa_coef_loc", 3.0)
            sigma_coef_loc = numpyro.sample(
                "sigma_coef_loc",
                dists.HalfNormal(kappa_coef_loc).expand((n_features,)).to_event(),
            )

            kappa_coef_type = numpyro.deterministic("kappa_coef_type", 3.0)
            sigma_coef_type = numpyro.sample(
                "sigma_coef_type",
                dists.HalfNormal(kappa_coef_type).expand((n_features,)).to_event(),
            )

            if self.coefficient_prior == "independent":
                # Independent priors are the "vanilla" implementation of regression with
                # possibly strong competition between the global regression
                # coefficients, type- and location-specific coefficients.
                with locs_plate:
                    coef_loc = numpyro.sample(
                        "coef_loc",
                        dists.Normal(0, sigma_coef_loc)
                        .expand((n_locs, n_features))
                        .to_event(1),
                    )
                with types_plate:
                    coef_type = numpyro.sample(
                        "coef_type",
                        dists.Normal(0, sigma_coef_type)
                        .expand((n_types, n_features))
                        .to_event(1),
                    )
            elif self.coefficient_prior == "sum_to_zero":
                # We use zero sum priors for the regression coefficients so they cannot
                # compete with the global regression coefficients. The zero sum
                # transformation applies along the last dimension, but we want the sum
                # in the batch direction to be zero, not in the event dimension. So we
                # sample in the transposed space and then transpose back.
                coef_loc = numpyro.sample(
                    "coef_loc_raw",
                    dists.TransformedDistribution(
                        dists.Normal(0, sigma_coef_loc[:, None])
                        .expand((n_features, n_locs - 1))
                        .to_event(1),
                        dists.transforms.ZeroSumTransform(),
                    ),
                )
                coef_loc = numpyro.deterministic("coef_loc", coef_loc.T)
                ifnt.testing.assert_allclose(coef_loc.sum(axis=0), 0, atol=1e-4)

                coef_type = numpyro.sample(
                    "coef_type_raw",
                    dists.TransformedDistribution(
                        dists.Normal(0, sigma_coef_type[:, None])
                        .expand((n_features, n_types - 1))
                        .to_event(1),
                        dists.transforms.ZeroSumTransform(),
                    ),
                )
                coef_type = numpyro.deterministic("coef_type", coef_type.T)
                ifnt.testing.assert_allclose(coef_type.sum(axis=0), 0, atol=1e-4)
        else:
            assert n_features is None and features is None, (
                "There should be no features."
            )

        # Sample observation noise, make predictions, and observe.
        y_hat = (
            ifnt.index_guard(z)[week_id]
            + ifnt.index_guard(A)[loc_id, week_id]
            + ifnt.index_guard(B)[type_id, week_id]
            + ifnt.index_guard(C)[loc_id, type_id]
        )
        if mu is not None:
            y_hat = y_hat + mu
        if a is not None:
            y_hat = y_hat + ifnt.index_guard(a)[loc_id]
        if b is not None:
            y_hat = y_hat + ifnt.index_guard(b)[type_id]
        if self.coefficient_prior:
            y_hat = y_hat + jnp.vecdot(
                features,
                coef
                + ifnt.index_guard(coef_loc)[loc_id]
                + ifnt.index_guard(coef_type)[type_id],
            )

        kappa_y = numpyro.deterministic("kappa_y", 3.0)
        sigma_y = numpyro.sample(
            "sigma_y", dists.HalfNormal(kappa_y).expand((n_locs, n_types)).to_event()
        )

        y_scale = ifnt.index_guard(sigma_y)[loc_id, type_id]
        if self.dof is None:
            obs_dist = dists.Normal(y_hat, y_scale)
        else:
            obs_dist = dists.StudentT(self.dof, y_hat, y_scale)
        subsample_size = None if target is None else target.size
        with numpyro.plate("obs", n_obs, subsample_size):
            numpyro.sample("y", obs_dist, obs=target)

    def get_expected_shapes(
        self,
        *,
        n_locs: int,
        n_types: int,
        n_weeks: int,
        n_obs: int,
        n_features: int = None,
        **kwargs,
    ) -> dict[str, tuple]:
        shapes = {
            # No "kappa_mu" because there is only a single "mu" parameter.
            "sigma_mu": (),
            # No "mu" because it does not appear in collapsed models.
            "kappa_z": (),
            "sigma_z": ((), ()),
            "z": ((), (n_weeks,)),
            # Location specific effects.
            "kappa_a": (),
            "sigma_a": ((), ()),
            # No "a" because it does not appear in collapsed models.
            "kappa_A": (),
            "sigma_A": ((n_locs,), ()),
            "A": ((n_locs,), (n_weeks,)),
            # Type-specific effects.
            "kappa_b": (),
            "sigma_b": ((), ()),
            # No "b" because it does not appear in collapsed models.
            "kappa_B": (),
            "sigma_B": ((n_types,), ()),
            "B": ((n_types,), (n_weeks,)),
            # Interaction effects.
            "kappa_C": (),
            "sigma_C": ((), ()),
            "C": ((), (n_locs, n_types)),
            # Observations.
            "kappa_y": (),
            "sigma_y": ((), (n_locs, n_types)),
            "y": ((n_obs,), ()),
        }
        if self.coefficient_prior:
            shapes.update(
                {
                    # Global effects.
                    "sigma_coef": (),
                    "coef": ((), (n_features,)),
                    # Location-specific effects.
                    "kappa_coef_loc": (),
                    "sigma_coef_loc": ((), (n_features,)),
                    # Type-specific effects.
                    "kappa_coef_type": (),
                    "sigma_coef_type": ((), (n_features,)),
                }
            )
        if self.coefficient_prior == "sum_to_zero":
            shapes.update(
                {
                    "coef_loc_raw": ((n_features,), (n_locs,)),
                    "coef_loc": (n_locs, n_features),
                    "coef_type_raw": ((n_features,), (n_types,)),
                    "coef_type": (n_types, n_features),
                }
            )
        elif self.coefficient_prior == "independent":
            shapes.update(
                {
                    "coef_loc": ((n_locs,), (n_features,)),
                    "coef_type": ((n_types,), (n_features,)),
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
    ) -> dict[str, Any]:
        static_args = {
            "n_locs": n_locs,
            "n_types": n_types,
            "n_weeks": n_weeks,
            "n_obs": n_obs,
        }
        if self.coefficient_prior:
            static_args["n_features"] = n_features
        return static_args

    def get_traced_args(
        self,
        *,
        loc_id: jnp.ndarray,
        type_id: jnp.ndarray,
        week_id: jnp.ndarray,
        target: jnp.ndarray,
        features: jnp.ndarray = None,
        **kwargs,
    ):
        traced_args = {
            "loc_id": loc_id,
            "type_id": type_id,
            "week_id": week_id,
            "target": target,
        }
        if self.coefficient_prior:
            traced_args["features"] = features
        return traced_args
