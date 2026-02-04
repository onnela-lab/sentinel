import functools
import ifnt
from jax import numpy as jnp
import numpyro
import optax
import pytest
from sentinel.models import (
    BaseModel,
    GaussianProcessCollapsedModel,
)
from sentinel.util import assert_distribution_shapes
from typing import Type


@pytest.fixture(params=[None, "independent", "sum_to_zero"])
def coefficient_prior(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=[(13, 17, 23), (23, 13, 17), (17, 23, 13)], ids=str)
def sizes(request: pytest.FixtureRequest) -> tuple:
    return request.param


@pytest.fixture
def simulated_data(sizes: tuple) -> dict:
    # Generate random data.
    rng = ifnt.random.JaxRandomState(19)
    n_locs, n_types, n_weeks = sizes
    n_obs = 123
    n_features = 5
    loc_id = rng.randint((n_obs,), 0, n_locs)
    type_id = rng.randint((n_obs,), 0, n_types)
    week_id = rng.randint((n_obs,), 0, n_weeks)
    target = rng.normal((n_obs,))
    features = rng.normal((n_obs, n_features))
    data = {
        "n_locs": n_locs,
        "n_types": n_types,
        "n_weeks": n_weeks,
        "n_obs": n_obs,
        "loc_id": loc_id,
        "type_id": type_id,
        "week_id": week_id,
        "target": target,
        "n_features": n_features,
        "features": features,
    }
    return data


def test_base_model_locked() -> None:
    class Model(BaseModel):
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            super().__init__()

    model = Model(foo="bar")
    assert model.foo == "bar"
    with pytest.raises(RuntimeError):
        model.foo = "baz"
    with pytest.raises(RuntimeError):
        del model.foo


@pytest.mark.parametrize(
    "model_cls",
    [
        functools.partial(
            GaussianProcessCollapsedModel, kernel="matern32", method="fft", dof=None
        ),
        functools.partial(
            GaussianProcessCollapsedModel, kernel="matern32", method="fft", dof=2
        ),
    ],
)
@pytest.mark.parametrize(
    "guide_cls",
    [numpyro.infer.autoguide.AutoDiagonalNormal, numpyro.infer.autoguide.AutoDelta],
)
def test_model_end_to_end(
    model_cls: Type[BaseModel],
    simulated_data: dict,
    coefficient_prior: str,
    guide_cls: Type[numpyro.infer.autoguide.AutoGuide],
) -> None:
    if (
        guide_cls is numpyro.infer.autoguide.AutoDelta
        and coefficient_prior == "sum_to_zero"
    ):
        pytest.skip("AutoDelta has a bug and doesn't handle constraints properly.")
    # Validate shapes and check model is locked.
    rng = ifnt.random.JaxRandomState(0)
    model = model_cls(coefficient_prior=coefficient_prior)
    args = model.get_args(**simulated_data)
    assert_distribution_shapes(model.get_expected_shapes(**args), model, rng.get_key())(
        **args
    )

    # Check SVI initialization and one update step.
    guide = guide_cls(model)
    optim = optax.adam(1e-2)
    loss_fn = numpyro.infer.Trace_ELBO()
    svi = numpyro.infer.SVI(
        model, guide, optim, loss_fn, **model.get_static_args(**simulated_data)
    )
    traced_args = model.get_traced_args(**simulated_data)
    state = svi.init(rng.get_key(), **traced_args)
    state1, loss1 = svi.update(state, **traced_args)
    assert jnp.isfinite(loss1)
