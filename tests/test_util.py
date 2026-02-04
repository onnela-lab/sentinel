import ifnt
from jax import numpy as jnp
import numpyro
from numpyro import distributions as dists
import pathlib
import pytest
import tensorboardX
from sentinel import util
from unittest import mock


def test_assert_valid_shapes() -> None:
    def model(sample_y) -> None:
        with numpyro.plate("n", 10):
            x = numpyro.sample(
                "x", numpyro.distributions.Normal(0, 1).expand((7,)).to_event()
            )
            if sample_y:
                numpyro.deterministic("y", x[..., None] + jnp.ones(13))

    # Check for missing and extra shapes.
    with pytest.raises(AssertionError, match="found 2 missing shapes"):
        util.assert_distribution_shapes({}, model, rng_seed=9)(True)
    with pytest.raises(AssertionError, match="found 1 extra shapes"):
        util.assert_distribution_shapes(
            {"x": None, "y": None, "z": None}, model, rng_seed=9
        )(True)

    # Pre-seed the model and check for the correct outcome.
    model = numpyro.handlers.seed(model, rng_seed=9)
    util.assert_distribution_shapes({"x": ((10,), (7,)), "y": (10, 7, 13)}, model)(True)

    # Check incorrect batch and event shapes.
    with pytest.raises(AssertionError):
        util.assert_distribution_shapes({"x": ((9,), (7,))}, model)(False)
    with pytest.raises(AssertionError):
        util.assert_distribution_shapes({"x": ((10,), (7, 8))}, model)(False)


def test_tracker() -> None:
    tracker = util.Tracker({"x": 2}, y=1)
    for i in range(0, 20, 2):
        tracker.update(x=i)
        assert tracker.get("x") == i
        dx = tracker.get("x", 1)
        assert dx is None or dx == 2
        ddx = tracker.get("x", 2)
        assert ddx is None or ddx == 0
        assert tracker.get("x", 3) is None

        tracker.update(y={"a": i})
        assert tracker.get("x") == i, "x should not have changed"
        assert tracker.get("y") == {"a": i}
        dy = tracker.get("y", 1)
        assert dy is None or dy == {"a": 2}


def test_merge_dicts() -> None:
    assert util.merge_dicts({"a": 1}, {"b": 2}, {"c": 3}) == {"a": 1, "b": 2, "c": 3}
    with pytest.raises(ValueError):
        util.merge_dicts({"a": 1}, {"a": 2})


def test_add_nested_events(tmp_path) -> None:
    scalars = {
        "a": [3, 4],
        "b": {
            "c": 9,
        },
    }
    writer = tensorboardX.SummaryWriter(tmp_path)
    with mock.patch("tensorboardX.SummaryWriter.add_scalar") as add_scalar:
        util.add_nested_events(
            writer.add_scalar, scalars, prefix="foo", sep=".", global_step=3
        )

    add_scalar.assert_called_with("foo.b.c", 9, global_step=3)


def test_get_distribution_args_from_quantiles() -> None:
    x1, x2, q1, q2 = 0.5, 20.5, 0.025, 0.975
    tol = 1e-6
    args = util.get_distribution_args_from_quantiles(dists.InverseGamma, x1, x2, q1, q2)
    dist = dists.InverseGamma(**args)
    ifnt.testing.assert_allclose(dist.cdf(x1), q1, atol=tol)
    ifnt.testing.assert_allclose(dist.cdf(x2), q2, atol=tol)


def test_load_module(tmp_path: pathlib.Path) -> None:
    module_path = tmp_path / "module.py"
    module_path.write_text("hello = 'world'")
    module = util.load_module(module_path)
    assert module.hello == "world"
