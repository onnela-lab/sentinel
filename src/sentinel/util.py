import contextlib
from datetime import datetime, timedelta
import importlib.util
import jax
from jax import numpy as jnp
import logging
import numpyro
from numpyro import distributions as dists
import optax
import pathlib
from types import ModuleType
from typing import Any, Callable, Type


def assert_distribution_shapes(
    expected_shapes: dict[str, tuple], model: Callable, rng_seed: int | None = None
) -> Callable:
    """
    Transform a numpyro model to a function that asserts the shapes of the sample sites
    match the expected batch and event shapes.

    Args:
        expected_shapes: Dictionary mapping sample site names to tuples of batch and
            event shapes.
        model: Numpyro model.
        rng_seed: Optional random number generator seed.

    Returns:
        Wrapped model that asserts the shapes of the sample sites match the expected
        batch and event shapes.
    """
    if rng_seed is not None:
        model = numpyro.handlers.seed(model, rng_seed=rng_seed)

    def _wrapped(*args, **kwargs):
        with numpyro.handlers.trace() as trace:
            model(*args, **kwargs)

        actual_shapes = {
            name: (
                (site["fn"].batch_shape, site["fn"].event_shape)
                if site["type"] == "sample"
                else jnp.shape(site["value"])
            )
            for name, site in trace.items()
            if site["type"] in {"deterministic", "sample"}
        }

        missing_shapes = set(actual_shapes) - set(expected_shapes)
        assert not missing_shapes, (
            f"found {len(missing_shapes)} missing shapes: {missing_shapes}"
        )

        extra_shapes = set(expected_shapes) - set(actual_shapes)
        assert not extra_shapes, (
            f"found {len(extra_shapes)} extra shapes: {extra_shapes}"
        )

        for name, shapes in actual_shapes.items():
            site = trace[name]
            if site["type"] == "sample":
                assert len(shapes) == 2 and all(
                    isinstance(shape, tuple) for shape in shapes
                ), (
                    f"Expected a tuple of tuples for actual shapes but got {shapes} "
                    f"for stochastic site `{name}`."
                )
                (actual_batch_shape, actual_event_shape) = shapes
                assert len(expected_shapes[name]) == 2 and all(
                    isinstance(shape, tuple) for shape in expected_shapes[name]
                ), (
                    "Expected a tuple of tuples for expected shapes but got "
                    f"{expected_shapes[name]} for stochastic site `{name}`."
                )
                expected_batch_shape, expected_event_shape = expected_shapes[name]
                assert (
                    actual_batch_shape == expected_batch_shape
                    and actual_event_shape == expected_event_shape
                ), (
                    f"Expected batch shape {expected_batch_shape} and event shape "
                    f"{expected_event_shape} but got batch shape {actual_batch_shape} "
                    f"and event shape {actual_event_shape} for stochastic site "
                    f"`{name}`."
                )
            elif site["type"] == "deterministic":
                actual_shape = shapes
                expected_shape = expected_shapes[name]
                assert actual_shape == expected_shape, (
                    f"Expected shape {expected_shape} but got {actual_shape} for "
                    f"deterministic site `{name}`."
                )
            else:
                raise NotImplementedError(site["type"])

    return _wrapped


def merge_dicts(*args: dict) -> dict:
    """
    Merge dictionaries, checking for duplicate keys.

    Args:
        *args: Dictionaries to merge.

    Returns:
        Merged dictionary.
    """
    seen = set()
    for arg in args:
        duplicates = seen & set(arg)
        if duplicates:
            raise ValueError(
                f"Dictionaries have duplicate keys: {', '.join(duplicates)}."
            )
        seen.update(arg)
    return {key: value for arg in args for key, value in arg.items()}


class Tracker:
    """
    Track values and changes.

    Args:
        orders: Dictionary mapping variable names to the order of changes to track.
        func: Function to evaluate changes which must take two arguments (the previous
            and current values) and return a measure of change.
        **order_kwargs: Keyword arguments mapping variable names to the order of changes
            to track.
    """

    def __init__(
        self,
        orders: dict[str, int] | None = None,
        *,
        func: Callable | None = None,
        **order_kwargs,
    ) -> None:
        orders = merge_dicts(orders or {}, order_kwargs)
        assert orders, "Orders must be specified."
        for key, value in orders.items():
            assert isinstance(value, int) and value >= 0, (
                f"Order must be non-negative integers; got {value} for {key}."
            )
        self.orders = orders
        self.state: dict[str, list[jnp.ndarray]] = {}
        self.func = func or (lambda x, y: y - x)

    def update(self, values: dict | None = None, **value_kwargs) -> None:
        """
        Update values and changes.

        Args:
            values: Dictionary mapping variable names to values. Values that are
                :code:`None` are silently dropped.
            **kwargs: Keyword arguments mapping variable names to values.
        """
        # Consolidate values from the optional dictionary and keyword arguments.
        assert values is None or isinstance(values, dict), (
            f"`values` must be `None` or a dictionary; got {values}."
        )
        values = merge_dicts(values or {}, value_kwargs)

        # Update the values if they are not None.
        for key, value in values.items():
            if value is None:
                continue
            order = self.orders.get(key, None)
            if order is None:
                raise RuntimeError("Order must be specified for all variables.")
            previous = self.state.get(key, [])
            current = [value]
            for i in range(min(order, len(previous))):
                current.append(jax.tree.map(self.func, previous[i], current[i]))
            self.state[key] = current

    def get(self, key, order: int = 0, default: Any = None) -> jnp.ndarray | None:
        """
        Get the values or changes in values for a given order.

        Args:
            key: Variable name.
            order: Order of changes to get.
            default: Default value to return if the key is not found.

        Returns:
            Value of the `order`-th change for the variable `key`.
        """
        sequence = self.state.get(key)
        if sequence and order < len(sequence):
            return sequence[order]
        return default


def _normalize_tree_key(key: Any) -> Any:
    if isinstance(key, jax.tree_util.DictKey):
        return key.key
    elif isinstance(key, jax.tree_util.SequenceKey):
        return key.idx
    raise NotImplementedError(f"Key normalization not implemented for {key}.")


def add_nested_events(
    func: Callable, values: Any, sep: str = "/", prefix: str | None = None, **kwargs
) -> None:
    """
    Add summaries from a nested structure to a tensorboard writer.

    Args:
        func: :code:`add_*` function of a summary writer.
        values: Nested structure of scalars.
        sep: Separator for the path of the scalar.
        prefix: Optional prefix for the path of the scalar.
        **kwargs: Keyword arguments passed to `writer.add_scalar`.
    """
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(values)
    for path, value in leaves_with_path:
        if value is None:
            continue
        path = sep.join([str(_normalize_tree_key(key)) for key in path])
        if prefix:
            path = f"{prefix}{sep}{path}"
        func(path, value, **kwargs)


def get_distribution_args_from_quantiles(
    dist_cls: Type[dists.Distribution],
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    q1: jnp.ndarray,
    q2: jnp.ndarray,
    arg_names: list[str] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> dict[str, jnp.ndarray]:
    """
    Evaluate the parameters of a distribution with two parameters such that
    :code:`cdf(x1) == q1` and :code:`cdf(x2) == q2`.

    Args:
        dist_cls: Distribution class.
        x1: Lower value.
        x2: Upper value.
        q1: Lower quantile.
        q2: Upper quantile.
        arg_names: Argument names to optimize, defaults to
            :code:`dist_cls.arg_constraints`.
        max_iter: Maximum number of iterations to run.
        tol: Tolerance for convergence.

    Returns:
        Dictionary of distribution parameters.
    """

    def _biject_to(params: dict) -> dict:
        # Biject the parameters to the constraints of the distribution.
        return {
            key: dists.biject_to(dist_cls.arg_constraints[key])(value)
            for key, value in params.items()
        }

    def _loss(params: dict) -> jnp.ndarray:
        # Evaluate the L2 distance between the target and actual quantiles.
        dist = dist_cls(**_biject_to(params))
        return (dist.cdf(x1) - q1) ** 2 + (dist.cdf(x2) - q2) ** 2

    # Initialize the parameters.
    arg_names = arg_names or dist_cls.arg_constraints
    params = {key: jnp.zeros(()) for key in arg_names}

    # Set up the optimizer and jitted update function.
    solver = optax.lbfgs()
    state = solver.init(params)
    value_and_grad = optax.value_and_grad_from_state(_loss)

    @jax.jit
    def _update(state, params):
        value, grads = value_and_grad(params, state=state)
        updates, state = solver.update(
            grads, state, params, value=value, grad=grads, value_fn=_loss
        )
        params = optax.apply_updates(params, updates)
        return state, params, grads

    # Run the optimization.
    for _ in range(max_iter):
        state, params, grads = _update(state, params)

        # Return the projected parameters if converged.
        if all(jnp.abs(grad) < tol for grad in grads.values()):
            return _biject_to(params)

    raise RuntimeError(f"Maximum number of iterations ({max_iter}) exceeded.")


def get_distribution_from_quantiles(
    dist_cls: Type[dists.Distribution],
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    q1: jnp.ndarray,
    q2: jnp.ndarray,
    arg_names: list[str] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> dists.Distribution:
    args = get_distribution_args_from_quantiles(
        dist_cls, x1, x2, q1, q2, arg_names, max_iter, tol
    )
    return dist_cls(**args)


def load_module(path: str | pathlib.Path) -> ModuleType:
    """
    Load a module from a file path.

    Args:
        path: Path to the Python file.

    Returns:
        Python module loaded from :code:`path`.
    """
    path = pathlib.Path(path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None:
        raise ImportError(f"Failed to create module spec for {path}.")

    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Failed to get module loader for {path}.")
    spec.loader.exec_module(module)

    return module


@contextlib.contextmanager
def atomic_open(path: str | pathlib.Path, mode: str = "w"):
    """
    Atomically open a file for writing by writing to a temporary file and renaming it.

    Args:
        path: Path to open for writing.
        mode: Mode in which to open the file.
    """
    path = pathlib.Path(path)
    tmp = path.with_suffix(".atomic")
    with open(tmp, mode) as fp:
        yield fp
    tmp.rename(path)


class Timer:
    def __init__(self) -> None:
        self.start = None
        self.end = None

    def __enter__(self) -> "Timer":
        self.start = datetime.now()
        return self

    def __exit__(self, *_) -> None:
        self.end = datetime.now()

    @property
    def duration(self) -> timedelta:
        assert self.start is not None, "Start the timer using a context manager."
        end = datetime.now() if self.end is None else self.end
        return end - self.start


@contextlib.contextmanager
def log_duration(logger: logging.Logger, msg: str, level: int = logging.INFO):
    start = datetime.now()
    yield
    duration = datetime.now() - start
    logger.log(level, f"{msg} in {duration}.")
