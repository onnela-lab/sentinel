import argparse
import functools
import jax
from jax import numpy as jnp
import logging
import numpyro
import os
import pandas as pd
import pathlib
import shutil
import tensorboardX
from time import time
from tqdm import tqdm
from ..util import (
    add_nested_events,
    load_module,
    log_duration,
    Timer,
    Tracker,
)
from .config import BaseConfig
from .util import dump_state, Result


class _Args:
    config: pathlib.Path
    output: pathlib.Path
    data: pathlib.Path
    n_epochs: int
    seed: int
    checkpoint_every: float
    log_every: int


def __main__(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-every", help="Checkpoint every N seconds.", type=float
    )
    parser.add_argument("--log-every", help="Log every N epochs.", type=int)
    parser.add_argument("--n-epochs", help="Override number of epochs.", type=int)
    parser.add_argument("--seed", help="Random seed.", type=int, default=42)
    parser.add_argument(
        "--config",
        help="Path to Python configuration file which must declare a `setup` method.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--data",
        help="Path to data.",
        type=pathlib.Path,
    )
    parser.add_argument("output", help="Output directory.", type=pathlib.Path)
    args: _Args = parser.parse_args(argv)

    # Set up logging.
    logger = logging.getLogger("sentinel.fit")
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Set up the output directory.
    args.output.mkdir(parents=True, exist_ok=True)

    # Abort if we've already completed the run.
    final_path = args.output / "final.pkl"
    if final_path.is_file():
        logger.info(f"Final checkpoint `{final_path}` already exists; exiting.")
        return

    # Get the configuration and overwrite values that are explicitly specified.
    config_path = args.config
    if config_path is None:
        config_path = args.output / "config.py"
    if not config_path.is_file():
        raise ValueError("Specify a configuration file using the `--config` switch.")
    try:
        config: BaseConfig = load_module(config_path).setup()
    except Exception as ex:
        raise RuntimeError(
            f"Failed to load configuration from `{config_path.resolve()}`: {ex}."
        ) from ex
    overrides = {
        "checkpoint_every": args.checkpoint_every,
        "n_epochs": args.n_epochs,
        "seed": args.seed,
        "log_every": args.log_every,
    }
    if "CI" in os.environ and overrides["n_epochs"] is None:
        logger.warning("Running in CI environment; reducing number of epochs.")
        overrides["n_epochs"] = 1
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    if config.enable_x64:
        numpyro.enable_x64()

    # Load the data and separate into static and traced arguments. We don't use the aux
    # data directly but dump it as part of checkpoints.
    data_path = args.data
    if data_path is None:
        data_path = args.output / "data"
    if not data_path.exists():
        raise ValueError("Specify a data path using the `--data` switch.")
    data, aux = config.load_data(data_path)
    model = config.get_model()
    static_args = model.get_static_args(**data)
    traced_args = model.get_traced_args(**data)

    # Possibly reparameterize after getting all the arguments we need.
    model = config.reparam_model(model, static_args | traced_args)

    # Render the model.
    try:
        numpyro.render_model(
            model,
            model_kwargs=static_args | traced_args,
            filename=args.output / "model.pdf",
            render_distributions=True,
        )
    except ImportError:
        pass

    # We need to initialize the SVI object no matter what because it's stateful and
    # cannot be serialized.
    key = jax.random.key(config.seed)
    key, init_key = jax.random.split(key)
    guide = config.get_guide(model, static_args | traced_args)
    svi = numpyro.infer.SVI(
        model,
        guide,
        config.get_optim(),
        numpyro.infer.TraceMeanField_ELBO(),
        **static_args,
    )
    with log_duration(logger, "Initialized SVI state"):
        state = svi.init(init_key, **traced_args)

    # Now that the state is initialized, the guide is initialized.
    try:
        numpyro.render_model(
            guide,
            model_kwargs=static_args | traced_args,
            filename=args.output / "guide.pdf",
            render_distributions=True,
        )
    except ImportError:
        pass

    def _get_params_and_median(state):
        params = svi.get_params(state)
        median = guide.median(params)
        return params, median

    # Load the checkpoint if it exists.
    checkpoint_path = args.output / "checkpoint.pkl"
    if checkpoint_path.is_file():
        result: Result = pd.read_pickle(checkpoint_path)
        state = result.state
        key = result.key
        last_checkpoint_at = time()
        logger.info(f"Loaded checkpoint from `{checkpoint_path}`.")
    else:
        # Set the last checkpoint time to zero so we immediately dump the state after
        # the first epoch.
        last_checkpoint_at = 0
    last_tensorboard_log = -float("inf")

    # Figure out the number of epochs (full-batch: 1 step per epoch).
    n_steps_per_epoch = 1
    n_epochs = config.n_epochs
    step = int(state.optim_state[0])
    epoch = step // n_steps_per_epoch
    if epoch > n_epochs:
        raise RuntimeError(
            f"Current epoch number ({epoch}) exceeds the total number of epochs "
            f"({n_epochs}). You are likely using a checkpoint from a different run."
        )

    delta_tracker = Tracker(loss=1)
    summary_writer = tensorboardX.SummaryWriter(args.output, flush_secs=30)

    @jax.jit
    def _run_one_epoch(carry, _):
        """
        Run one epoch (full-batch update).
        """
        key, state = carry
        key, _ = jax.random.split(key)
        state, loss = svi.update(state, **traced_args)
        statistics = {
            "loss": loss,
            "step": state.optim_state[0],
        }
        return (key, state), statistics

    @functools.partial(jax.jit, static_argnames=["n"])
    def _run_one_scan(carry, n):
        return jax.lax.scan(_run_one_epoch, carry, jnp.empty(n))

    try:
        with tqdm(range(n_epochs), initial=epoch) as progress, summary_writer:
            while epoch < n_epochs:
                with Timer() as epoch_timer:
                    # Run either a single epoch or scan over epochs.
                    with Timer() as scan_timer:
                        if config.scan_size == 1:
                            (key, state), statistics = _run_one_epoch(
                                (key, state), None
                            )
                            statistics = {
                                key: value[None] for key, value in statistics.items()
                            }
                        else:
                            (key, state), statistics = _run_one_scan(
                                (key, state), min(config.scan_size, n_epochs - epoch)
                            )

                    if jnp.any(~jnp.isfinite(statistics["loss"])):
                        raise ValueError("non-finite loss value encountered.")

                    # Report summary statistics.
                    last_step = step
                    for step, loss in zip(statistics["step"], statistics["loss"]):
                        # We need to convert explicitly convert to int because
                        # tensorboardx does not handle jax types correctly.
                        step = int(step)
                        delta_tracker.update(loss=loss)
                        delta = delta_tracker.get("loss", 1)
                        add_nested_events(
                            summary_writer.add_scalar,
                            {
                                "value": loss,
                                # We report the negative delta so we can log-scale in
                                # tensorboard.
                                "delta": None if delta is None else -delta,
                            },
                            global_step=step,
                            prefix="loss",
                        )
                    epoch = step // n_steps_per_epoch
                    summary_writer.add_scalar(
                        "iter_per_second/scan_only",
                        (step - last_step) / scan_timer.duration.total_seconds(),
                        global_step=step,
                    )

                    # Update statistics with epoch-level summaries.
                    state, epoch_statistics = config.evaluate_epoch_statistics(state)

                    # Log more complex events.
                    if epoch >= last_tensorboard_log + config.log_every:
                        params, median = _get_params_and_median(state)
                        add_nested_events(
                            summary_writer.add_histogram,
                            {
                                "params": params,
                                "median": median,
                            },
                            global_step=step,
                        )
                        last_tensorboard_log = epoch

                    # Write a checkpoint if it's time.
                    if time() - last_checkpoint_at > config.checkpoint_every:
                        dump_state(
                            checkpoint_path,
                            svi,
                            state,
                            key,
                            epoch,
                            aux,
                            epoch_statistics,
                        )
                        last_checkpoint_at = time()
                        logger.info(
                            f"Wrote checkpoint to `{checkpoint_path}` at epoch {epoch}."
                        )

                    # Update the progress bar.
                    progress.set_description(f"loss={loss:.5f}")
                    progress.update(len(statistics["step"]))

                summary_writer.add_scalar(
                    "iter_per_second/epoch",
                    (step - last_step) / epoch_timer.duration.total_seconds(),
                    global_step=step,
                )
    except KeyboardInterrupt:
        dump_state(checkpoint_path, svi, state, key, epoch, aux, epoch_statistics)
        logger.warning(
            f"Detected keyboard interrupt; wrote checkpoint to `{checkpoint_path}` at "
            f"epoch {epoch}; exiting."
        )
        raise

    # Write the final state to disk. Also copy the final state to the checkpoint in case
    # we want to later restart with more epochs.
    dump_state(final_path, svi, state, key, epoch, aux, epoch_statistics)
    shutil.copy(final_path, checkpoint_path)
    logger.info(f"Wrote final result to `{final_path}`; exiting.")


if __name__ == "__main__":
    __main__()
