import ifnt
import isoweek
import jax
from jax import numpy as jnp
import pandas as pd
import pathlib
import pytest
import shutil
from sentinel.fit.__main__ import __main__
from sentinel.fit import Result


CONFIG_PATHS = [
    path
    for path in pathlib.Path("configs").glob("**/*.py")
    if not path.name.startswith("_")
]
N_OBS = 123


@pytest.fixture
def simulated_data_path(tmp_path: pathlib.Path) -> pathlib.Path:
    # Generate random data and write to disk.
    rng = ifnt.random.JaxRandomState(19)
    n_locs = 13
    n_types = 17
    n_weeks = 23
    loc_id = rng.randint((N_OBS,), 0, n_locs)
    type_id = rng.randint((N_OBS,), 0, n_types)
    week_id = rng.randint((N_OBS,), 0, n_weeks)
    target = jnp.exp(rng.normal((N_OBS,)))

    # Construct non-numeric identifiers.
    isoweeks = [(isoweek.Week(2016, 9) + i).isoformat() for i in week_id]
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    data_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "location_id": loc_id,
            "location": [f"loc_{alphabet[i]}" for i in loc_id],
            "type_id": type_id,
            "type": [f"type_{alphabet[i]}" for i in type_id],
            "isoweek_id": week_id,
            "isoweek": isoweeks,
            "volume": target,
        }
    ).to_csv(data_path, index=False)
    return data_path


@pytest.mark.parametrize(
    "config_path", CONFIG_PATHS, ids=lambda x: "/".join(x.parts[1:])
)
def test_fit(
    config_path: pathlib.Path, simulated_data_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    output_path = tmp_path / "output"
    args = [
        "--n-epochs=2",
        f"--config={config_path}",
        f"--data={simulated_data_path}",
        output_path,
    ]
    __main__(list(map(str, args)))
    final_path = output_path / "final.pkl"
    assert final_path.is_file()
    result: Result = pd.read_pickle(final_path)
    assert int(result.state.optim_state[0]) == 2
    assert isinstance(result.statistics, dict)
    assert "z" in result.median

    (tensorboard_event_path,) = output_path.glob("events.out.tfevents.*")
    assert tensorboard_event_path.stat().st_size > 10


def test_fit_restart(simulated_data_path: pathlib.Path, tmp_path: pathlib.Path) -> None:
    # Run the first fit with two epochs.
    assert CONFIG_PATHS, "No configuration files found."
    config_path = CONFIG_PATHS[0]
    output_path = tmp_path / "output"

    args = [
        "--n-epochs=2",
        f"--config={config_path}",
        f"--data={simulated_data_path}",
        output_path,
    ]
    __main__(list(map(str, args)))
    final_path = output_path / "final.pkl"
    assert final_path.is_file()
    result: Result = pd.read_pickle(final_path)
    assert int(result.state.optim_state[0]) == 2
    assert jnp.isclose(result.epoch, 2)

    # Move the final state to the checkpoint path. Then run an extra epoch.
    final_path.rename(output_path / "checkpoint.pkl")
    args = [
        "--n-epochs=3",
        f"--config={config_path}",
        f"--data={simulated_data_path}",
        output_path,
    ]
    __main__(list(map(str, args)))
    assert final_path.is_file()
    result1: Result = pd.read_pickle(final_path)
    assert int(result1.state.optim_state[0]) == 3
    assert jnp.isclose(result1.epoch, 3)

    # Rerun three epochs in one go.
    shutil.rmtree(output_path)
    args = [
        "--n-epochs=3",
        f"--config={config_path}",
        f"--data={simulated_data_path}",
        output_path,
    ]
    __main__(list(map(str, args)))
    assert final_path.is_file()
    result2: Result = pd.read_pickle(final_path)
    assert int(result2.state.optim_state[0]) == 3
    assert jnp.isclose(result2.epoch, 3)

    # Sample from the posterior with the same key and check the results are the same.
    jax.tree.map(ifnt.testing.assert_allclose, result1.params, result2.params)
