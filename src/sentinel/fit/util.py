from dataclasses import dataclass
from datetime import datetime
import isoweek
import jax
from jax import numpy as jnp
import logging
from matplotlib import pyplot as plt
import numpyro
from numpyro.infer.svi import SVIState
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import pathlib
import pickle
import shutil
from sklearn import preprocessing
from typing import Any
from ..util import atomic_open


LOGGER = logging.getLogger(__name__)


@dataclass
class Result:
    """
    Complete state for fitting and restarting models fitting.
    """

    state: SVIState
    key: jax.random.PRNGKey
    params: dict[str, jnp.ndarray]
    median: dict[str, jnp.ndarray]
    epoch: int
    statistics: Any


def load_holidays(start, end) -> pd.DataFrame:
    # Get holidays in the right range from the calendar.
    calendar = USFederalHolidayCalendar()
    holidays = (
        calendar.holidays(start=start, end=end, return_name=True)
        .reset_index(name="holiday")
        .rename({"index": "date"}, axis=1)
    )
    holidays["isoweek"] = holidays["date"].map(
        lambda x: isoweek.Week.withdate(x).isoformat()
    )
    holidays["holiday_id"] = preprocessing.LabelEncoder().fit_transform(
        holidays.holiday
    )
    holidays["day_name"] = holidays.date.dt.day_name()
    return holidays


def load_data(path: pathlib.Path, *, remove_zeros: bool = False) -> tuple[dict, Any]:
    """
    Load data from a path.

    Args:
        path: Path to the data. The data must be a CSV file with columns `isoweek_id`,
            `location_id`, `type_id`, and `volume`.
        remove_zeros: Remove elements without observations.

    Returns:
        Dictionary with data for the model and auxiliary information.
    """
    raw = pd.read_csv(path)
    if remove_zeros:
        raw = raw[raw.volume > 0]

    # Get dimensions. Weeks may not be contiguous, and we use `max + 1` instead.
    n_weeks = raw.isoweek_id.max() + 1
    n_locs = raw.location_id.nunique()
    n_types = raw.type_id.nunique()
    n_obs = len(raw)

    # Verify key structure.
    assert raw.isoweek_id.min() == 0
    for key in ["location_id", "type_id"]:
        assert raw[key].min() == 0
        assert raw[key].nunique() == raw[key].max() + 1

    LOGGER.info(
        f"Loading data with {n_obs:,} observations in {n_weeks:,} weeks, {n_locs:,} "
        f"locations, with {n_types:,} types."
    )

    # Get tensors as jax arrays.
    target_scaler = preprocessing.StandardScaler()
    target = jnp.asarray(
        target_scaler.fit_transform(jnp.log1p(raw.volume.values[:, None]))[:, 0]
    )
    week_id = jnp.asarray(raw.isoweek_id)
    type_id = jnp.asarray(raw.type_id)
    loc_id = jnp.asarray(raw.location_id)

    # Construct lookup tables for ids.
    to_id = {
        key: raw.groupby(key)[f"{key}_id"].first()
        for key in ["location", "type", "isoweek"]
    }
    from_id = {
        key: raw.groupby(f"{key}_id")[key].first()
        for key in ["location", "type", "isoweek"]
    }

    # Get holidays in the right range from the calendar and add week identifiers.
    start = isoweek.Week.fromstring(raw.isoweek.min()).monday()
    end = isoweek.Week.fromstring(raw.isoweek.max()).sunday()
    holidays = load_holidays(start=start, end=end)
    holidays["isoweek_id"] = holidays.isoweek.map(to_id["isoweek"].__getitem__)
    n_holidays = holidays.holiday.nunique()

    # Multi-hot encode the holidays. We use `add` so we can verify that there aren't any
    # duplicate holidays in the same week. Unlikely, but good to have sanity checks.
    holiday_features = (
        jnp.zeros((n_weeks, n_holidays))
        .at[holidays.isoweek_id.values, holidays.holiday_id.values]
        .add(1)
    )
    assert holiday_features.max() == 1
    # We expand the holidays here because we can't mini-batch otherwise and de-mean the
    # features globally.
    holiday_features = jnp.asarray(holiday_features[week_id])
    holiday_features = holiday_features - holiday_features.mean(axis=0)

    # Assemble the data for the model.
    data = {
        "n_weeks": n_weeks,
        "n_types": n_types,
        "n_locs": n_locs,
        "n_obs": n_obs,
        "week_id": week_id,
        "type_id": type_id,
        "loc_id": loc_id,
        "target": target,
        "n_features": n_holidays,
        "features": holiday_features,
    }
    # We need to construct the Mondays carefully because there are missing sections in
    # the isoweeks.
    first_week = isoweek.Week.fromstring(from_id["isoweek"].min())
    mondays = [(first_week + i).monday() for i in range(n_weeks)]
    aux = {
        "to_id": to_id,
        "from_id": from_id,
        "raw": raw,
        "start": start,
        "end": end,
        "holidays": holidays,
        "mondays": mondays,
        "target_scaler": target_scaler,
    }
    return data, aux


def dump_state(
    path: pathlib.Path,
    svi: numpyro.infer.SVI,
    state: SVIState,
    key: jax.random.PRNGKey,
    epoch: int,
    aux: dict,
    statistics: Any,
) -> None:
    """
    Dump the training state to disk for later resumption.
    """
    params = svi.get_params(state)
    result = Result(state, key, params, svi.guide.median(params), epoch, statistics)
    with atomic_open(path, "wb") as fp:
        pickle.dump(result, fp)

    # Also save some figures for diagnostics and copy the checkpoint so we can look at
    # the evolution of parameters over time.
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamped_path = path.parent / f"{now}_{epoch}_{path.name}"
    shutil.copy(path, timestamped_path)

    matrices = {
        "A": (
            result.median["A"] - result.median["A"].mean(axis=1, keepdims=True),
            aux["from_id"]["location"],
        ),
        "B": (
            result.median["B"] - result.median["B"].mean(axis=1, keepdims=True),
            aux["from_id"]["type"],
        ),
    }
    for key, (value, from_id) in matrices.items():
        fig, ax = plt.subplots()
        vmax = jnp.abs(value).max()
        im = ax.imshow(
            value,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
            interpolation="none",
        )
        ticks = range(value.shape[0])
        ax.yaxis.set_ticks(ticks)
        ax.yaxis.set_ticklabels(f"{i}: {from_id.loc[i]}" for i in ticks)
        fig.colorbar(im, ax=ax, label=key)
        fig.tight_layout()
        fig.savefig(path.parent / f"{now}_{key}.png")
        plt.close(fig)
