```python
import datetime
import matplotlib as mpl
from matplotlib import pyplot as plt
import isoweek
import itertools
import numpy as np
import pandas as pd
from sklearn import preprocessing
```

```python
# Load the data and apply basic validation.
tycho = pd.read_csv(
    "../data/ProjectTycho_Level2_v1.1.0.csv",
    parse_dates=["from_date", "to_date"],
).rename(
    {" event": "event"},
    axis=1,
)

assert (tycho.country == "US").all(), "All records must be from the United States."

# Only consider cases.
tycho = tycho[tycho.event == "CASES"]

# Only consider state-level data, not city level data.
tycho = tycho[tycho.loc_type == "STATE"]

# Map parts of New York to the state.
tycho = tycho.replace({"state": {
    "UPSTATE NEW YORK": "NEW YORK",
    "NEW YORK CITY": "NEW YORK",
}})

# Restrict to continental United States.
tycho = tycho[tycho.state.isin({
    # "AK",  # Alaska.
    "AL", "AR",
    # "AS",  # American Samoa.
    "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS",
    "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO",
    # "MP",  # Northern Mariana Islands.
    "MS", "MT", "NC", "ND", "NE",
    "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA",
    # "PR",  # Puerto Rico.
    "RI", "SC", "SD", "TN",
    "TX", "UT", "VA",
    # "VI",  # Virgin Islands.
    "VT", "WA", "WI", "WV", "WY"
})]

# Exclude diseases with very sparse observations.
tycho = tycho[~tycho.disease.isin([
    "BABESIOSIS",
    "EHRLICHIOSIS/ANAPLASMOSIS",
    "ROCKY MOUNTAIN SPOTTED FEVER",
    "TUBERCULOSIS [PHTHISIS PULMONALIS]",
])]

# Add year information and restrict to a three-year period from 2011 to 2013
# (inclusive). The most recent data from 2014 are not complete for the whole year.
tycho["from_year"] = tycho.from_date.dt.year
year_lower = 2011
year_upper = 2014
tycho = tycho[(year_lower <= tycho.from_year) & (tycho.from_year < year_upper)]

# Add iso week identifiers. Iso weeks start on Mondays; epi weeks start on Sundays
# (which we verify). We use the iso week that contains the end of the epi week because
# that maximizes the overlap.
np.testing.assert_array_equal(tycho.from_date.dt.weekday, 6)
tycho["isoweek"] = tycho.to_date.map(lambda x: isoweek.Week.withdate(x).isoformat())
tycho["monday"] = tycho.isoweek.map(lambda x: isoweek.Week.fromstring(x).monday())

# Verify weeks are consecutive.
sorted_mondays = np.sort(tycho.monday.unique())
deltas = np.diff(sorted_mondays)
assert all(deltas == datetime.timedelta(days=7))

print("\n".join([
    f"number of records: {tycho.shape[0]}",
    f"number of weeks: {tycho.isoweek.nunique()}",
    f"number of diseases: {tycho.disease.nunique()}",
    f"number of states: {tycho.state.nunique()}",
]))
```

```python
# Dump the data in a standard form that we will also use elsewhere.
processed = pd.DataFrame({
    "isoweek": tycho.isoweek,
    "monday": tycho.monday,
    "type": tycho.disease,
    "location": tycho.state,
    "volume": tycho.number,
})
for key in ["isoweek", "type", "location"]:
    processed[f"{key}_id"] = preprocessing.LabelEncoder().fit_transform(processed[key])
processed.to_csv("../data/project_tycho_processed_cases.csv", index=False)
```

```python
for x, y in itertools.combinations(["isoweek_id", "type_id", "location_id"], 2):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    # Volumes.
    ax1.set_title("volumes")
    agg = processed.groupby([x, y]).volume.sum().reset_index()
    i, j, z = agg.values.T
    agg = np.zeros((processed[x].nunique(), processed[y].nunique()))
    agg[i, j] = z
    im = ax1.imshow(agg.T, aspect="auto", interpolation="none", norm=mpl.colors.LogNorm())

    # Counts.
    ax2.set_title("counts")
    agg = processed.groupby([x, y]).volume.count().reset_index()
    i, j, z = agg.values.T
    agg = np.zeros((processed[x].nunique(), processed[y].nunique()))
    agg[i, j] = z
    im = ax2.imshow(agg.T, aspect="auto", interpolation="none", norm=mpl.colors.LogNorm())

    ax1.set_xlabel(x)
    ax1.set_ylabel(y)
    ax2.set_xlabel(x)
    fig.tight_layout()
```
