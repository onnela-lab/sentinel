```python
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import isoweek
```

```python
# Paths — adjust as needed.
RESULT_PATH = "../workspace/helmert/final.pkl"
DATA_PATH = "../data/project_tycho_processed_cases.csv"
```

```python
# Load fitted result and raw data for labels.
with open(RESULT_PATH, "rb") as f:
    result = pickle.load(f)
median = result.median

raw = pd.read_csv(DATA_PATH)
n_weeks = raw.isoweek_id.max() + 1

# Build lookup tables.
disease_names = raw.groupby("type_id")["type"].first().sort_index()
first_week = isoweek.Week.fromstring(raw.isoweek.min())
mondays = [str((first_week + i).monday()) for i in range(n_weeks)]
```

```python
# Global temporal effect (trim padding).
z = median["z"][:n_weeks]

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(z)
ax.set_xticks(range(0, n_weeks, 13))
ax.set_xticklabels([mondays[i] for i in range(0, n_weeks, 13)], rotation=45, ha="right")
ax.set_ylabel("z")
ax.set_title("Global temporal effect")
fig.tight_layout()
```

```python
# Disease-specific temporal effects.
B = median["B"][:, :n_weeks]
n_types = B.shape[0]

n_cols = 3
n_rows = -(-n_types // n_cols)  # ceil division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 2.5 * n_rows), sharex=True)
axes = axes.flat

for i in range(n_types):
    ax = axes[i]
    ax.plot(B[i])
    ax.set_title(disease_names.loc[i], fontsize=9)
    ax.set_ylabel("B")

# x-axis labels on bottom row only.
for ax in axes[n_types - n_cols : n_types]:
    ax.set_xticks(range(0, n_weeks, 26))
    ax.set_xticklabels([mondays[i] for i in range(0, n_weeks, 26)], rotation=45, ha="right", fontsize=7)

# Hide unused subplots.
for i in range(n_types, len(axes)):
    axes[i].set_visible(False)

fig.suptitle("Disease-specific temporal effects", y=1.01)
fig.tight_layout()
```
