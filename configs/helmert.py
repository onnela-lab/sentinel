"""
Configuration with full Helmert decoupling for z/A/B/C and within-coefficient ANOVA.

Extends reparam_guide_helmert2.py with a per-feature Helmert rotation on the
regression coefficients:

1. **4-variable Helmert rotation** on global scalar means (z_bar, A_bar, B_bar, C_bar).
2. **Per-location rotation (A vs C row effects)**.
3. **Per-type rotation (B vs C column effects)**.
4. **Double-centered C interaction**.
5. **Per-feature 3-variable Helmert** on (coef[k], mean(coef_loc[:,k]),
   mean(coef_type[:,k])).  For each holiday feature k, the total effect
   ``coef[k] + mean(coef_loc[:,k]) + mean(coef_type[:,k])`` is well-identified but
   the individual terms are not.  The Helmert rotation decouples the identified sum
   from the two null directions.  Per-location and per-type deviations of ``coef_loc``
   and ``coef_type`` are centered per feature (column-wise).

The model is unchanged.
"""

from jax import numpy as jnp
import numpyro
from numpyro import distributions as dist
import optax

from sentinel.fit import Config
from sentinel.models import GaussianProcessCollapsedModel

SQRT2 = jnp.sqrt(2.0)
SQRT3 = jnp.sqrt(3.0)
SQRT6 = jnp.sqrt(6.0)
SQRT12 = jnp.sqrt(12.0)


def _helmert4_inverse(s, d1, d2, d3):
    """Recover (z_bar, A_bar, B_bar, C_bar) from 4-variable Helmert coordinates."""
    z_bar = s / 2 + d1 / SQRT2 + d2 / SQRT6 + d3 / SQRT12
    A_bar = s / 2 - d1 / SQRT2 + d2 / SQRT6 + d3 / SQRT12
    B_bar = s / 2 - 2 * d2 / SQRT6 + d3 / SQRT12
    C_bar = s / 2 - 3 * d3 / SQRT12
    return z_bar, A_bar, B_bar, C_bar


def _helmert3_inverse(s, d1, d2):
    """Recover (a, b, c) from 3-variable Helmert coordinates.

    Works element-wise on vectors (e.g. one Helmert per feature).
    """
    a = s / SQRT3 + d1 / SQRT2 + d2 / SQRT6
    b = s / SQRT3 - d1 / SQRT2 + d2 / SQRT6
    c = s / SQRT3 - 2 * d2 / SQRT6
    return a, b, c


def _center_rows(x):
    """Center each row (subtract per-row mean over the last axis)."""
    return x - x.mean(axis=-1, keepdims=True)


def _center_cols(x):
    """Center each column (subtract per-column mean over axis 0)."""
    return x - x.mean(axis=0, keepdims=True)


def _double_center(x):
    """Double-center a 2-D matrix (zero row means and column means)."""
    return (
        x - x.mean(axis=-1, keepdims=True) - x.mean(axis=-2, keepdims=True) + x.mean()
    )


def _rotate_sum_diff(raw_sum, raw_diff):
    """Rotate centered (sum, diff) pair into (component_a, component_b).

    Both inputs are centered (zero grand mean) before use.  Returns::

        a = (sum_c + diff_c) / sqrt(2)
        b = (sum_c - diff_c) / sqrt(2)

    where ``a + b`` is the identified sum direction.
    """
    s_c = raw_sum - raw_sum.mean()
    d_c = raw_diff - raw_diff.mean()
    return (s_c + d_c) / SQRT2, (s_c - d_c) / SQRT2


class _ReparamGuideHelmert3:
    """
    Guide with full Helmert decoupling for z/A/B/C and within-coefficient ANOVA.

    Blocks ``z``, ``A``, ``B``, ``C``, ``coef``, ``coef_loc``, ``coef_type``
    from AutoNormal and handles them with structured rotations and centerings.
    """

    BLOCKED = ["z", "A", "B", "C", "coef", "coef_loc", "coef_type"]

    def __init__(self, model, *, init_loc_fn, init_scale):
        def blocked(**kwargs):
            with numpyro.handlers.block(hide=self.BLOCKED):
                model(**kwargs)

        self._auto = numpyro.infer.autoguide.AutoNormal(
            blocked,
            init_loc_fn=init_loc_fn,
            init_scale=init_scale,
        )
        self._init_scale = init_scale

    def _sample_z_A_B_C(self, n_locs, n_types, n_weeks_padded):
        """Sample z, A, B, C with 4-variable Helmert and per-row decoupling."""
        # --- 4-variable Helmert rotation for global scalar means ---
        s = numpyro.param("s", 0.0)
        d1 = numpyro.param("d1", 0.0)
        d2 = numpyro.param("d2", 0.0)
        d3 = numpyro.param("d3", 0.0)
        z_bar, A_bar, B_bar, C_bar = _helmert4_inverse(s, d1, d2, d3)

        # --- z ---
        z_dev = numpyro.param("z_dev", jnp.zeros(n_weeks_padded))
        z_scale = numpyro.param(
            "z_scale",
            jnp.full(n_weeks_padded, self._init_scale),
            constraint=dist.constraints.positive,
        )
        z_loc = (z_dev - z_dev.mean()) + z_bar
        numpyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        # --- Per-location rotation: A row means vs C row effects ---
        alpha_sum = numpyro.param("alpha_sum", jnp.zeros(n_locs))
        alpha_diff = numpyro.param("alpha_diff", jnp.zeros(n_locs))
        alpha, r = _rotate_sum_diff(alpha_sum, alpha_diff)

        # --- A ---
        A_dev = numpyro.param("A_dev", jnp.zeros((n_locs, n_weeks_padded)))
        A_scale = numpyro.param(
            "A_scale",
            jnp.full((n_locs, n_weeks_padded), self._init_scale),
            constraint=dist.constraints.positive,
        )
        A_loc = _center_rows(A_dev) + alpha[:, None] + A_bar
        numpyro.sample("A", dist.Normal(A_loc, A_scale).to_event(1))

        # --- Per-type rotation: B row means vs C column effects ---
        beta_sum = numpyro.param("beta_sum", jnp.zeros(n_types))
        beta_diff = numpyro.param("beta_diff", jnp.zeros(n_types))
        beta, c = _rotate_sum_diff(beta_sum, beta_diff)

        # --- B ---
        B_dev = numpyro.param("B_dev", jnp.zeros((n_types, n_weeks_padded)))
        B_scale = numpyro.param(
            "B_scale",
            jnp.full((n_types, n_weeks_padded), self._init_scale),
            constraint=dist.constraints.positive,
        )
        B_loc = _center_rows(B_dev) + beta[:, None] + B_bar
        numpyro.sample("B", dist.Normal(B_loc, B_scale).to_event(1))

        # --- C ---
        C_int_dev = numpyro.param("C_int_dev", jnp.zeros((n_locs, n_types)))
        C_scale = numpyro.param(
            "C_scale",
            jnp.full((n_locs, n_types), self._init_scale),
            constraint=dist.constraints.positive,
        )
        C_loc = r[:, None] + c[None, :] + _double_center(C_int_dev) + C_bar
        numpyro.sample("C", dist.Normal(C_loc, C_scale).to_event())

    def _sample_coefs(self, n_locs, n_types, n_features):
        """Sample coef, coef_loc, coef_type with per-feature 3-variable Helmert."""
        # --- Per-feature Helmert: (coef, mean(coef_loc), mean(coef_type)) ---
        coef_s = numpyro.param("coef_s", jnp.zeros(n_features))
        coef_d1 = numpyro.param("coef_d1", jnp.zeros(n_features))
        coef_d2 = numpyro.param("coef_d2", jnp.zeros(n_features))
        coef_bar, coef_loc_bar, coef_type_bar = _helmert3_inverse(
            coef_s,
            coef_d1,
            coef_d2,
        )

        # --- coef: global coefficients ---
        coef_scale = numpyro.param(
            "coef_scale",
            jnp.full(n_features, self._init_scale),
            constraint=dist.constraints.positive,
        )
        numpyro.sample("coef", dist.Normal(coef_bar, coef_scale).to_event())

        # --- coef_loc: per-location deviations (centered per feature) + mean ---
        coef_loc_dev = numpyro.param(
            "coef_loc_dev",
            jnp.zeros((n_locs, n_features)),
        )
        coef_loc_scale = numpyro.param(
            "coef_loc_scale",
            jnp.full((n_locs, n_features), self._init_scale),
            constraint=dist.constraints.positive,
        )
        coef_loc_loc = _center_cols(coef_loc_dev) + coef_loc_bar[None, :]
        numpyro.sample(
            "coef_loc",
            dist.Normal(coef_loc_loc, coef_loc_scale).to_event(1),
        )

        # --- coef_type: per-type deviations (centered per feature) + mean ---
        coef_type_dev = numpyro.param(
            "coef_type_dev",
            jnp.zeros((n_types, n_features)),
        )
        coef_type_scale = numpyro.param(
            "coef_type_scale",
            jnp.full((n_types, n_features), self._init_scale),
            constraint=dist.constraints.positive,
        )
        coef_type_loc = _center_cols(coef_type_dev) + coef_type_bar[None, :]
        numpyro.sample(
            "coef_type",
            dist.Normal(coef_type_loc, coef_type_scale).to_event(1),
        )

    def __call__(self, **kwargs):
        n_locs = kwargs["n_locs"]
        n_types = kwargs["n_types"]
        n_weeks_padded = kwargs["n_weeks_padded"]
        n_features = kwargs["n_features"]

        # AutoNormal handles all non-blocked sites.
        self._auto(**kwargs)

        # Structured reparameterizations.
        self._sample_z_A_B_C(n_locs, n_types, n_weeks_padded)
        self._sample_coefs(n_locs, n_types, n_features)

    def median(self, params):
        result = self._auto.median(params)

        # --- z, A, B, C ---
        z_bar, A_bar, B_bar, C_bar = _helmert4_inverse(
            params["s"],
            params["d1"],
            params["d2"],
            params["d3"],
        )

        z_dev = params["z_dev"]
        result["z"] = (z_dev - z_dev.mean()) + z_bar

        alpha, r = _rotate_sum_diff(params["alpha_sum"], params["alpha_diff"])

        A_dev = params["A_dev"]
        result["A"] = _center_rows(A_dev) + alpha[:, None] + A_bar

        beta, c = _rotate_sum_diff(params["beta_sum"], params["beta_diff"])

        B_dev = params["B_dev"]
        result["B"] = _center_rows(B_dev) + beta[:, None] + B_bar

        C_int_dev = params["C_int_dev"]
        result["C"] = r[:, None] + c[None, :] + _double_center(C_int_dev) + C_bar

        # --- coef, coef_loc, coef_type ---
        coef_bar, coef_loc_bar, coef_type_bar = _helmert3_inverse(
            params["coef_s"],
            params["coef_d1"],
            params["coef_d2"],
        )

        result["coef"] = coef_bar
        result["coef_loc"] = (
            _center_cols(params["coef_loc_dev"]) + coef_loc_bar[None, :]
        )
        result["coef_type"] = (
            _center_cols(params["coef_type_dev"]) + coef_type_bar[None, :]
        )

        return result


class ReparamGuideHelmert3Config(Config):
    def get_guide(self, model, args):
        init_loc_fn = numpyro.infer.init_to_uniform(radius=0.01)
        init_scale = 0.001
        return _ReparamGuideHelmert3(
            model,
            init_loc_fn=init_loc_fn,
            init_scale=init_scale,
        )


def setup() -> ReparamGuideHelmert3Config:
    model = GaussianProcessCollapsedModel(
        kernel="matern32",
        method="fft",
        coefficient_prior="independent",
        dof=None,
    )

    schedule = [
        (optax.constant_schedule(1e-2), 50_000),
        (optax.constant_schedule(1e-3), 50_000),
        (optax.constant_schedule(1e-4), 50_000),
    ]
    funcs, steps = zip(*schedule)
    *boundaries, n_epochs = jnp.cumsum(jnp.asarray(steps))
    lr_schedule = optax.join_schedules(funcs, boundaries)

    return ReparamGuideHelmert3Config(
        model=model,
        guide=None,
        optim=optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_schedule(lr_schedule),
            optax.scale(-1),
        ),
        n_epochs=n_epochs,
        enable_x64=False,
        log_every=10_000,
        scan_size=1_000,
    )
