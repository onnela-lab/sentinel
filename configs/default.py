from jax import numpy as jnp
import numpyro
import optax
from sentinel.fit import Config
from sentinel.models import GaussianProcessCollapsedModel


class CustomInitConfig(Config):
    def get_guide(
        self, model: GaussianProcessCollapsedModel, args: dict
    ) -> numpyro.infer.autoguide.AutoGuide:
        init_loc_fn = numpyro.infer.init_to_uniform(radius=0.01)
        init_scale = 0.001
        return numpyro.infer.autoguide.AutoNormal(
            model, init_loc_fn=init_loc_fn, init_scale=init_scale
        )


def setup() -> CustomInitConfig:
    model = GaussianProcessCollapsedModel(
        kernel="matern32", method="fft", coefficient_prior="independent", dof=None
    )

    schedule = [
        (optax.constant_schedule(1e-2), 50_000),
        (optax.constant_schedule(1e-3), 50_000),
        (optax.constant_schedule(1e-4), 50_000),
    ]
    funcs, steps = zip(*schedule)
    *boundaries, n_epochs = jnp.cumsum(jnp.asarray(steps))
    lr_schedule = optax.join_schedules(funcs, boundaries)

    return CustomInitConfig(
        model=model,
        guide=None,
        optim=optax.chain(
            optax.scale_by_adam(), optax.scale_by_schedule(lr_schedule), optax.scale(-1)
        ),
        n_epochs=n_epochs,
        enable_x64=False,
        log_every=10_000,
        scan_size=1_000,
    )
