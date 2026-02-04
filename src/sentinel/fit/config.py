import numpyro
from numpyro.infer.svi import SVIState
import optax
import pathlib

from ..models import Model
from .util import load_data


class BaseConfig:
    """
    Base configuration for fitting models.

    Args:
        n_epochs: Number of epochs to train.
        seed: Random seed for reproducibility.
        log_every: Log every :code:`log_every` epochs.
        checkpoint_every: Save a checkpoint every :code:`checkpoint_every` seconds.
        enable_x64: Enable 64-bit precision.
        scan_size: Number of epochs to wrap in a :code:`jax.lax.scan`.

    Methods:
        load_data: Load data from a file.
        get_model: Get the model to fit.
        get_guide: Get the guide to fit the model.
        get_optim: Get the optimizer for training.
    """

    def __init__(
        self,
        *,
        n_epochs: int,
        seed: int = 42,
        log_every: int = 10,
        checkpoint_every: float = 300,
        enable_x64: bool = False,
        scan_size: int = 1,
    ) -> None:
        self.n_epochs = n_epochs
        self.seed = seed
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.enable_x64 = enable_x64
        self.scan_size = scan_size
        assert log_every >= scan_size, "`log_every` cannot be smaller than `scan_size`."

    def load_data(self, path: pathlib.Path) -> dict:
        raise NotImplementedError

    def get_model(self) -> Model:
        raise NotImplementedError

    def get_guide(self, model: Model, data: dict) -> numpyro.infer.autoguide.AutoGuide:
        raise NotImplementedError

    def get_optim(self) -> optax.GradientTransformation:
        raise NotImplementedError

    def evaluate_epoch_statistics(self, state: SVIState) -> tuple[SVIState, dict]:
        """
        Evaluate statistics of the optimizer state after each epoch.
        """
        return state, {}

    def reparam_model(self, model, args):
        return model


class Config(BaseConfig):
    """
    Simple configuration for fitting models.

    Args:
        model: Model to fit.
        guide: Guide to fit the model.
        optim: Optimizer for training.
        n_epochs: Number of epochs to train.
        seed: Random seed for reproducibility.
        log_every: Log every :code:`log_every` epochs.
        checkpoint_every: Save a checkpoint every :code:`checkpoint_every` seconds.
        enable_x64: Enable 64-bit precision.

    Methods:
        load_data: Load data from a file.
        get_model: Get the model to fit.
        get_guide: Get the guide to fit the model.
        get_optim: Get the optimizer for training.
    """

    def __init__(
        self,
        *,
        model: Model,
        guide: numpyro.infer.autoguide.AutoGuide,
        optim: optax.GradientTransformation,
        n_epochs: int,
        seed: int = 42,
        log_every: int = 10,
        checkpoint_every: float = 300,
        enable_x64: bool = False,
        scan_size: int = 1,
    ) -> None:
        super().__init__(
            n_epochs=n_epochs,
            seed=seed,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            enable_x64=enable_x64,
            scan_size=scan_size,
        )
        self.model = model
        self.guide = guide
        self.optim = optim

    def get_guide(self, model: Model, data: dict) -> numpyro.infer.autoguide.AutoGuide:
        return self.guide

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.optim

    def load_data(self, path):
        return load_data(path)
