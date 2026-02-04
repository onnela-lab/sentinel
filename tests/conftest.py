import numpyro
import jax

jax.config.update("jax_platform_name", "cpu")

numpyro.enable_validation()
numpyro.enable_x64()
