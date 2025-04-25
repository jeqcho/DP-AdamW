so haiku wasn't detected by cursor
tensorflow too
we then got
```
ImportError: cannot import name 'update_moment' from 'optax._src.transform' (/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/optax/_src/transform.py)
```

Let's downgrade optax

the DP-AdamBC last commit is Jan 8 2024
Initial commit is Dec 12 2023.

Let's downgrade to pre Dec 12 2023.

Downgrade optax from 0.2.4 to 0.1.7 Jul 26 2023

```
Traceback (most recent call last):
  File "/Users/jeqcho/DP-AdamBC/jax_implementation/main.py", line 19, in <module>
    from jax.config import config
ModuleNotFoundError: No module named 'jax.config'
```

Downgrade jax from 0.6.0 to 0.4.21 Dec 4 2023

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
flax 0.10.6 requires jax>=0.5.1, but you have jax 0.4.21 which is incompatible.
orbax-checkpoint 0.11.12 requires jax>=0.5.0, but you have jax 0.4.21 which is incompatible.
chex 0.1.89 requires jax>=0.4.27, but you have jax 0.4.21 which is incompatible.
Successfully installed jax-0.4.21
```

downgrade flax from 0.10.6 to 0.7.5 Oct 27 2023

downgrade chex from 0.1.89 to 0.1.85 Nov 22 2023

downgrade orbax-checkpoint from 0.11.12 to 0.4.7 Dec 7 2023

```
Traceback (most recent call last):
  File "/Users/jeqcho/DP-AdamBC/jax_implementation/main.py", line 6, in <module>
    from models.classifiers import get_classifier
  File "/Users/jeqcho/DP-AdamBC/jax_implementation/models/classifiers.py", line 2, in <module>
    from models.cnn import CNN2, CNN5
  File "/Users/jeqcho/DP-AdamBC/jax_implementation/models/cnn.py", line 3, in <module>
    import haiku as hk
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/haiku/__init__.py", line 20, in <module>
    from haiku import experimental
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/haiku/experimental/__init__.py", line 59, in <module>
    from haiku.experimental import flax
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/haiku/experimental/flax.py", line 18, in <module>
    from haiku._src.flax.flax_module import Module
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/haiku/_src/flax/flax_module.py", line 20, in <module>
    import flax.core
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/flax/__init__.py", line 19, in <module>
    from .configurations import (
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/flax/configurations.py", line 93, in <module>
    flax_filter_frames = define_bool_state(
                         ^^^^^^^^^^^^^^^^^^
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/flax/configurations.py", line 42, in define_bool_state
    return jax_config.define_bool_state('flax_' + name, default, help)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'Config' object has no attribute 'define_bool_state'
```
again Downgrade jax from 0.6.0 to 0.4.21 Dec 4 2023

```
RuntimeError: jaxlib version 0.6.0 is newer than and incompatible with jax version 0.4.21. Please update your jax and/or jaxlib packages.
```

downgrade jaxlib from 0.6.0 to 0.4.21 Dec 4 2023

```
Traceback (most recent call last):
  File "/Users/jeqcho/DP-AdamBC/jax_implementation/main.py", line 6, in <module>
    from models.classifiers import get_classifier
  File "/Users/jeqcho/DP-AdamBC/jax_implementation/models/classifiers.py", line 2, in <module>
    from models.cnn import CNN2, CNN5
  File "/Users/jeqcho/DP-AdamBC/jax_implementation/models/cnn.py", line 3, in <module>
    import haiku as hk
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/haiku/__init__.py", line 20, in <module>
    from haiku import experimental
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/haiku/experimental/__init__.py", line 55, in <module>
    from haiku.experimental import jaxpr_info
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/haiku/experimental/jaxpr_info.py", line 18, in <module>
    from haiku._src.jaxpr_info import as_html
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/haiku/_src/jaxpr_info.py", line 106, in <module>
    ComputeFlopsFn = Callable[[jax_core.JaxprEqn, Expression], int]
                               ^^^^^^^^^^^^^^^^^
AttributeError: module 'jax.extend.core' has no attribute 'JaxprEqn'
```

downgrade dm-haiku from 0.0.14 to 0.0.11 Nov 10 2023

```
Traceback (most recent call last):
  File "/Users/jeqcho/DP-AdamBC/jax_implementation/main.py", line 7, in <module>
    from trainer.trainers import get_trainer
  File "/Users/jeqcho/DP-AdamBC/jax_implementation/trainer/trainers.py", line 3, in <module>
    from trainer.iterative import IterativeTrainer
  File "/Users/jeqcho/DP-AdamBC/jax_implementation/trainer/iterative.py", line 5, in <module>
    import optax
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/optax/__init__.py", line 17, in <module>
    from optax import contrib
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/optax/contrib/__init__.py", line 17, in <module>
    from optax._src.contrib.mechanic import MechanicState
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/optax/_src/contrib/mechanic.py", line 38, in <module>
    from optax._src import utils
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/optax/_src/utils.py", line 22, in <module>
    import jax.scipy.stats.norm as multivariate_normal
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/jax/scipy/stats/__init__.py", line 40, in <module>
    from jax._src.scipy.stats.kde import gaussian_kde as gaussian_kde
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/jax/_src/scipy/stats/kde.py", line 26, in <module>
    from jax.scipy import linalg, special
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/jax/scipy/linalg.py", line 18, in <module>
    from jax._src.scipy.linalg import (
  File "/Users/jeqcho/DP-AdamBC/.venv/lib/python3.11/site-packages/jax/_src/scipy/linalg.py", line 402, in <module>
    @_wraps(scipy.linalg.tril)
            ^^^^^^^^^^^^^^^^^
AttributeError: module 'scipy.linalg' has no attribute 'tril'
```

scipy <=1.12.0,>= 1.6.0 according to https://github.com/octo-models/octo/issues/71

downgrade scipy from 1.15.2 to 1.11.4 Nov 18 2023

```
ModuleNotFoundError: No module named 'wandb'
```