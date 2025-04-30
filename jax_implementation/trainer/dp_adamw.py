import functools
import jax
import jax.numpy as jnp
import optax
from optax._src import base, combine, numerics, utils
from optax._src.transform import bias_correction, update_moment, update_moment_per_elem_norm, add_decayed_weights
from optax._src.alias import _scale_by_learning_rate
from typing import Any, Callable, NamedTuple, Optional
import chex
import configlib
import math
from functools import partial
from itertools import chain
from opacus.accountants.rdp import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
import pickle
import os

# Import necessary components from other trainer files
from trainer.dp_iterative import DPIterativeTrainer, noise_and_normalize
from trainer.utils import tree_zeros_like, tree_flatten_1dim, grad_norm, tree_ones_like
from trainer.dp_adambc import scale_by_adam_corr
from data_utils.jax_dataloader import NumpyLoader, Cycle

# --- Arguments specific to DPAdamW ---
parser = configlib.add_parser("DP-AdamW Trainer config")
# Remove duplicate definitions - these are expected to be defined in dp_adambc.py
# parser.add_argument("--beta_1", default=0.9, type=float)
# parser.add_argument("--beta_2", default=0.999, type=float)
# parser.add_argument("--eps", type=float, default=1e-8)
# parser.add_argument("--eps_root", type=float, default=1e-8)
# parser.add_argument("--adam_corr", default=False, action='store_true', help="Enable bias correction for DP-AdamW similar to DPAdam.")
# AdamW specific argument - Keep this one
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for DPAdamW optimizer')

class ScaleByAdamWStateCorr(NamedTuple):
    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: base.Updates
    nu: base.Updates
    nu_corr: base.Updates
    count_tree: None  # individual param record of update count

class ScaleByAdamWStateCorrLong(NamedTuple):
    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: base.Updates
    nu: base.Updates
    nu_corr: base.Updates
    mu_clean: base.Updates
    nu_clean: base.Updates
    perc_corr: None

# def scale_by_adamw_corr(
#         batch_size: int,
#         dp_noise_multiplier: float,
#         dp_l2_norm_clip: float,
#         b1: float,
#         b2: float,
#         eps: float, # Adam epsilon
#         eps_root: float, # Epsilon for corrected nu
#         mu_dtype: Optional[Any] = None,
# ) -> base.GradientTransformation:
#     mu_dtype = utils.canonicalize_dtype(mu_dtype)

#     def init_fn(params):
#         mu = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
#         nu = jax.tree_util.tree_map(jnp.zeros_like, params)
#         nu_corr = jax.tree_util.tree_map(jnp.zeros_like, params) # Potentially remove if unused
#         mu_clean = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
#         nu_clean = jax.tree_util.tree_map(jnp.zeros_like, params)
#         return ScaleByAdamWStateCorrLong(
#             count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, # nu_corr=nu_corr,
#             mu_clean=mu_clean, nu_clean=nu_clean, perc_corr=None
#         )

#     def update_fn(updates, state, params=None):
#         # `params` are needed for the weight decay term, but Optax handles
#         # weight decay separately, so we don't use `params` here directly.
#         del params

#         # DPAdam logic: updates contain (noised_updates, clean_clipped_updates)
#         noised_updates, clean_updates = updates

#         # Update moments with noised gradients
#         mu = update_moment(noised_updates, state.mu, b1, 1)
#         nu = update_moment_per_elem_norm(noised_updates, state.nu, b2, 2)
#         count_inc = numerics.safe_int32_increment(state.count)

#         # Bias correction for moments
#         mu_hat = bias_correction(mu, b1, count_inc)
#         nu_hat = bias_correction(nu, b2, count_inc)

#         # DP Noise correction for second moment (from dp_adambc.py)
#         noise_err = (1 / batch_size ** 2) * dp_noise_multiplier ** 2 * dp_l2_norm_clip ** 2
#         nu_hat_corr = jax.tree_map(lambda x: jnp.maximum(x - noise_err, eps_root), nu_hat)

#         # Calculate the Adam update *without* weight decay (as per AdamW)
#         # Use the DP-corrected second moment estimate
#         updates = jax.tree_util.tree_map(
#             lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat, nu_hat_corr)

#         # Update clean moments (using clipped, unnoised gradients) for logging/analysis
#         mu_clean = update_moment(clean_updates, state.mu_clean, b1, 1)
#         nu_clean = update_moment_per_elem_norm(clean_updates, state.nu_clean, b2, 2)
#         # mu_hat_clean = bias_correction(mu_clean, b1, count_inc) # Optional: if needed for logging
#         # nu_hat_clean = bias_correction(nu_clean, b2, count_inc) # Optional: if needed for logging

#         # Logging percentage of corrected nu (from dp_adambc.py)
#         # Use tree_flatten_1dim from trainer.utils instead of optax's non-existent tree_flatten_float_dtype
#         num_corr1 = jnp.sum(tree_flatten_1dim(jax.tree_map(lambda x: jnp.sum((x - noise_err) > eps_root), nu_hat)))
#         dummy_count = jnp.sum(tree_flatten_1dim(jax.tree_map(lambda x: jnp.sum(~jnp.isnan(x)), nu_hat)))
#         perc_corr1 = num_corr1 / dummy_count
#         perc_corr2 = 0 # Placeholder, adjust if needed

#         mu = utils.cast_tree(mu, mu_dtype)
#         new_state = ScaleByAdamWStateCorrLong(
#             count=count_inc, mu=mu, nu=nu, # nu_corr=nu_hat_corr, # Store corrected nu if needed
#             mu_clean=mu_clean, nu_clean=nu_clean, perc_corr=(perc_corr1, perc_corr2)
#         )
#         return updates, new_state

#     return base.GradientTransformation(init_fn, update_fn)

def adamw(
    batch_size: int,
    dp_noise_multiplier: float,
    dp_l2_norm_clip: float,
    learning_rate: float,
    b1: float,
    b2: float,
    eps: float,
    eps_root: float,
    eps_root_decay: float,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 0.0, # AdamW specific
) -> base.GradientTransformation:
    return combine.chain(
        # This transformation handles the Adam moment updates with DP correction
        scale_by_adam_corr(
            batch_size=batch_size, dp_noise_multiplier=dp_noise_multiplier, dp_l2_norm_clip=dp_l2_norm_clip,
            b1=b1, b2=b2, eps=eps, eps_root=eps_root, eps_root_decay=eps_root_decay, mu_dtype=mu_dtype),
        add_decayed_weights(weight_decay=weight_decay),
        _scale_by_learning_rate(learning_rate),
    ) 

# --- DPAdamWTrainer Class (Adapted from DPAdamTrainer in dp_adambc.py) ---
class DPAdamWTrainer(DPIterativeTrainer):
    def __init__(
            self,
            conf: configlib.Config,
            model_fn,
            train_set,
            test_set,
            seed: int,
    ):
        super().__init__(conf, model_fn, train_set, test_set, seed)
        self.helper_loader = self.loader(self.train_set)
        self.helper_loader_itr = iter(Cycle(self.helper_loader))
        self.setup_optimizer_adamw()  # overwrite default optimizer

    def setup_optimizer_adamw(self):
        """Sets up the DPAdamW optimizer using optax."""
        beta_1 = self.conf.beta_1
        # beta_2 = 1 - ((1-self.conf.beta_1) ** 2)  # (1-b1) = sqrt(1-b2)
        beta_2 = self.conf.beta_2
        
        # Use dp_adamw optax chain, including weight decay
        # The --adam_corr flag enables the bias correction within scale_by_adamw
        if self.conf.adam_corr:
            self.opt = optax.inject_hyperparams(adamw)(
                batch_size=self.conf.batch_size, dp_noise_multiplier=self.noise_multiplier,
                dp_l2_norm_clip=self.conf.dp_l2_norm_clip,
                learning_rate=self.lr, b1=beta_1, b2=beta_2,
                eps=self.conf.eps, eps_root=self.conf.eps_root, eps_root_decay=-1,
                weight_decay=self.conf.weight_decay, # AdamW specific
            )
        else:
            self.opt = optax.inject_hyperparams(optax.adamw)(
                learning_rate=self.lr, b1=self.conf.beta_1, b2=self.conf.beta_2, eps_root=self.conf.tmp_bias,
                weight_decay=self.conf.weight_decay,
            )

        if self.conf.reload_ckpt_path is None:
            self.opt_state = self.opt.init(self.theta)
        else:
            self.opt_state = self.opt_state_restored # Assumes state structure matches

    # --- Copy and potentially adapt compute_update and apply_update --- 
    # --- from DPAdamTrainer in dp_adambc.py                   ---

    def compute_update(self, theta, X, y, metadata={}, *kwargs):
        batch_size = X.shape[0]

        if self.conf.virtual_batch_size is None:
            virtual_batch_size = batch_size
        else:
            virtual_batch_size = self.conf.virtual_batch_size

        virtual_batch_num = math.ceil(batch_size / virtual_batch_size)
        grads = None
        loss = 0.
        for i in range(virtual_batch_num):
            start = i * virtual_batch_size
            end = (i + 1) * virtual_batch_size
            per_example_clipped_grads, losses, _grads, _dp_grads_only_clipped, _dp_grads_only_unclipped, \
                clip_mask, grad_norms = self.dp_grads(
                    theta, X[start:end], y[start:end],
                    l2_norm_clip=self.conf.dp_l2_norm_clip, save_clip_unclip=False
                )
            loss += float(jnp.sum(losses))

            if grads is None:
                grads = _grads
            else:
                grads = jax.tree_util.tree_map(jnp.add, grads, _grads)

        mean_clipped_grads = jax.tree_map(lambda x: x / batch_size, grads)

        if not self.conf.clipping_only:
            grads, noise_tree = noise_and_normalize(
                grads, self.conf.dp_l2_norm_clip, self.noise_multiplier, batch_size, next(self.prng_seq),
                save_noise_tree=False
            )
        else:
            grads = mean_clipped_grads

        metadata["loss"] = float(loss / batch_size)

        return {'grads': grads, 'clipped_grads': mean_clipped_grads}

    def apply_update(self, update, metadata={}, *kwargs):
        clipped_grads = update['clipped_grads']
        update = update['grads']
        metadata['update_norm'] = float(grad_norm(update))

        if self.conf.adam_corr:
            update, self.opt_state = self.opt.update((update, clipped_grads), self.opt_state, params=self.theta)
            _, _, _, _, _, _, perc_corr = self.opt_state[2][0]
            metadata['perc_corr1'] = float(perc_corr[0])
            metadata['perc_corr2'] = float(perc_corr[1])
        else:
            update, self.opt_state = self.opt.update(update, self.opt_state, params=self.theta)

        self.theta = optax.apply_updates(self.theta, update)

    # Inherit step() method from DPIterativeTrainer, it calls compute_update and apply_update
    # Ensure DPIterativeTrainer's step method handles privacy accounting correctly. 
    def step(self, metadata={}, *kwargs):
        X, y = next(self.train_loader_itr)
        y = jax.nn.one_hot(y, 10)
        update = self.compute_update(self.theta, X, y, metadata=metadata)

        self.apply_update(update, metadata)

        metadata['learning_rate'] = self.opt_state.hyperparams['learning_rate']

        # Privacy Accountant
        self.privacy_accountant.step(noise_multiplier=self.noise_multiplier,
                                     sample_rate=self.conf.batch_size / len(self.train_loader.dataset))
        eps = self.privacy_accountant.get_epsilon(delta=self.conf.delta)
        metadata['eps'] = eps

        return metadata