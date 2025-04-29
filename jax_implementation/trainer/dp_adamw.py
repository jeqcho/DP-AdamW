import functools
import jax
import jax.numpy as jnp
import optax
from optax._src import base, combine, numerics, utils
from optax._src.transform import bias_correction, update_moment, update_moment_per_elem_norm
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

# Adapting ScaleByAdamStateCorrLong and scale_by_adam_corr from dp_adambc.py
# Keep the structure for moment tracking and bias correction
class ScaleByAdamWState(NamedTuple):
    count: chex.Array
    mu: base.Updates
    nu: base.Updates
    # nu_corr: base.Updates # Retained for compatibility if needed, but focus is decoupled decay
    mu_clean: base.Updates
    nu_clean: base.Updates
    perc_corr: Optional[tuple] # For logging bias correction effect

def scale_by_adamw(
        batch_size: int,
        dp_noise_multiplier: float,
        dp_l2_norm_clip: float,
        b1: float,
        b2: float,
        eps: float, # Adam epsilon
        eps_root: float, # Epsilon for corrected nu
        mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    """Rescales updates by the Adam algorithm with DP noise correction.

    References:
      [AdamW Paper](https://arxiv.org/abs/1711.05101)
      Based on scale_by_adam_corr from dp_adambc.py for DP logic.

    Args:
      batch_size: Batch size for noise variance calculation.
      dp_noise_multiplier: Noise multiplier for DP.
      dp_l2_norm_clip: Clipping bound for DP.
      b1: Exponential decay rate for the first moment estimates.
      b2: Exponential decay rate for the second moment estimates.
      eps: Term added to the denominator to improve numerical stability.
      eps_root: Floor value for the corrected second moment estimate.
      mu_dtype: Optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
      A `GradientTransformation` object.
    """
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)
        # nu_corr = jax.tree_util.tree_map(jnp.zeros_like, params) # Potentially remove if unused
        mu_clean = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu_clean = jax.tree_util.tree_map(jnp.zeros_like, params)
        return ScaleByAdamWState(
            count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, # nu_corr=nu_corr,
            mu_clean=mu_clean, nu_clean=nu_clean, perc_corr=None
        )

    def update_fn(updates, state, params=None):
        # `params` are needed for the weight decay term, but Optax handles
        # weight decay separately, so we don't use `params` here directly.
        del params

        # DPAdam logic: updates contain (noised_updates, clean_clipped_updates)
        noised_updates, clean_updates = updates

        # Update moments with noised gradients
        mu = update_moment(noised_updates, state.mu, b1, 1)
        nu = update_moment_per_elem_norm(noised_updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)

        # Bias correction for moments
        mu_hat = bias_correction(mu, b1, count_inc)
        nu_hat = bias_correction(nu, b2, count_inc)

        # DP Noise correction for second moment (from dp_adambc.py)
        noise_err = (1 / batch_size ** 2) * dp_noise_multiplier ** 2 * dp_l2_norm_clip ** 2
        nu_hat_corr = jax.tree_map(lambda x: jnp.maximum(x - noise_err, eps_root), nu_hat)

        # Calculate the Adam update *without* weight decay (as per AdamW)
        # Use the DP-corrected second moment estimate
        updates = jax.tree_util.tree_map(
            lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat, nu_hat_corr)

        # Update clean moments (using clipped, unnoised gradients) for logging/analysis
        mu_clean = update_moment(clean_updates, state.mu_clean, b1, 1)
        nu_clean = update_moment_per_elem_norm(clean_updates, state.nu_clean, b2, 2)
        # mu_hat_clean = bias_correction(mu_clean, b1, count_inc) # Optional: if needed for logging
        # nu_hat_clean = bias_correction(nu_clean, b2, count_inc) # Optional: if needed for logging

        # Logging percentage of corrected nu (from dp_adambc.py)
        # Use tree_flatten_1dim from trainer.utils instead of optax's non-existent tree_flatten_float_dtype
        num_corr1 = jnp.sum(tree_flatten_1dim(jax.tree_map(lambda x: jnp.sum((x - noise_err) > eps_root), nu_hat)))
        dummy_count = jnp.sum(tree_flatten_1dim(jax.tree_map(lambda x: jnp.sum(~jnp.isnan(x)), nu_hat)))
        perc_corr1 = num_corr1 / dummy_count
        perc_corr2 = 0 # Placeholder, adjust if needed

        mu = utils.cast_tree(mu, mu_dtype)
        new_state = ScaleByAdamWState(
            count=count_inc, mu=mu, nu=nu, # nu_corr=nu_hat_corr, # Store corrected nu if needed
            mu_clean=mu_clean, nu_clean=nu_clean, perc_corr=(perc_corr1, perc_corr2)
        )
        return updates, new_state

    return base.GradientTransformation(init_fn, update_fn)

def dp_adamw(
    learning_rate: optax.ScalarOrSchedule,
    batch_size: int,
    dp_noise_multiplier: float,
    dp_l2_norm_clip: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 5e-8,
    eps_root: float = 5e-8,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 0.0, # AdamW specific
) -> base.GradientTransformation:
    """AdamW optimizer with DP noise correction.

    Combines the Adam scaling logic (with DP corrections) and decoupled weight decay.

    Args:
      learning_rate: A fixed global scaling factor.
      batch_size: Batch size for noise variance calculation.
      dp_noise_multiplier: Noise multiplier for DP.
      dp_l2_norm_clip: Clipping bound for DP.
      b1: Exponential decay rate for the first moment estimates.
      b2: Exponential decay rate for the second moment estimates.
      eps: Term added to the denominator to improve numerical stability.
      eps_root: Floor value for the corrected second moment estimate.
      mu_dtype: Optional `dtype` to be used for the first order accumulator.
      weight_decay: Strength of the decoupled weight decay regularization.

    Returns:
      A `GradientTransformation` object.
    """
    return combine.chain(
        # This transformation handles the Adam moment updates with DP correction
        scale_by_adamw(
            batch_size=batch_size,
            dp_noise_multiplier=dp_noise_multiplier,
            dp_l2_norm_clip=dp_l2_norm_clip,
            b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
        # This transformation applies the decoupled weight decay
        optax.add_decayed_weights(weight_decay=weight_decay),
        # Use the internal scaling function, matching dp_adambc.py
        _scale_by_learning_rate(learning_rate)
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
        # self.helper_loader = self.loader(self.train_set) # If needed, uncomment
        # self.helper_loader_itr = iter(Cycle(self.helper_loader))
        self.setup_optimizer_adamw()  # Use the new setup method

    def setup_optimizer_adamw(self):
        """Sets up the DPAdamW optimizer using optax."""
        # Use dp_adamw optax chain, including weight decay
        # The --adam_corr flag enables the bias correction within scale_by_adamw
        if not self.conf.adam_corr:
             # Potentially add a version of dp_adamw without bias correction if needed
             print("Warning: Running DPAdamW without --adam_corr (bias correction) is not fully implemented in this structure yet.")
             # Fallback to a simpler AdamW for now or raise error
             # For now, we assume --adam_corr is expected for the DP bias correction part

        self.opt = optax.inject_hyperparams(dp_adamw)(
            learning_rate=self.lr,
            batch_size=self.conf.batch_size,
            dp_noise_multiplier=self.noise_multiplier,
            dp_l2_norm_clip=self.conf.dp_l2_norm_clip,
            b1=self.conf.beta_1,
            b2=self.conf.beta_2,
            eps=self.conf.eps,
            eps_root=self.conf.eps_root,
            weight_decay=self.conf.weight_decay # AdamW specific
            # mu_dtype can be added if needed
        )

        if self.conf.reload_ckpt_path is None:
            self.opt_state = self.opt.init(self.theta)
        else:
            self.opt_state = self.opt_state_restored # Assumes state structure matches

    # --- Copy and potentially adapt compute_update and apply_update --- 
    # --- from DPAdamTrainer in dp_adambc.py                   ---

    def compute_update(self, theta, X, y, metadata={}, *kwargs):
        """Computes per-example clipped grads and optionally adds noise."""
        batch_size = X.shape[0]

        if self.conf.virtual_batch_size is None:
            virtual_batch_size = batch_size
        else:
            virtual_batch_size = self.conf.virtual_batch_size

        virtual_batch_num = math.ceil(batch_size / virtual_batch_size)
        grads_sum_clipped = None # Sum of clipped gradients
        loss = 0.
        for i in range(virtual_batch_num):
            start = i * virtual_batch_size
            end = (i + 1) * virtual_batch_size
            # Use self.dp_grads from DPIterativeTrainer base class
            per_example_clipped_grads, losses, _sum_clipped, \
                _dp_grads_only_clipped, _dp_grads_only_unclipped, \
                clip_mask, grad_norms = self.dp_grads(
                    theta, X[start:end], y[start:end],
                    l2_norm_clip=self.conf.dp_l2_norm_clip, save_clip_unclip=False # Set save_clip_unclip=True for debugging
                )
            loss += float(jnp.sum(losses))

            if grads_sum_clipped is None:
                grads_sum_clipped = _sum_clipped
            else:
                grads_sum_clipped = jax.tree_util.tree_map(jnp.add, grads_sum_clipped, _sum_clipped)

        # Average the summed clipped gradients
        mean_clipped_grads = jax.tree_map(lambda x: x / batch_size, grads_sum_clipped)

        # Add noise to the *summed* clipped gradients before averaging for DP guarantee
        # Note: The DPAdamTrainer implementation seemed to add noise *after* averaging.
        # Standard DP-SGD adds noise to the sum. Let's follow that.
        # However, the scale_by_adamw expects the per-batch average noise variance correction.
        # Let's stick to the DPAdamTrainer structure for consistency for now.

        # Add noise to the *averaged* clipped gradients (like DPAdamTrainer)
        noised_mean_clipped_grads, noise_tree = noise_and_normalize(
            grads_sum_clipped, # Pass the sum here
            self.conf.dp_l2_norm_clip,
            self.noise_multiplier,
            batch_size, # Normalize by batch size inside noise_and_normalize
            next(self.prng_seq),
            save_noise_tree=False
        )

        metadata["loss"] = float(loss / batch_size)

        # Return both noised and clean (but clipped) averaged gradients
        # This matches the input format expected by scale_by_adamw's update_fn
        return {'noised_grads': noised_mean_clipped_grads, 'clipped_grads': mean_clipped_grads}

    def apply_update(self, update_dict, metadata={}, *kwargs):
        """Applies the update using the DPAdamW optimizer."""
        noised_grads = update_dict['noised_grads']
        clipped_grads = update_dict['clipped_grads'] # Clean, clipped grads for bias correction tracking

        metadata['update_norm'] = float(grad_norm(noised_grads))

        # Pass the tuple (noised_grads, clipped_grads) to the optimizer
        # dp_adamw internally uses scale_by_adamw which expects this tuple
        updates, self.opt_state = self.opt.update((noised_grads, clipped_grads), self.opt_state, self.theta) # Pass params for weight decay

        # Log bias correction percentage if available in state
        # Assuming the state structure follows ScaleByAdamWState and is nested within opt_state
        # This might need adjustment based on optax.inject_hyperparams structure
        try:
            # Accessing state: opt_state -> ScaleByAdamWState (likely first element after inject_hyperparams)
            # -> perc_corr attribute
            if isinstance(self.opt_state[0], ScaleByAdamWState):
                 perc_corr = self.opt_state[0].perc_corr
                 if perc_corr is not None:
                     metadata['perc_corr1'] = float(perc_corr[0])
                     metadata['perc_corr2'] = float(perc_corr[1]) # Adjust if only one value is stored
        except (AttributeError, IndexError, TypeError):
             pass # Ignore if state structure is different or perc_corr not found

        self.theta = optax.apply_updates(self.theta, updates)

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