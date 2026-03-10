"""GNO probe model utilities and helpers."""

import jax
import jax.numpy as jnp
import jraph
from omegaconf import DictConfig


def scale_rel_ws(graphs, probe_targets, combined_mask, root=10.0):
    """
    Apply relative wind speed scaling using Frederik's technique.

    This scaling is applied on top of regular normalization to improve
    numerical conditioning during training.

    Args:
        graphs: Graph with freestream velocity in globals[0]
        probe_targets: Target velocity values to scale
        combined_mask: Mask indicating valid nodes (wt_mask + probe_mask)
        root: Power root for scaling (default: 10.0)

    Returns:
        Scaled velocity values

    Note:
        Uses sign-preserving power operation: sign(x) * |x|^(1/root)
        This handles negative values correctly since JAX doesn't support
        fractional powers of negative numbers directly.
    """
    probe_targets = 1 - probe_targets / graphs.globals.squeeze()[0]
    # Apply root scaling while preserving sign
    probe_targets = jnp.sign(probe_targets) * (jnp.abs(probe_targets)) ** (1.0 / root)
    probe_targets = probe_targets * combined_mask
    return probe_targets


def inverse_scale_rel_ws(graphs, probe_predictions, combined_mask, root=10.0):
    """
    Inverse of relative wind speed scaling (Frederik's technique).

    Converts scaled predictions back to physical velocity values.

    Args:
        graphs: Graph with freestream velocity in globals[0]
        probe_predictions: Scaled model predictions
        combined_mask: Mask indicating valid nodes (wt_mask + probe_mask)
        root: Power root for scaling (default: 10.0, must match forward scaling)

    Returns:
        Unscaled velocity predictions in physical units

    Note:
        Uses sign-preserving power operation: sign(x) * |x|^root
        This is the mathematical inverse of the forward scaling operation.
    """
    # Inverse root scaling while preserving sign
    probe_predictions = jnp.sign(probe_predictions) * (jnp.abs(probe_predictions)) ** root
    probe_predictions = (1 - probe_predictions) * graphs.globals.squeeze()[0]
    probe_predictions = probe_predictions * combined_mask
    return probe_predictions


def initialize_GNO_probe(
    cfg: DictConfig,
    model,
    rng_key: jax.Array,  # PRNGKey is a function, not a type
    graphs: jraph.GraphsTuple,
    probe_graphs: jraph.GraphsTuple,
    wt_mask: jnp.ndarray,
    probe_mask: jnp.ndarray,
):
    if (
        cfg.model.regularization.encoder_dropout_rate > 0.0
        or cfg.model.regularization.processor_dropout_rate > 0.0
        or cfg.model.regularization.decoder_dropout_rate > 0.0
    ):
        dropout_active = True
        params_key, dropout_key = jax.random.split(rng_key, 2)
        rngs = {"params": params_key, "dropout": dropout_key}
        params = model.init(
            rngs,
            graphs,
            probe_graphs,
            wt_mask,
            probe_mask,
        )
    else:
        dropout_active = False
        params = model.init(
            rng_key,
            graphs,
            probe_graphs,
            wt_mask,
            probe_mask,
            train=True,
        )
    return params, dropout_active
