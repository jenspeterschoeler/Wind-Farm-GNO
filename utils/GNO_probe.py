import os
import sys
from typing import Tuple

import jax
import jax.numpy as jnp
import jraph
import numpy as np
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf


def scale_rel_ws(graphs, probe_targets, combined_mask, root=10.0):
    """Uses Frederiks technique for scaling of velocities, is used on top of regular scaling"""
    probe_targets = 1 - probe_targets / graphs.globals.squeeze()[0]
    # print(probe_targets)
    # probe_targets = probe_targets**(1 / root)
    probe_targets = jnp.sign(probe_targets) * (jnp.abs(probe_targets)) ** (
        1.0 / root
    )  # it doesn't like negative values with powers, sign + abs work around
    probe_targets = probe_targets * combined_mask
    return probe_targets


def inverse_scale_rel_ws(graphs, probe_predictions, combined_mask, root=10.0):
    """Inverse of Frederiks technique for scaling of velocities, is used on top of regular scaling"""
    probe_predictions = (
        jnp.sign(probe_predictions) * (jnp.abs(probe_predictions)) ** root
    )  # TODO THIS IS PROBABLY BAD
    # probe_predictions = probe_predictions**root
    probe_predictions = (1 - probe_predictions) * graphs.globals.squeeze()[0]
    probe_predictions = probe_predictions * combined_mask
    return probe_predictions


def initialize_GNO_probe(
    cfg: DictConfig,
    model,
    rng_key: jax.random.PRNGKey,
    graphs: jraph.GraphsTuple,
    probe_graphs: jraph.GraphsTuple,
    wt_mask: jnp.ndarray,
    probe_mask: jnp.ndarray,
) -> Tuple[TrainState, jraph.GraphsTuple]:

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
