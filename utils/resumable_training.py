"""
Resumable Training Callable for Hydra Submitit Launcher

Implements submitit's checkpoint interface for graceful timeout handling
and automatic job resubmission with state preservation.

Usage:
    The ResumableGNOTraining class wraps the training function to enable:
    1. Graceful shutdown on SIGUSR1 (pre-timeout signal from SLURM)
    2. Checkpoint saving before job timeout
    3. Automatic resubmission from saved state via submitit

    When using with Hydra submitit launcher:
    - Set `max_num_timeout` > 0 in launcher config to enable resubmission
    - Set `signal_delay_s` to give time for checkpoint saving before timeout
    - The training loop checks preemption_checker() and saves checkpoint when True
"""

import logging
import signal
from collections.abc import Callable
from typing import Any

import submitit
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ResumableGNOTraining:
    """
    Checkpointable callable for GNO training with submitit.

    This class wraps the training function to enable:
    1. Graceful shutdown on SIGUSR1 (pre-timeout signal)
    2. Checkpoint saving before job timeout
    3. Automatic resubmission from saved state

    Usage with Hydra submitit launcher:
        The launcher will automatically call checkpoint() when the job
        receives SIGUSR1 (signal_delay_s seconds before timeout).

    Example:
        >>> training = ResumableGNOTraining()
        >>> cfg = training(cfg, wandb_run=wandb_run)
    """

    def __init__(self):
        """Initialize with preemption flag and signal handler."""
        self._preemption_requested = False
        self._setup_signal_handler()

    def _setup_signal_handler(self):
        """Register SIGUSR1 handler for preemption signal from SLURM."""

        def handler(signum, frame):
            logger.warning(
                "Received SIGUSR1 - preemption/timeout imminent! "
                "Will save checkpoint at next opportunity."
            )
            self._preemption_requested = True

        # Register handler for SIGUSR1 (SLURM preemption signal)
        signal.signal(signal.SIGUSR1, handler)
        logger.info("Registered SIGUSR1 handler for graceful preemption")

    @property
    def preemption_requested(self) -> bool:
        """Check if preemption has been requested via signal."""
        return self._preemption_requested

    def __call__(
        self,
        cfg: DictConfig,
        wandb_run: Any = None,
        pretrained_checkpoint_path: str | None = None,
    ) -> DictConfig:
        """
        Execute training with resume capability.

        This method imports and calls train_GNO_probe_resumable, passing
        a preemption checker that monitors the SIGUSR1 signal flag.

        Args:
            cfg: Hydra configuration
            wandb_run: Optional W&B run object
            pretrained_checkpoint_path: Optional path for transfer learning

        Returns:
            Updated configuration after training
        """
        from train_GNO_probe import train_GNO_probe_resumable

        # Pass preemption checker to training loop
        return train_GNO_probe_resumable(
            cfg,
            wandb_run=wandb_run,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            preemption_checker=lambda: self._preemption_requested,
        )

    def checkpoint(
        self,
        cfg: DictConfig,
        wandb_run: Any = None,
        pretrained_checkpoint_path: str | None = None,
    ) -> submitit.helpers.DelayedSubmission:
        """
        Called by submitit when job is about to be preempted/timeout.

        This method is invoked when SIGUSR1 is received (signal_delay_s
        before the actual timeout). It:
        1. Sets the preemption flag (which triggers checkpoint save in training loop)
        2. Returns a DelayedSubmission to requeue the job with same arguments

        The actual checkpoint saving happens in the training loop when it
        checks preemption_checker() and sees True.

        Args:
            cfg: Same args as __call__
            wandb_run: Same args as __call__ (will be None on resubmission)
            pretrained_checkpoint_path: Same args as __call__

        Returns:
            DelayedSubmission with same arguments for automatic requeue
        """
        logger.warning("checkpoint() called by submitit - preparing for resubmission")

        # Signal training loop to save checkpoint
        # (The training loop will save and exit cleanly)
        self._preemption_requested = True

        # Create fresh instance for resubmission
        # Note: We don't pickle model state - it's saved in Orbax checkpoint
        # The new instance will auto-detect and load the checkpoint
        training_callable = ResumableGNOTraining()

        return submitit.helpers.DelayedSubmission(
            training_callable,
            cfg,
            wandb_run=None,  # W&B will reinitialize in new job
            pretrained_checkpoint_path=pretrained_checkpoint_path,
        )


def create_preemption_checker() -> tuple[Callable[[], bool], Callable[[int, Any], None]]:
    """
    Create a preemption checker and signal handler pair.

    This is a standalone utility for cases where ResumableGNOTraining
    is not used but preemption checking is still needed.

    Returns:
        Tuple of (preemption_checker, signal_handler)
        - preemption_checker: Callable that returns True if preemption requested
        - signal_handler: Function to register with signal.signal(SIGUSR1, handler)

    Example:
        >>> checker, handler = create_preemption_checker()
        >>> signal.signal(signal.SIGUSR1, handler)
        >>> # In training loop:
        >>> if checker():
        ...     save_checkpoint()
        ...     sys.exit(0)
    """
    preemption_flag = {"requested": False}

    def checker() -> bool:
        return preemption_flag["requested"]

    def handler(signum, frame):
        logger.warning("SIGUSR1 received - preemption requested")
        preemption_flag["requested"] = True

    return checker, handler
