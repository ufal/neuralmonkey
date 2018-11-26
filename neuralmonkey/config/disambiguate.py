"""Module for disambiguating and enhancing configuration."""

from argparse import Namespace
from datetime import timedelta
import re
import time
from typing import List, Union, Callable

import numpy as np

from neuralmonkey.dataset import BatchingScheme
from neuralmonkey.logging import warn
from neuralmonkey.tf_manager import get_default_tf_manager
from neuralmonkey.trainers.delayed_update_trainer import DelayedUpdateTrainer


def disambiguate_configuration(cfg: Namespace, train_mode: bool) -> None:

    if train_mode:
        _disambiguate_train_cfg(cfg)

    if cfg.tf_manager is None:
        cfg.tf_manager = get_default_tf_manager()

    if (cfg.batch_size is None) == (cfg.batching_scheme is None):
        raise ValueError("You must specify either batch_size or "
                         "batching_scheme (not both).")

    if cfg.batch_size is not None:
        assert cfg.batching_scheme is None
        cfg.batching_scheme = BatchingScheme(batch_size=cfg.batch_size)
    else:
        assert cfg.batching_scheme is not None
        cfg.batch_size = cfg.batching_scheme.batch_size

    if cfg.runners_batch_size is None:
        cfg.runners_batch_size = cfg.batching_scheme.batch_size

    cfg.runners_batching_scheme = BatchingScheme(
        batch_size=cfg.runners_batch_size,
        token_level_batching=cfg.batching_scheme.token_level_batching,
        use_leftover_buckets=True)

    cfg.evaluation = [(e[0], e[0], e[1]) if len(e) == 2 else e
                      for e in cfg.evaluation]

    if cfg.evaluation:
        cfg.main_metric = "{}/{}".format(cfg.evaluation[-1][0],
                                         cfg.evaluation[-1][-1].name)
    else:
        cfg.main_metric = "{}/{}".format(cfg.runners[-1].decoder_data_id,
                                         cfg.runners[-1].loss_names[0])

        if not cfg.tf_manager.minimize_metric:
            raise ValueError("minimize_metric must be set to True in "
                             "TensorFlowManager when using loss as "
                             "the main metric")


def _disambiguate_train_cfg(cfg: Namespace) -> None:

    if not isinstance(cfg.val_dataset, List):
        cfg.val_datasets = [cfg.val_dataset]
    else:
        cfg.val_datasets = cfg.val_dataset

    if not isinstance(cfg.trainer, List):
        cfg.trainers = [cfg.trainer]
    else:
        cfg.trainers = cfg.trainer

    # deal with delayed trainer and logging periods
    # the correct way if there are more trainers is perhaps to do a
    # lowest common denominator of their batches_per_update.
    # But we can also warn because it is a very weird setup.

    delayed_trainers = [t for t in cfg.trainers
                        if isinstance(t, DelayedUpdateTrainer)]

    denominator = 1
    if len(cfg.trainers) > 1 and delayed_trainers:
        warn("Weird setup: using more trainers and one of them is delayed "
             "update trainer. No-one can vouch for your safety, user!")
        warn("Using the lowest common denominator of all delayed trainers'"
             " batches_per_update parameters for logging period")
        warn("Note that if you are using a multi-task trainer, it is on "
             "your own risk")

        denominator = np.lcm.reduce([t.batches_per_update
                                     for t in delayed_trainers])
    elif delayed_trainers:
        assert len(cfg.trainers) == 1
        denominator = cfg.trainers[0].batches_per_update

    cfg.log_timer = _resolve_period(cfg.logging_period, denominator)
    cfg.val_timer = _resolve_period(cfg.validation_period, denominator)


def _resolve_period(period: Union[str, int],
                    denominator: int) -> Callable[[int, float], bool]:

    def get_batch_logger(period: int) -> Callable[[int, float], bool]:
        def is_time(step: int, _: float) -> bool:
            return step != 0 and step % period == 0
        return is_time

    def get_time_logger(period: float) -> Callable[[int, float], bool]:
        def is_time(step: int, last_time: float) -> bool:
            if step % denominator != 0:
                return False
            return last_time + period < time.process_time()
        return is_time

    if isinstance(period, int):
        if period % denominator != 0:
            raise ValueError(
                "When using delayed update trainer, the logging/validation "
                "periods must be divisible by batches_per_update.")

        return get_batch_logger(period)

    regex = re.compile(
        r"((?P<days>\d+?)d)?((?P<hours>\d+?)h)?((?P<minutes>\d+?)m)?"
        r"((?P<seconds>\d+?)s)?")
    parts = regex.match(period)

    if not parts:
        raise ValueError(
            "Validation or logging period have incorrect format. "
            "It should be in format: 3h; 5m; 14s")

    time_params = {}
    for (name, param) in parts.groupdict().items():
        if param:
            time_params[name] = int(param)

    delta_seconds = timedelta(**time_params).total_seconds()
    if delta_seconds <= 0:
        raise ValueError("Validation or logging period must be bigger than 0")

    return get_time_logger(delta_seconds)
