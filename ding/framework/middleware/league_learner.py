from dataclasses import dataclass
import logging
from time import sleep
from ding.framework.storage.storage import Storage
from ding.worker.learner.base_learner import BaseLearner
from typing import TYPE_CHECKING, Any, Callable, Dict, List
if TYPE_CHECKING:
    from ding.framework import Task, Context
    from ding.utils.log_writer_helper import DistributedWriter
    from ding.framework.middleware.league_actor import ActorData
    from ding.policy import Policy


@dataclass
class LearnerMeta:
    player_id: str
    checkpoint: Storage
    train_iter: int = 0


@dataclass
class LearnerModel:
    player_id: str
    state_dict: Any
    train_iter: int = 0


class LeagueLearner:

    def __init__(self, task: "Task", cfg: dict, policy_fn: Callable, player_id: str) -> None:
        self.task = task
        self.cfg = cfg
        self.policy_fn = policy_fn
        self.player_id = player_id
        self._learner = self._get_learner()
        self.task.on("actor_data_player_".format(player_id), self._on_actor_data)

    def _on_actor_data(self, actor_data: "ActorData"):
        cfg = self.cfg
        for _ in range(cfg.policy.learn.update_per_collect):
            self._learner.train(actor_data.train_data, actor_data.env_step)

        self._learner.policy.state_dict()

    def _get_learner(self) -> BaseLearner:
        cfg = self.cfg
        policy = self.policy_fn().learn_mode
        learner = BaseLearner(
            cfg.policy.learn.learner, policy, exp_name=cfg.exp_name, instance_name=self.player_id + '_learner'
        )
        return learner

    def _save_checkpoint(self) -> None:
        pass

    def __call__(self, _: "Context") -> None:
        while not self.task.finish:
            sleep(1)
