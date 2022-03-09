import os
import logging
from time import sleep
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ding.framework import Task, Context
    from ding.league.v2 import BaseLeague, Job
    from ding.framework.middleware.league_learner import LearnerMeta


class LeagueCoordinator:

    def __init__(self, task: "Task", cfg: dict, league: "BaseLeague") -> None:
        self.task = task
        self.cfg = cfg
        self.league = league
        self._job_iter = self._create_job_iter()

    def on_actor_greeting(self, actor_id: str):
        self._distribute_job(actor_id)

    def on_learner_meta(self, learner_meta: "LearnerMeta"):
        self.league.update_active_player(learner_meta.player_id, learner_meta.train_iter)
        self.league.create_historical_player(learner_meta.player_id, learner_meta.checkpoint)

    def on_actor_job(self, job: "Job"):
        actor_id = job.actor_id
        self.league.update_payoff(job)
        self._distribute_job(actor_id)

    def _create_job_iter(self) -> "Job":
        i = 0
        while True:
            player_num = len(self.league.active_players_ids)
            player_id = self.league.active_players_ids[i % player_num]
            job = self.league.get_job_info(player_id)
            i += 1
            yield job

    def __call__(self, _: "Context") -> None:
        logging.info("League start on node {}".format(self.task.router.node_id))
        while not self.task.finish:
            sleep(1)

    def _distribute_job(self, actor_id: str) -> None:
        job = next(self._job_iter)
        job.actor_id = actor_id
        # sleep(0.3)
        print("Distribute new job to actor {}".format(actor_id), os.getpid())
        print(self.league.payoff)
        self.task.emit("league_job_actor_{}".format(actor_id), job)
