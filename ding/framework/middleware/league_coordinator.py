from collections import defaultdict
from dataclasses import dataclass, field
import logging
from time import sleep
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ding.framework import Task, Context
    from ding.league.v2 import BaseLeague, Job


@dataclass
class ActorJob:
    running_jobs: list = field(default_factory=list)
    earliest_job: int = 0


class LeagueCoordinator:

    def __init__(self, task: "Task", cfg: dict, league: "BaseLeague") -> None:
        self.task = task
        self.cfg = cfg
        self.league = league
        self._job_iter = self._create_job_iter()

    def on_greet_actor(self, actor_id):
        self._distribute_job(actor_id)

    def on_model_meta(self, model_meta):
        player_info = {}
        self.league.update_active_player(player_info)

    def on_job_reply(self, job: "Job"):
        actor_id = job.actor_id
        self.league.judge_snapshot(job.player_id)
        job_finish_info = {
            'eval_flag': True,
            'launch_player': job['launch_player'],
            'player_id': job['player_id'],
            'result': [e['result'] for e in job["episode_info"]],
        }
        self.league.finish_job(job_finish_info)
        self._distribute_job(actor_id)

    def _create_job_iter(self):
        i = 0

        def _job_iter() -> "Job":
            nonlocal i
            player_num = len(self.league.active_players_ids)
            player_id = self.league.active_players_ids[i % player_num]
            job = self.league.get_job_info(player_id)
            i += 1
            return job

        return _job_iter

    def __call__(self, ctx: "Context") -> None:
        logging.info("League start on node {}".format(self.task.router.node_id))
        while not self.task.finish:
            sleep(1)

    def _distribute_job(self, actor_id: str) -> None:
        job = self._job_iter()
        job.actor_id = actor_id
        self.task.emit("job_actor_{}".format(actor_id), job)
