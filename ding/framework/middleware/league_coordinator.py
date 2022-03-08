from collections import defaultdict
from dataclasses import dataclass
import logging
from time import sleep
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ding.framework import Task, Context
    from ding.league.v2 import BaseLeague, Job


@dataclass
class ActorJob:
    running_num: int = 0
    last_run_ts: int = 0


class LeagueCoordinator:

    def __init__(self, task: "Task", cfg: dict, league: "BaseLeague") -> None:
        self.task = task
        self.cfg = cfg
        self.league = league
        self._job_iter = self._create_job_iter()
        self._actor_jobs = defaultdict(ActorJob)
        self._max_job_num_for_each_actor = 3

    def on_greet_actor(self, actor_id):
        self._actor_jobs[actor_id] = 0

    def on_model_meta(self, model_meta):
        player_info = {}
        self.league.update_active_player(player_info)

    def on_job_reply(self, job):
        actor_id, job_id = job["actor_id"], job["job_id"]
        self.league.judge_snapshot(job["player_id"])
        job_finish_info = {
            'eval_flag': True,
            'launch_player': job['launch_player'],
            'player_id': job['player_id'],
            'result': [e['result'] for e in job["episode_info"]],
        }
        self.league.finish_job(job_finish_info)

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
        actor_num = self.cfg.task.workers.league_actor
        while len(self._actor_jobs) < actor_num:
            sleep(0.01)

        for actor_id in self._actor_jobs:
            job = self._job_iter()
            job["actor_id"] = i
            self._job_balancer["actor_{}".format(i)][job["job_id"]] = job
            self.task.emit("job_actor_{}".format(i), job)

        while not self.task.finish:
            sleep(1)
