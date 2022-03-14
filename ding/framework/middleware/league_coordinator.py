from time import sleep
from typing import TYPE_CHECKING
from threading import Lock

if TYPE_CHECKING:
    from ding.framework import Task, Context
    from ding.league.v2 import BaseLeague, Job
    from ding.league import PlayerMeta


class LeagueCoordinator:

    def __init__(self, task: "Task", cfg: dict, league: "BaseLeague") -> None:
        self.task = task
        self.cfg = cfg
        self.league = league
        self._job_iter = self._create_job_iter()
        self._lock = Lock()

    def on_actor_greeting(self, actor_id: str):
        self._distribute_job(actor_id)

    def on_learner_player_meta(self, player_meta: "PlayerMeta"):
        self.league.update_active_player(player_meta)
        self.league.create_historical_player(player_meta)

    def on_actor_job(self, job: "Job"):
        self.league.update_payoff(job)

    def _create_job_iter(self) -> "Job":
        i = 0
        while True:
            player_num = len(self.league.active_players_ids)
            player_id = self.league.active_players_ids[i % player_num]
            job = self.league.get_job_info(player_id)
            i += 1
            yield job

    def __call__(self, _: "Context") -> None:
        sleep(1)

    def _distribute_job(self, actor_id: str) -> None:
        with self._lock:
            job = next(self._job_iter)
        job.actor_id = actor_id
        self.task.emit("league_job_actor_{}".format(actor_id), job)
