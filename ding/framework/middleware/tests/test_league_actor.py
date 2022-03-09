from time import sleep
import pytest
from copy import deepcopy
from ding.framework.middleware.tests.league_config import cfg
from ding.framework.middleware.league_actor import ActorData, LeagueActor

from ding.framework.task import Task
from ding.league.v2.base_league import BaseLeague, Job
from ding.model import VAC
from ding.policy.ppo import PPOPolicy
from dizoo.league_demo.game_env import GameEnv


def prepare_test():
    global cfg
    cfg = deepcopy(cfg)

    def env_fn():
        return GameEnv(cfg.env.env_type)

    def policy_fn():
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)
        return policy

    league = BaseLeague(cfg.policy.other.league)
    return cfg, env_fn, policy_fn, league


@pytest.mark.unittest
def test_league_actor():
    cfg, env_fn, policy_fn, league = prepare_test()
    league: BaseLeague
    with Task(async_mode=True) as task:
        league_actor = LeagueActor(task, cfg=cfg, env_fn=env_fn, policy_fn=policy_fn)

        def test_actor():
            job: Job = league.get_job_info()
            testcases = {
                "on_actor_greeting": False,
                "on_actor_job": False,
                "on_actor_data": False,
            }

            def on_actor_greeting(actor_id):
                assert actor_id == task.router.node_id
                testcases["on_actor_greeting"] = True

            def on_actor_job(job_: Job):
                assert job_.launch_player == job.launch_player
                testcases["on_actor_job"] = True

            def on_actor_data(actor_data):
                assert isinstance(actor_data, ActorData)
                testcases["on_actor_data"] = True

            task.on("actor_greeting", on_actor_greeting)
            task.on("actor_job", on_actor_job)
            task.on("actor_data_player_{}".format(job.launch_player), on_actor_data)

            def _test_actor(ctx):
                sleep(0.3)
                task.emit("league_job_actor_{}".format(task.router.node_id), job)
                sleep(0.3)
                try:
                    assert all(testcases.values())
                finally:
                    task.finish = True

            return _test_actor

        task.use(test_actor())
        task.use(league_actor)
        task.run()
