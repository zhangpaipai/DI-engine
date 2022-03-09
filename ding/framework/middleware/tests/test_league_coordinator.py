from copy import deepcopy
from time import sleep
from typing import List
from unittest.mock import patch
import pytest
from rx import return_value

from ding.framework.middleware.league_coordinator import LeagueCoordinator
from ding.framework.task import Task
from easydict import EasyDict
from ding.league.shared_payoff import BattleSharedPayoff

from ding.league.v2.base_league import BaseLeague, Job


def mock_cfg_league():
    cfg = EasyDict(
        {
            'league': {
                'player_category': ['default'],
                'path_policy': 'league_demo/policy',
                'active_players': {
                    'main_player': 2
                },
                'main_player': {
                    'one_phase_step': 200,
                    'branch_probs': {
                        'pfsp': 0.0,
                        'sp': 1.0
                    },
                    'strong_win_rate': 0.7
                },
                'payoff': {
                    'type': 'battle',
                    'decay': 0.99,
                    'min_win_rate_games': 8
                },
                'metric': {
                    'mu': 0,
                    'sigma': 8.333333333333334,
                    'beta': 4.166666666666667,
                    'tau': 0.0,
                    'draw_probability': 0.02
                }
            },
            'task': {
                'workers': {
                    'league_coordinator': 1,
                    'league_actor': 2,
                    'league_learner': 3
                }
            }
        }
    )
    league = BaseLeague(cfg.league)
    return cfg, league


@pytest.mark.unittest
def test_league_coordinator():
    cfg, league = mock_cfg_league()

    with Task(async_mode=True) as task:
        coordinator = LeagueCoordinator(task, cfg=cfg, league=league)
        task.use(coordinator)

        jobs: List[Job] = []

        def test_greeting():
            """
            When we send greet messages to the coordinator,
            two jobs will be distributed by it.
            """

            def get_job(job):
                jobs.append(job)

            for i in range(2):
                task.on("job_actor_{}".format(i), get_job)

            for i in range(2):
                task.emit("actor_greeting", i)
            sleep(0.3)
            assert len(jobs) == 2
            return jobs

        def test_reply():
            """
            When we send job reply to the coordinator,
            we'll receive two new jobs and the payoff will be upgraded.
            """
            _jobs = deepcopy(jobs)
            jobs.clear()

            def update_payoff(_self, job_info: dict):
                assert "result" in job_info

            with patch.object(BattleSharedPayoff, "update", new=update_payoff):
                player = league.get_player_by_id("main_player_default_0")
                for job in _jobs:
                    if job.launch_player == "main_player_default_0":
                        job.result = ["wins", "wins"]
                    else:
                        job.result = ["losses", "losses"]
                    task.emit("actor_reply", job)
                sleep(0.3)
                assert player.rating.mu != 0
                assert len(jobs) == 2
                assert jobs[0].actor_id != jobs[1].actor_id

        def test_model_meta():
            pass

        test_greeting()
        test_reply()
        test_model_meta()
