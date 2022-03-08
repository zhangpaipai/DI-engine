from code import compile_command
from time import sleep
import pytest

from ding.framework.middleware.league_coordinator import LeagueCoordinator
from ding.framework.task import Task
from easydict import EasyDict

from ding.league.v2.base_league import BaseLeague


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

        def start_actors(ctx):
            for i in range(2):
                task.emit("greet_actor", i)
            sleep(0.3)
            assert len(coordinator._actor_jobs) == 2
            task.finish = True

        task.use(coordinator)
        task.use(start_actors)
        task.run(max_step=1)
