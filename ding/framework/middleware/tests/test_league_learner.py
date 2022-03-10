from time import sleep
from typing import Any
import pytest

import pytest
from copy import deepcopy

from rx import return_value

from ding.framework.middleware.league_actor import ActorData
from ding.framework.middleware.tests.league_config import cfg
from unittest.mock import patch
from ding.framework.storage.file import FileStorage

from ding.league import ActivePlayer
from ding.framework.task import Task
from torch import tensor
from ding.league.player import PlayerMeta
from ding.league.v2.base_league import BaseLeague, Job
from ding.framework.middleware import LeagueLearner
from ding.model import VAC
from ding.policy.ppo import PPOPolicy


def prepare_test():
    global cfg
    cfg = deepcopy(cfg)

    def policy_fn():
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)
        return policy

    train_data = [
        {
            'obs': tensor([0., 1.]),
            'next_obs': tensor([0., 1.]),
            'action': tensor([0]),
            'logit': tensor([0.0842, 0.0347]),
            'value': tensor([-0.0464]),
            'reward': tensor([-20.]),
            'done': True,
            'collect_iter': 0,
            'traj_flag': True,
            'adv': tensor([-20.])
        }
    ] * 128

    league = BaseLeague(cfg.policy.other.league)
    return cfg, policy_fn, league, train_data


@pytest.mark.unittest
def test_league_learner():
    with Task(async_mode=True) as task:
        cfg, policy_fn, league, train_data = prepare_test()
        player: ActivePlayer = league.get_player_by_id("main_player_default_0")
        league_learner = LeagueLearner(task, cfg=cfg, policy_fn=policy_fn, player=player)
        task.use(league_learner)

        testcases = {"learner_player_meta": False, "save_storage": False}

        def on_learner_player_meta(player_meta: PlayerMeta):
            assert player_meta.player_id == player.player_id
            assert player_meta.checkpoint
            assert player_meta.total_agent_step == 120
            testcases["learner_player_meta"] = True

        def save_file(self, data: dict):
            print("Data", type(data))
            print("Data keys", data.keys())
            assert "ckpt.pth" in self.path
            assert data
            testcases["save_storage"] = True

        with patch.object(FileStorage, "save", new=save_file),\
            patch.object(ActivePlayer, "is_trained_enough", return_value=True):
            task.on("learner_player_meta", on_learner_player_meta)
            actor_data = ActorData(env_step=1, train_data=train_data)
            event = "actor_data_player_{}".format(player.player_id)
            task.emit(event, actor_data)

            passed = False
            for _ in range(5):
                if all(testcases.values()):
                    passed = True
                    break
                sleep(1)
            assert passed, testcases
