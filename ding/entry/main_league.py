from socket import timeout
from ding.framework import Task
import logging
import os
import torch

from ding.config import compile_config
from ding.worker import BaseLearner, BattleEpisodeSerialCollector, BattleInteractionSerialEvaluator, NaiveReplayBuffer
from ding.envs import BaseEnvManager
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from ding.league.v2 import BaseLeague
from ding.config.example.league_config import league_config
from ding.utils import DistributedWriter
from dizoo.league_demo.game_env import GameEnv
from ding.framework.middleware import LeagueCoordinator, LeagueActor, LeagueLearner
from rich import print


def main():
    cfg = compile_config(
        league_config,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        BattleEpisodeSerialCollector,
        BattleInteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    from rich import print
    print(cfg)
    exit()

    # policies = {}
    # for player_id in league.active_players_ids:
    #     model = VAC(**cfg.policy.model)
    #     policy = PPOPolicy(cfg.policy, model=model)
    #     policies[player_id] = policy
    # # model = VAC(**cfg.policy.model)
    # policy = PPOPolicy(cfg.policy, model=model)
    # policies['historical'] = policy
    def env_fn():
        return GameEnv(cfg.env.env_type)

    def policy_fn():
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)
        return policy

    with Task(async_mode=False) as task:
        if not task.router.is_active:
            logging.info("League should be executed in parallel mode, use `main_league.sh` to execute league!")
            exit(1)
        league = BaseLeague(cfg.policy.other.league)
        for worker, num in cfg.task.workers.items():
            # TODO Replace with register
            for i in range(num):
                if worker == "league_coordinator":
                    task.use(LeagueCoordinator(task, cfg=cfg, league=league))
                elif worker == "league_actor":
                    task.use(LeagueActor(task, cfg=cfg, env_fn=env_fn, policy_fn=policy_fn))
                elif worker == "league_learner":
                    task.use(LeagueLearner())
                else:
                    raise ValueError("Undefined worker type: {}".format(worker))
        task.run(100)


if __name__ == "__main__":
    main()
