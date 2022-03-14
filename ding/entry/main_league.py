from ding.framework import Task
import logging
from functools import partial

from ding.config import compile_config
from ding.worker import BaseLearner, BattleEpisodeSerialCollector, BattleInteractionSerialEvaluator, NaiveReplayBuffer
from ding.envs import BaseEnvManager
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from ding.league.v2 import BaseLeague
from ding.config.example.league_config import league_config
from dizoo.league_demo.game_env import GameEnv
from ding.framework.middleware import LeagueCoordinator, LeagueActor, LeagueLearner


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

    def env_fn():
        env = BaseEnvManager(
            env_fn=[lambda: GameEnv(cfg.env.env_type) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        env.seed(cfg.seed)
        return env

    def policy_fn(enable_field):
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model, enable_field=enable_field)
        return policy

    with Task(async_mode=False) as task:
        if not task.router.is_active:
            logging.info("League should be executed in parallel mode, use `main_league.sh` to execute league!")
            exit(1)
        league = BaseLeague(cfg.policy.other.league)
        i = -1
        for worker, num in cfg.task.workers.items():
            # TODO Replace with register
            for _ in range(num):
                i += 1
                if task.router.node_id != i:
                    continue
                if worker == "league_coordinator":  # One league
                    task.use(LeagueCoordinator(task, cfg=cfg, league=league))
                elif worker == "league_actor":  # All actors is the same
                    task.use(
                        LeagueActor(
                            task, cfg=cfg, env_fn=env_fn, policy_fn=partial(policy_fn, enable_field=['collect'])
                        )
                    )
                elif worker == "league_learner":  # On player on each learner
                    n_players = len(league.active_players_ids)
                    player = league.active_players[task.router.node_id % n_players]
                    task.use(
                        LeagueLearner(
                            task, cfg=cfg, policy_fn=partial(policy_fn, enable_field=['learn']), player=player
                        )
                    )
                else:
                    raise ValueError("Undefined worker type: {}".format(worker))
        task.run()


if __name__ == "__main__":
    main()
