import logging
from time import sleep
from ding.envs.env_manager.base_env_manager import BaseEnvManager
from ding.worker.collector.battle_episode_serial_collector import BattleEpisodeSerialCollector
from dizoo.league_demo.game_env import GameEnv
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ding.framework import Task, LeagueContext
    from ding.utils.log_writer_helper import DistributedWriter


def league_collector(task: "Task", cfg: dict, tb_logger: "DistributedWriter", policies: dict):
    player_ids = list(policies.keys())
    collectors = {}
    for player_id in player_ids:
        collector_env = BaseEnvManager(
            env_fn=[lambda: GameEnv(cfg.env.env_type) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        collector_env.seed(0)
        collectors[player_id] = BattleEpisodeSerialCollector(
            cfg.policy.collect.collector,
            collector_env,
            tb_logger=tb_logger,
            exp_name=cfg.exp_name,
            instance_name=player_id + '_colllector',
        )

    def _collect(ctx: "LeagueContext"):
        job = ctx.job
        logging.info("Collecting on node {}".format(task.router.node_id))

        # Reset policies
        main_player_id = job["player_id"][0]
        opponent_player_id = job['player_id'][1]
        opponent_policy = policies[opponent_player_id].collect_mode
        main_policy = policies[main_player_id].collect_mode
        collector = collectors[main_player_id]
        collector.reset_policy([main_policy, opponent_policy])

        # Collect data
        train_data, episode_info = collector.collect()  # TODO Do we need train_iter?
        train_data, episode_info = train_data[0], episode_info[0]  # only use main player data for training
        ctx.episode_info = episode_info
        for d in train_data:
            d['adv'] = d['reward']

        collect_output = {"job": job, "train_data": train_data, "env_step": collector.envstep}
        task.emit("collect_output", collect_output)  # Shoot and forget

    return _collect
