import logging
from time import sleep
from ding.worker.learner.base_learner import BaseLearner
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from ding.framework import Task, LeagueContext
    from ding.utils.log_writer_helper import DistributedWriter


def league_learner(task: "Task", cfg: dict, tb_logger: "DistributedWriter", player_id: str, policies: List[str]):
    policy = policies[player_id]
    learner = BaseLearner(
        cfg.policy.learn.learner,
        policy.learn_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        instance_name=player_id + '_learner'
    )

    def _learn(ctx: "LeagueContext"):
        if ctx.job["launch_player"] != player_id:
            return
        logging.info("Learning on node: {}, player: {}".format(task.router.node_id, player_id))
        train_data, env_step = ctx.train_data, ctx.env_step

        for _ in range(cfg.policy.learn.update_per_collect):
            learner.train(train_data, env_step)

        # state_dict = learner.policy.state_dict()
        ctx.player_info = learner.learn_info
        ctx.player_info['player_id'] = player_id

    return _learn
