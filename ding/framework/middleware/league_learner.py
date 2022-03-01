import logging
from time import sleep
from ding.worker.learner.base_learner import BaseLearner
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from ding.framework import Task, Context
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

    collect_stm = task.stream("collect_output")\
        .filter(lambda data: data["job"]["launch_player"] == player_id)

    def _learn(ctx: "Context"):
        while True:
            # if collect_stm.last:
            #     collect_output = collect_stm.last
            #     collect_stm.clear()
            #     break
            # else:
            #     sleep(0.01)
            collect_output = task.wait_for("collect_output")[0][0]
            job = collect_output["job"]
            if job["launch_player"] == player_id:
                break

        logging.info("Learning on node: {}, player: {}".format(task.router.node_id, player_id))
        train_data, env_step = collect_output["train_data"], collect_output["env_step"]

        for _ in range(cfg.policy.learn.update_per_collect):
            learner.train(train_data, env_step)

        state_dict = learner.policy.state_dict()

        learn_output = {
            "player_info": learner.learn_info,
            "player_id": player_id,
            "train_iter": learner.train_iter,
            "state_dict": state_dict
        }

        sleep(1)
        task.emit("learn_output", learn_output)  # Broadcast to other middleware

    return _learn
