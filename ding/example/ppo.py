import gym
import logging
from ding.model import VAC
from ding.policy import PPOPolicy
from ding.envs import DingEnvWrapper, BaseEnvManager
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import multistep_trainer, StepCollector, interaction_evaluator, CkptSaver, gae_estimator
from ding.utils import set_pkg_seed
from dizoo.classic_control.cartpole.config.cartpole_ppo_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManager(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(8)], cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManager(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(5)], cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(gae_estimator(cfg, policy.collect_mode))
        task.use(multistep_trainer(cfg, policy.learn_mode))
        task.use(CkptSaver(cfg, policy, train_freq=100))
        task.run(max_step=100000)


if __name__ == "__main__":
    main()