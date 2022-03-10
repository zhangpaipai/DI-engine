from dataclasses import dataclass
import logging
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Dict
from ding.envs.env_manager.base_env_manager import BaseEnvManager
from ding.worker.collector.battle_episode_serial_collector import BattleEpisodeSerialCollector
from ding.league import PlayerMeta
if TYPE_CHECKING:
    from ding.framework import Task, Context
    from ding.league.v2.base_league import Job
    from ding.policy import Policy
    from ding.framework.middleware.league_learner import LearnerModel


@dataclass
class ActorData:
    train_data: Any
    env_step: int = 0


class LeagueActor:

    def __init__(self, task: "Task", cfg: dict, env_fn: Callable, policy_fn: Callable) -> None:
        self.cfg = cfg
        self.task = task
        self.env_fn = env_fn
        self.policy_fn = policy_fn
        self._running = False
        self._collectors: Dict[str, BattleEpisodeSerialCollector] = {}
        self._policies: Dict[str, "Policy.collect_function"] = {}
        self._model_updated = True
        self.task.on("league_job_actor_{}".format(self.task.router.node_id), self._on_league_job)
        self.task.on("learner_model", self._on_learner_model)

    def _on_learner_model(self, learner_model: "LearnerModel"):
        player_meta = PlayerMeta(player_id=learner_model.player_id, checkpoint=None)
        policy = self._get_policy(player_meta)
        policy.load_state_dict(learner_model.state_dict)
        self._model_updated = True

    def _on_league_job(self, job: "Job") -> None:
        """
        Deal with job distributed by coordinator
        """
        self._running = True

        # Wait new active model for 10 seconds
        for _ in range(10):
            if self._model_updated:
                self._model_updated = False
                break
            logging.info(
                "Waiting for new model on actor: {}, player: {}".format(self.task.router.node_id, job.launch_player)
            )
            sleep(1)

        collector = self._get_collector(job.launch_player)
        policies = []
        main_player: "PlayerMeta" = None
        for player in job.players:
            policies.append(self._get_policy(player))
            if player.player_id == job.launch_player:
                main_player = player

        assert main_player, "Can not find active player"
        collector.reset_policy(policies)
        train_data, episode_info = collector.collect(train_iter=main_player.total_agent_step)
        train_data, episode_info = train_data[0], episode_info[0]  # only use main player data for training
        for d in train_data:
            d["adv"] = d["reward"]

        actor_data = ActorData(env_step=collector.envstep, train_data=train_data)
        self.task.emit("actor_data_player_{}".format(job.launch_player), actor_data)

        job.result = [e['result'] for e in episode_info]
        self.task.emit("actor_job", job)

        self._running = False

    def _get_collector(self, player_id: str) -> BattleEpisodeSerialCollector:
        if self._collectors.get(player_id):
            return self._collectors.get(player_id)
        cfg = self.cfg
        env = self.env_fn()
        collector = BattleEpisodeSerialCollector(
            cfg.policy.collect.collector,
            env,
            exp_name=cfg.exp_name,
            instance_name=player_id + '_collector',
        )
        self._collectors[player_id] = collector
        return collector

    def _get_policy(self, player: "PlayerMeta") -> "Policy.collect_function":
        player_id = player.player_id
        if self._policies.get(player_id):
            return self._policies.get(player_id)
        policy: "Policy.collect_function" = self.policy_fn().collect_mode
        self._policies[player_id] = policy
        if "historical" in player.player_id:
            policy.load_state_dict(player.checkpoint.load())

        return policy

    def __call__(self, _: "Context") -> None:
        """
        Send heartbeat to coordinator until receiving job
        """
        if not self._running:
            self.task.emit("actor_greeting", self.task.router.node_id)
        sleep(1)
