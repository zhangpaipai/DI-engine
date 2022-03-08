from dataclasses import dataclass, field
import logging
from typing import List
import uuid
import copy
import os
from abc import abstractmethod
from easydict import EasyDict
import os.path as osp
from ding.framework.storage.file import FileStorage

from ding.league.player import ActivePlayer, HistoricalPlayer, create_player
from ding.league.shared_payoff import create_payoff
from ding.utils import deep_merge_dicts
from ding.league.metric import LeagueMetricEnv
from ding.framework.storage import Storage


@dataclass
class PlayerMeta:
    player_id: str
    checkpoint: Storage


@dataclass
class Job:
    job_id: str
    launch_player: PlayerMeta
    players: List[PlayerMeta]
    result: list = field(default_factory=list)


class BaseLeague:
    """
    Overview:
        League, proposed by Google Deepmind AlphaStar. Can manage multiple players in one league.
    Interface:
        get_job_info, create_historical_player, update_active_player, update_payoff

    .. note::
        In ``__init__`` method, league would also initialized players as well(in ``_init_players`` method).
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        league_type='baseV2',
        import_names=["ding.league.v2.base_league"],
        # ---player----
        # "player_category" is just a name. Depends on the env.
        # For example, in StarCraft, this can be ['zerg', 'terran', 'protoss'].
        player_category=['default'],
        # Support different types of active players for solo and battle league.
        # For solo league, supports ['solo_active_player'].
        # For battle league, supports ['battle_active_player', 'main_player', 'main_exploiter', 'league_exploiter'].
        # active_players=dict(),
        # "use_pretrain" means whether to use pretrain model to initialize active player.
        use_pretrain=False,
        # "use_pretrain_init_historical" means whether to use pretrain model to initialize historical player.
        # "pretrain_checkpoint_path" is the pretrain checkpoint path used in "use_pretrain" and
        # "use_pretrain_init_historical". If both are False, "pretrain_checkpoint_path" can be omitted as well.
        # Otherwise, "pretrain_checkpoint_path" should list paths of all player categories.
        use_pretrain_init_historical=False,
        pretrain_checkpoint_path=dict(default='default_cate_pretrain.pth', ),
        # ---payoff---
        payoff=dict(
            # Supports ['battle']
            type='battle',
            decay=0.99,
            min_win_rate_games=8,
        ),
        metric=dict(
            mu=0,
            sigma=25 / 3,
            beta=25 / 3 / 2,
            tau=0.0,
            draw_probability=0.02,
        ),
    )

    def __init__(self, cfg: EasyDict) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`EasyDict`): League config.
        """
        self.cfg = deep_merge_dicts(self.default_config(), cfg)
        self.active_players = []
        self.historical_players = []
        self.payoff = create_payoff(self.cfg.payoff)
        metric_cfg = self.cfg.metric
        self.metric_env = LeagueMetricEnv(metric_cfg.mu, metric_cfg.sigma, metric_cfg.tau, metric_cfg.draw_probability)
        self._init_players()

    def _init_players(self) -> None:
        """
        Overview:
            Initialize players (active & historical) in the league.
        """
        # Add different types of active players for each player category, according to ``cfg.active_players``.
        for cate in self.cfg.player_category:  # Player's category (Depends on the env)
            for k, n in self.cfg.active_players.items():  # Active player's type
                for i in range(n):  # This type's active player number
                    name = '{}_{}_{}'.format(k, cate, i)
                    ckpt_path = osp.join(self.path_policy, '{}_ckpt.pth'.format(name))
                    player = create_player(
                        self.cfg, k, self.cfg[k], cate, self.payoff, ckpt_path, name, 0, self.metric_env.create_rating()
                    )
                    self.active_players.append(player)
                    self.payoff.add_player(player)

        # Add pretrain player as the initial HistoricalPlayer for each player category.
        if self.cfg.use_pretrain_init_historical:
            for cate in self.cfg.player_category:
                main_player_name = [k for k in self.cfg.keys() if 'main_player' in k]
                assert len(main_player_name) == 1, main_player_name
                main_player_name = main_player_name[0]
                name = '{}_{}_0_pretrain_historical'.format(main_player_name, cate)
                parent_name = '{}_{}_0'.format(main_player_name, cate)
                hp = HistoricalPlayer(
                    self.cfg.get(main_player_name),
                    cate,
                    self.payoff,
                    self.cfg.pretrain_checkpoint_path[cate],
                    name,
                    0,
                    self.metric_env.create_rating(),
                    parent_id=parent_name
                )
                self.historical_players.append(hp)
                self.payoff.add_player(hp)

        # Save active players' ``player_id``
        self.active_players_ids = [p.player_id for p in self.active_players]
        # Validate active players are unique by ``player_id``.
        assert len(self.active_players_ids) == len(set(self.active_players_ids))

    def get_job_info(self, player_id: str = None) -> dict:
        """
        Overview:
            Get info dict of the job which is to be launched to an active player.
        Arguments:
            - player_id (:obj:`str`): The active player's id.
        Returns:
            - job_info (:obj:`dict`): Job info.
        ReturnsKeys:
            - necessary: ``launch_player`` (the active player)
        """
        if player_id is None:
            player_id = self.active_players_ids[0]
        idx = self.active_players_ids.index(player_id)
        player = self.active_players[idx]
        player_meta = PlayerMeta(player_id=player_id, checkpoint=FileStorage(path=player.checkpoint_path))
        player_job_info = player.get_job()
        opponent_player_meta = PlayerMeta(
            player_id=player_job_info.opponent.player_id,
            checkpoint=FileStorage(path=player_job_info.opponent.checkpoint_path)
        )
        job = Job(job_id=str(uuid.uuid1()), launch_player=player_meta, players=[player_meta, opponent_player_meta])
        return job

    def create_historical_player(self, player_id: str, force: bool = False) -> bool:
        """
        Overview:
            Judge whether a player is trained enough for snapshot. If yes, call player's ``snapshot``, create a
            historical player(prepare the checkpoint and add it to the shared payoff), then mutate it, and return True.
            Otherwise, return False.
        Arguments:
            - player_id (:obj:`ActivePlayer`): The active player's id.
        Returns:
            - snapshot_or_not (:obj:`dict`): Whether the active player is snapshotted.
        """
        idx = self.active_players_ids.index(player_id)
        player = self.active_players[idx]
        if force or player.is_trained_enough():
            # Snapshot
            hp = player.snapshot(self.metric_env)
            self.historical_players.append(hp)
            self.payoff.add_player(hp)
            # Mutate
            self._mutate_player(player)
            return True
        else:
            return False

    @abstractmethod
    def _mutate_player(self, player: ActivePlayer) -> None:
        """
        Overview:
            Players have the probability to mutate, e.g. Reset network parameters.
            Called by ``self.judge_snapshot``.
        Arguments:
            - player (:obj:`ActivePlayer`): The active player that may mutate.
        """
        raise NotImplementedError

    def update_active_player(self, player_info: dict) -> None:
        """
        Overview:
            Update an active player's info.
        Arguments:
            - player_info (:obj:`dict`): Info dict of the player which is to be updated.
        ArgumentsKeys:
            - necessary: `player_id`, `train_iteration`
        """
        try:
            idx = self.active_players_ids.index(player_info['player_id'])
            player = self.active_players[idx]
            return self._update_player(player, player_info)
        except ValueError as e:
            logging.error(e)

    @abstractmethod
    def _update_player(self, player: ActivePlayer, player_info: dict) -> None:
        """
        Overview:
            Update an active player. Called by ``self.update_active_player``.
        Arguments:
            - player (:obj:`ActivePlayer`): The active player that will be updated.
            - player_info (:obj:`dict`): Info dict of the active player which is to be updated.
        """
        raise NotImplementedError

    def update_payoff(self, job: Job) -> None:
        """
        Overview:
            Finish current job. Update shared payoff to record the game results.
        Arguments:
            - job_info (:obj:`dict`): A dict containing job result information.
        """
        job_info = {
            "launch_player": job.launch_player.player_id,
            "player_id": list(map(lambda p: p.player_id, job.players)),
            "result": job.result
        }
        self.payoff.update(job_info)
        # Update player rating
        home_id, away_id = job_info['player_id']
        home_player, away_player = self.get_player_by_id(home_id), self.get_player_by_id(away_id)
        job_info_result = job_info['result']
        if isinstance(job_info_result[0], list):
            job_info_result = sum(job_info_result, [])
        home_player.rating, away_player.rating = self.metric_env.rate_1vs1(
            home_player.rating, away_player.rating, result=job_info_result
        )

    def get_player_by_id(self, player_id: str) -> 'Player':  # noqa
        if 'historical' in player_id:
            return [p for p in self.historical_players if p.player_id == player_id][0]
        else:
            return [p for p in self.active_players if p.player_id == player_id][0]
