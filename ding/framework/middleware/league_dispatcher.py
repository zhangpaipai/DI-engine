from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ding.framework import Task, LeagueContext
    from ding.league.base_league import BaseLeague


@dataclass
class Job:
    player_id: str


def league_dispatcher(task: "Task", league: "BaseLeague"):

    def _league(ctx: "LeagueContext"):
        logging.info("League dispatching on node {}".format(task.router.node_id))
        # Random pick a player
        i = ctx.total_step % len(league.active_players_ids)
        player_id = league.active_players_ids[i]

        # Get players of both side
        logging.info("Get player {}".format(player_id))
        job = league.get_job_info(player_id)
        # Job example
        # {
        #     'agent_num': 2,
        #     'launch_player': 'main_exploiter_default_0',
        #     'player_id': ['main_exploiter_default_0', 'main_player_default_0'],
        #     'checkpoint_path': [
        #         'league_demo_ppo/policy/main_exploiter_default_0_ckpt.pth',
        #         'league_demo_ppo/policy/main_player_default_0_ckpt.pth'
        #     ],
        #     'player_active_flag': [True, True]
        # }
        logging.info("Job: {}".format(job))

        ctx.job = job

        yield

        # league.update_active_player(ctx.player_info)
        # league.judge_snapshot(ctx.job["player_id"])
        job_finish_info = {
            'eval_flag': True,
            'launch_player': job['launch_player'],
            'player_id': job['player_id'],
            'result': [e['result'] for e in ctx.episode_info],
        }
        # league.finish_job(job_finish_info)
        logging.info("League finish info: {}".format(job_finish_info))

    return _league
