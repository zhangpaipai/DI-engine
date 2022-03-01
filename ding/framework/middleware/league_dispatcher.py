import logging
import random
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ding.framework import Task, LeagueContext
    from ding.league.base_league import BaseLeague


def league_dispatcher(task: "Task", league: "BaseLeague", policies: dict):

    def update_learn_output(learn_output):
        logging.info("Get lern output {}".format(learn_output["player_info"]))
        player_info, player_id, state_dict = learn_output["player_info"], learn_output["player_id"], learn_output[
            "state_dict"]
        policies[player_id]._model.load_state_dict(state_dict)
        league.update_active_player(player_info)
        league.judge_snapshot(player_id)

    task.on("learn_output", update_learn_output)

    # Wait for all players online
    online_learners = {}
    for player_id in league.active_players_ids:
        online_learners[player_id] = False

    def learner_online(player_id):
        online_learners[player_id] = True

    task.on("learner_online", learner_online)

    def win_loss_result(player_id, result):
        player = online_learners[player_id]
        player.rating = league.metric_env.rate_1vsC(
            player.rating, league.metric_env.create_rating(mu=10, sigma=1e-8), result
        )

    task.on("win_loss_result", win_loss_result)

    def _league(ctx: "LeagueContext"):
        logging.info("League dispatching on node {}".format(task.router.node_id))
        # Random pick a player
        i = random.choice(range(len(league.active_players_ids)))
        player_id = league.active_players_ids[i]

        # Get players of both side
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

        job_finish_info = {
            'eval_flag': True,
            'launch_player': job['launch_player'],
            'player_id': job['player_id'],
            'result': [e['result'] for e in ctx.episode_info],
        }

        league.finish_job(job_finish_info)

    return _league
