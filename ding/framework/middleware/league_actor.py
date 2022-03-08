from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ding.framework import Task, Context
    from ding.league.base_league import BaseLeague


class LeagueActor:

    def __init__(self) -> None:
        pass

    def on_job(self, job) -> None:
        pass

    def __call__(self, ctx: "Context") -> None:
        pass
