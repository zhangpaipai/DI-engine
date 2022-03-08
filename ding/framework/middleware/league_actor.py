from time import sleep
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ding.framework import Task, Context


class LeagueActor:

    def __init__(self, task: "Task", cfg: dict) -> None:
        self.task = task

    def on_job(self, job) -> None:
        pass

    def __call__(self, ctx: "Context") -> None:
        self.task.emit("greet_actor", self.task.router.node_id)
        while not self.task.finish:
            sleep(1)
