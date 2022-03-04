from time import sleep
from typing import List
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ding.framework import Task, Context


def sync_context(task: "Task", send_keys: List[str] = None, recv_keys: List[str] = None):
    send_keys = send_keys or []
    recv_keys = recv_keys or []
    event = "sync_context"

    def filter_payload(payload: dict):
        return any([payload.get(key) for key in recv_keys])

    payload_stm = task.stream(event, buffer_size=1).filter(filter_payload)

    def _sync_context(ctx: "Context"):
        while recv_keys:
            if payload_stm.last:
                payload: dict = payload_stm.last
                payload_stm.clear()
                for key in recv_keys:
                    setattr(ctx, key, payload.get(key))
                break
            else:
                sleep(0.01)

        yield

        payload = {}
        for key in send_keys:
            payload[key] = getattr(ctx, key)

        if payload:
            task.emit("sync_context", payload, only_remote=True)

    return _sync_context
