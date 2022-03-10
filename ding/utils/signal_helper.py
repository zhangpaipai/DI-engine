import sys
from typing import Any, Callable
import signal

sigint_handlers = {}


def sigint_handler(signal: int, frame: Any):
    """
    Overview:
        Execute all the signal handlers, exit with 1.
    Arguments:
        - signal (:obj:`int`): Signal number.
        - frame (:obj:`frame`): Signal frame.
    """
    for fn in sigint_handlers.values():
        fn(signal, frame)
    sys.exit(1)


def add_sigint_handler(id_: int, handler: Callable):
    """
    Overview:
        Global signal handler register, support binding multiple handlers on one signal
    Arguments:
        - id_ (:obj:`int`): Id.
        - handler (:obj:`Callable`): Callback handler.
    """
    if not sigint_handlers:
        signal.signal(signal.SIGINT, sigint_handler)
    sigint_handlers[id_] = handler


def remove_sigint_handler(id_: int):
    """
    Overview:
        Global signal handler register, support binding multiple handlers on one signal
    Arguments:
        - id_ (:obj:`int`): Id.
    """
    del sigint_handlers[id_]
