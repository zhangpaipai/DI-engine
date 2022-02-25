from collections import deque
from typing import Any, Callable, List


class Stream:
    """
    This class caches data in a queue and provides a series of functional interfaces to manipulate the queue.
    But note that it differs from the replay buffer in that it will only focus on caching data and stream processing,
    rather than integrating deeply with the algorithm as the replay buffer does.
    """

    def __init__(self, buffer_size: int = 1):
        self._queue = deque(maxlen=buffer_size)
        self._filter = None

    def put(self, data: Any) -> "Stream":
        """
        Overview:
            Put data into the queue.
        Arguments:
            - data (:obj:`Any`): The data.
        Returns:
            - self (:obj:`Stream`): Self stream instance.
        """
        if self._filter and not self._filter(data):
            return self
        self._queue.append(data)
        return self

    def filter(self, fn: Callable) -> "Stream":
        """
        Overview:
            Register a filter before data put into the queue.
        Arguments:
            - fn (:obj:`Callable`): The filter function.
        Returns:
            - self (:obj:`Stream`): Self stream instance.
        """
        self._filter = fn
        return self

    @property
    def last(self) -> Any:
        """
        Overview:
            Get last data from the queue.
        Returns:
            - data (:obj:`Any`): Last data from the queue.
        """
        if self._queue:
            return self._queue[-1]

    @property
    def all(self) -> List[Any]:
        """
        Overview:
            Get all data from the queue
        Returns:
            - queue (:obj:`List[Any]`): All data from the queue, converted into a list.
        """
        return list(self._queue)
