import signal
import sys
from unittest.mock import patch
import pytest
from ding.utils.signal_helper import add_sigint_handler, remove_sigint_handler
import signal
import os


@pytest.mark.unittest
def test_signal_helper():
    flags = {1: 0, 2: 0}

    def handler1(signal, frame):
        flags[1] += 1

    def handler2(signal, frame):
        flags[2] += 1

    with patch.object(sys, "exit", return_value=None):
        add_sigint_handler(1, handler1)
        add_sigint_handler(2, handler2)
        os.kill(os.getpid(), signal.SIGINT)
        assert flags[1] == 1
        assert flags[2] == 1

        remove_sigint_handler(2)
        os.kill(os.getpid(), signal.SIGINT)
        assert flags[1] == 2
        assert flags[2] == 1
