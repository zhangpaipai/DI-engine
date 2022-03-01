from time import sleep
import pytest
from ding.framework import Stream


@pytest.mark.unittest
def test_stream():
    stm = Stream(buffer_size=2).filter(lambda data: data % 2)
    for i in range(6):
        stm.put(i)

    assert stm.last == 5
    assert len(stm.all) == 2

    stm.clear()
    assert len(stm.all) == 0
