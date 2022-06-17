from typing import List

import pytest

import numpy as np

from dfp.utils.util import (
    fast_hist,
    fill_break_line,
    flood_fill,
    refine_room_region,
)


@pytest.mark.parametrize("shape", [[16, 16], [32, 32]])
class TestUtilCase:
    def test_fill_break_line(self, shape: List[int]):
        inp = np.ones(shape)
        out = fill_break_line(inp)
        assert out.shape == tuple(shape)

    def test_flood_fill(self, shape: List[int]):
        inp = np.ones(shape)
        inp = np.reshape(inp, (*shape, -1))
        out = flood_fill(inp)
        assert out.shape == tuple(shape)

    def test_refine_room_region(self, shape: List[int]):
        inp = np.random.randint(2, size=shape)
        inp = np.reshape(inp, (*shape, -1))
        out = refine_room_region(inp, inp)
        assert out.shape == (*shape, 1)

    def test_fast_hist(self, shape: List[int]):
        inp = np.random.randint(2, size=shape)
        out = fast_hist(inp, inp)
        assert out.shape == (9, 9)
