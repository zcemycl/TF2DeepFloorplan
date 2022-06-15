import pytest

import numpy as np

from dfp.deploy import colorize, post_process


@pytest.mark.parametrize(
    "h,w,c", [(8, 8, 3), (16, 16, 3), (32, 32, 3), (64, 64, 3)]
)
def test_colorize(h, w, c, mocker):
    inp = np.random.randint(2, size=(h, w))
    out = np.zeros((h, w, c))
    m = mocker.patch("dfp.deploy.ind2rgb")
    m.return_value = out
    r, cw = colorize(inp, inp)
    assert r.shape == (h, w, c) and cw.shape == (h, w, c)


@pytest.mark.parametrize(
    "h,w,c", [(8, 8, 3), (16, 16, 3), (32, 32, 3), (64, 64, 3)]
)
def test_post_process(h, w, c):
    inp = np.ones((h, w, 1))
    r, cw = post_process(inp, inp, [h, w, c])
    assert r.shape == (h, w, 1) and cw.shape == (h, w)
