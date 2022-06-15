import unittest

import numpy as np

from dfp.utils.rgb_ind_convertor import floorplan_fuse_map, ind2rgb, rgb2ind


class TestRgbIndConvertorCase(unittest.TestCase):
    def test_rgb2ind(self):
        inp = np.array([[[255, 60, 128], [192, 255, 255]]])
        expected = [9, 2]
        out = rgb2ind(inp, floorplan_fuse_map)
        out = list(out[0])
        self.assertListEqual(out, expected)

    def test_ind2rgb(self):
        inp = np.array([[9, 2]])
        expected = np.array([[[255, 60, 128], [192, 255, 255]]])
        out = ind2rgb(inp, floorplan_fuse_map)
        self.assertSequenceEqual(out.tolist(), expected.tolist())


if __name__ == "__main__":
    unittest.main()
