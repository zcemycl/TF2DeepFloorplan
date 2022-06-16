import os
import unittest

import matplotlib
import matplotlib.pyplot as plt

from dfp.utils.legend import export_legend, handle, main, norm255to1
from dfp.utils.rgb_ind_convertor import floorplan_fuse_map


class TestLegendCase(unittest.TestCase):
    rgbs = list(floorplan_fuse_map.values())
    colors = [
        "background",
        "closet",
        "bathroom",
        "living room\nkitchen\ndining room",
        "bedroom",
        "hall",
        "balcony",
        "not used",
        "not used",
        "door/window",
        "wall",
    ]
    rgbs01 = [norm255to1(rgb) for rgb in rgbs]

    def test_norm255to1(self):
        self.assertEqual(len(self.__class__.rgbs01[0]), 3)

    def test_handle(self):
        self.__class__.handles = [
            handle("s", self.__class__.rgbs01[i])
            for i in range(len(self.__class__.colors))
        ]

        self.assertIsInstance(
            self.__class__.handles[0], matplotlib.lines.Line2D
        )

    def test_export_legend(self):
        handles = [handle("s", [0.5, 0.5, 0.5])]
        legend = plt.legend(
            handles,
            "Leo",
            loc=3,
            framealpha=1,
            frameon=True,
        )
        export_legend(legend, filename="tmp.png")
        ans = os.path.isfile("tmp.png")
        os.system("rm tmp.png")
        self.assertTrue(ans)

    def test_main(self):
        main()
        ans = os.path.isfile("legend.png")
        os.system("rm legend.png")
        self.assertTrue(ans)


if __name__ == "__main__":
    unittest.main()
