from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .rgb_ind_convertor import floorplan_fuse_map


def export_legend(
    legend: matplotlib.legend.Legend,
    filename: str = "legend.png",
    expand: List[int] = [-5, -5, 5, 5],
):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def norm255to1(x: List[int]) -> List[float]:
    return [p / 255 for p in x]


def handle(m: str, c: List[float]):
    return plt.plot([], [], marker=m, color=c, ls="none")[0]


def main():
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
    colors2 = [norm255to1(rgb) for rgb in list(floorplan_fuse_map.values())]
    handles = [handle("s", colors2[i]) for i in range(len(colors))]
    labels = colors
    legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=True)
    export_legend(legend)


if __name__ == "__main__":
    main()
