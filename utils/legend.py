import matplotlib.pyplot as plt
from rgb_ind_convertor import *

def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

if __name__ == "__main__":
    over255 = lambda x: [p/255 for p in x]
    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
    colors = ["background", "closet", "bathroom",
          "living room\nkitchen\ndining room",
          "bedroom","hall","balcony","not used","not used",
          "door/window","wall"]
    colors2 = [over255(rgb) for rgb in list(floorplan_fuse_map.values())]
    handles = [f("s", colors2[i]) for i in range(len(colors))]
    labels = colors
    legend = plt.legend(handles, labels, loc=3,framealpha=1, frameon=True)
    export_legend(legend)
