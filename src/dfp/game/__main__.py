import logging
import math
import sys
import time
from argparse import Namespace

import numpy as np
import pygame

from ..deploy import deploy_plot_res, main
from ..utils.settings import overwrite_args_with_toml

logging.basicConfig(level=logging.INFO)

args = Namespace(tomlfile="docs/game.toml")
args = overwrite_args_with_toml(args)
finname = "resources/30939153.jpg"


def draw_map(
    map: np.ndarray, win: pygame.Surface, tile_size_w: int, tile_size_h: int
) -> None:
    h, w = map.shape
    for row in range(h):
        for col in range(w):
            color = (
                (100, 100, 100) if map[row, col] == 1.0 else (200, 200, 200)
            )
            pygame.draw.rect(
                win,
                color,
                (
                    col * tile_size_w,
                    row * tile_size_h,
                    tile_size_w,
                    tile_size_h,
                ),
            )


if __name__ == "__main__":
    args.image = finname
    start = time.time()
    result = main(args)
    end = time.time()
    logging.info(f"DFP takes {end-start} s to output map.")

    h, w = result.shape
    bg = rgb_im = np.zeros((h, w, 3))
    bg[result != 10] = [200, 200, 200]
    bg[result == 10] = [100, 100, 100]
    bg = np.transpose(bg, (1, 0, 2))
    SCREEN_HEIGHT = h
    SCREEN_WIDTH = w * 2
    TILE_SIZE = 1
    # pdb.set_trace()
    # deploy_plot_res(result)
    # plt.show()

    pygame.init()
    win = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.font.SysFont("Arial", 18)
    pygame.display.set_caption("Deep Floorplan Raycasting")

    clock = pygame.time.Clock()
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
        # draw_map(result, win, TILE_SIZE, TILE_SIZE)
        # plot floorplan
        surf = pygame.surfarray.make_surface(bg)
        win.blit(surf, (0, 0))

        # plot fps
        fps = str(int(clock.get_fps()))
        fps_text = font.render(fps, 1, pygame.Color("coral"))
        win.blit(fps_text, (10, 0))

        pygame.display.flip()

        clock.tick()

    pygame.quit()
