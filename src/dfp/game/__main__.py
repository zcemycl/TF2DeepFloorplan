import logging
import math
import random
import sys
import time
from argparse import Namespace
from typing import Dict, Tuple

import numpy as np
import pygame

from ..deploy import main
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


def draw_dfp(map: np.ndarray, win: pygame.Surface):
    surf = pygame.surfarray.make_surface(bg)
    win.blit(surf, (0, 0))


def show_fps(win: pygame.Surface, clock: pygame.time.Clock):
    font = pygame.font.SysFont("Arial", 18)
    fps = str(int(clock.get_fps()))
    fps_text = font.render(fps, 1, pygame.Color("coral"))
    win.blit(fps_text, (10, 0))


def player_control(
    keys: Dict[int, bool], angle: float, x: float, y: float
) -> Tuple[float, float, float]:
    if keys[pygame.K_LEFT]:
        angle -= 0.01
    if keys[pygame.K_RIGHT]:
        angle += 0.01
    if keys[pygame.K_w]:
        x -= math.sin(angle) * 0.1
        y += math.cos(angle) * 0.1
    if keys[pygame.K_s]:
        x += math.sin(angle) * 0.1
        y -= math.cos(angle) * 0.1
    if keys[pygame.K_d]:
        x += math.sin(angle - math.pi / 2) * 0.1
        y -= math.cos(angle - math.pi / 2) * 0.1
    if keys[pygame.K_a]:
        x -= math.sin(angle - math.pi / 2) * 0.1
        y += math.cos(angle - math.pi / 2) * 0.1
    return angle, x, y


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
    FOV = math.pi / 3
    HALF_FOV = FOV / 2
    # pdb.set_trace()
    posy, posx = np.where(result != 10)
    posidx = random.randint(0, len(posy) - 1)
    player_x, player_y = posx[posidx], posy[posidx]
    player_angle = math.pi
    # print()
    # deploy_plot_res(result)
    # plt.show()
    pygame.init()
    win = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Deep Floorplan Raycasting")

    clock = pygame.time.Clock()
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        keys = pygame.key.get_pressed()
        # control
        player_angle, player_x, player_y = player_control(
            keys, player_angle, player_x, player_y
        )

        # if keys[pygame.K_a]: player_
        # draw_map(result, win, TILE_SIZE, TILE_SIZE)
        # draw floorplan
        draw_dfp(bg, win)

        # draw fps
        show_fps(win, clock)

        # draw player
        pygame.draw.circle(win, (255, 0, 0), (player_x, player_y), 8)

        # draw player direction
        pygame.draw.line(
            win,
            (0, 255, 0),
            (player_x, player_y),
            (
                player_x - math.sin(player_angle) * 50,
                player_y + math.cos(player_angle) * 50,
            ),
            3,
        )
        # fov (left, right)
        pygame.draw.line(
            win,
            (0, 255, 0),
            (player_x, player_y),
            (
                player_x - math.sin(player_angle - HALF_FOV) * 50,
                player_y + math.cos(player_angle - HALF_FOV) * 50,
            ),
            3,
        )
        pygame.draw.line(
            win,
            (0, 255, 0),
            (player_x, player_y),
            (
                player_x - math.sin(player_angle + HALF_FOV) * 50,
                player_y + math.cos(player_angle + HALF_FOV) * 50,
            ),
            3,
        )

        pygame.display.flip()

        clock.tick()

    pygame.quit()
