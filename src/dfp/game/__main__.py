import logging
import math
import random
import sys
import time
from argparse import Namespace
from typing import Dict, Tuple, Union

import numpy as np
import pygame
from numba import jit, njit

from ..deploy import main
from ..utils.settings import overwrite_args_with_toml

logging.basicConfig(level=logging.INFO)

args = Namespace(tomlfile="docs/game.toml")
args = overwrite_args_with_toml(args)
finname = "resources/30939153.jpg"
FOV = math.pi / 3
HALF_FOV = FOV / 2
CASTED_RAYS = 120
STEP_ANGLE = FOV / CASTED_RAYS


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
    keys: Dict[int, bool],
    angle: float,
    x: float,
    y: float,
    w: int,
    h: int,
    binary_map: np.ndarray,
) -> Tuple[float, float, float]:
    dangle = 0.01
    dpos = 1
    origx, origy = x, y
    if keys[pygame.K_LEFT]:
        angle -= dangle
    if keys[pygame.K_RIGHT]:
        angle += dangle
    if keys[pygame.K_w]:
        x -= math.sin(angle) * dpos
        y += math.cos(angle) * dpos
    if keys[pygame.K_s]:
        x += math.sin(angle) * dpos
        y -= math.cos(angle) * dpos
    if keys[pygame.K_d]:
        x += math.sin(angle - math.pi / 2) * dpos
        y -= math.cos(angle - math.pi / 2) * dpos
    if keys[pygame.K_a]:
        x -= math.sin(angle - math.pi / 2) * dpos
        y += math.cos(angle - math.pi / 2) * dpos
    if keys[pygame.K_q]:
        pygame.quit()
        sys.exit(0)
    if x < 0:
        x = 0
    if x >= w:
        x = w - 1
    if y < 0:
        y = 0
    if y >= h:
        y = h - 1
    if binary_map[int(y), int(x)] == 1:
        return angle, origx, origy
    return angle, x, y


@njit
def ray_cast_dda(
    x: float,
    y: float,
    angle: float,
    depth: Union[float, int],
    w: int,
    h: int,
    binary_map: np.ndarray,
) -> Tuple[float, float, bool]:
    vPlayer = np.array([x, y])
    vRayStart = vPlayer.copy()
    vMouseCell = np.array(
        [x - math.sin(angle) * depth, y + math.cos(angle) * depth]
    )
    vRayDir = (vMouseCell - vPlayer) / np.linalg.norm((vMouseCell - vPlayer))
    vRayUnitStepSize = np.array(
        [
            np.sqrt(1 + (vRayDir[1] / (vRayDir[0] + 1e-6)) ** 2),
            np.sqrt(1 + (vRayDir[0] / (vRayDir[1] + 1e-6)) ** 2),
        ]
    )
    vMapCheck = vRayStart.copy()
    vRayLength1D = np.array([0, 0])
    vStep = np.array([0, 0])

    for i in range(2):
        vStep[i] = -1 if vRayDir[i] < 0 else 1
        vRayLength1D[i] = (
            (vRayStart[i] - vMapCheck[i]) * vRayUnitStepSize[i]
            if vRayDir[i] < 0
            else ((vMapCheck[i] + 1) - vRayStart[i]) * vRayUnitStepSize[i]
        )

    bTileFound = False
    fMaxDistance = depth
    fDistance = 0
    while not bTileFound and fDistance < fMaxDistance:
        if vRayLength1D[0] < vRayLength1D[1]:
            vMapCheck[0] += vStep[0]
            fDistance = vRayLength1D[0]
            vRayLength1D[0] += vRayUnitStepSize[0]
        else:
            vMapCheck[1] += vStep[1]
            fDistance = vRayLength1D[1]
            vRayLength1D[1] += vRayUnitStepSize[1]

        if 0 <= vMapCheck[0] < w and 0 <= vMapCheck[1] < h:
            if binary_map[int(vMapCheck[1]), int(vMapCheck[0])] == 1:
                bTileFound = True
        else:
            break

    return angle, fDistance, bTileFound


def draw_player_direction(
    x: float,
    y: float,
    angle: float,
    depth: Union[float, int],
    win: pygame.Surface,
):
    pygame.draw.line(
        win,
        (0, 255, 0),
        (x, y),
        (
            x - math.sin(angle) * depth,
            y + math.cos(angle) * depth,
        ),
        3,
    )


@jit(nopython=True)
def foo(x, y, angle, depth, w, h, binary_map, STEP_ANGLE, CASTED_RAYS):
    return [
        ray_cast_dda(x, y, angle + i * STEP_ANGLE, depth, w, h, binary_map)
        for i in range(CASTED_RAYS)
    ]


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
    result[result != 10] = 0
    result[result == 10] = 1
    bg = np.transpose(bg, (1, 0, 2))
    SCREEN_HEIGHT = h
    SCREEN_WIDTH = w * 2
    TILE_SIZE = 1
    MAX_DEPTH = min(h, w) // 2
    SCALE = (SCREEN_WIDTH / 2) / CASTED_RAYS

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

        # 3d
        pygame.draw.rect(win, (100, 100, 100), (w, h / 2, w, h))
        pygame.draw.rect(win, (0, 150, 200), (w, -h / 2, w, h))

        keys = pygame.key.get_pressed()
        # control
        player_angle, player_x, player_y = player_control(
            keys, player_angle, player_x, player_y, w, h, result
        )

        # draw_map(result, win, TILE_SIZE, TILE_SIZE)
        # draw floorplan
        draw_dfp(bg, win)

        # draw fps
        show_fps(win, clock)

        # draw player
        pygame.draw.circle(win, (255, 0, 0), (player_x, player_y), 8)

        # # draw player direction
        wallangles = foo(
            player_x,
            player_y,
            player_angle - HALF_FOV,
            MAX_DEPTH,
            w,
            h,
            result,
            STEP_ANGLE,
            CASTED_RAYS,
        )

        for idx, (angle, fdist, iswall) in enumerate(wallangles):
            draw_player_direction(player_x, player_y, angle, fdist, win)
            wall_height = 21000 / (fdist + 0.00001)

            if iswall:
                pygame.draw.rect(
                    win,
                    (255, 255, 255),
                    (
                        int(w + idx * SCALE),
                        int((h / 2) - wall_height / 2),
                        int(SCALE),
                        int(wall_height),
                    ),
                )
        pygame.display.flip()

        clock.tick()

    pygame.quit()
