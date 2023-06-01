import logging
import math
import sys
import time
from typing import Tuple, Union

import numpy as np
import pygame
from numba import jit, njit

from .controller import Controller
from .model import Model
from .view import View

logging.basicConfig(level=logging.INFO)


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
            np.sqrt(1 + (vRayDir[1] / (vRayDir[0] + 1e-10)) ** 2),
            np.sqrt(1 + (vRayDir[0] / (vRayDir[1] + 1e-10)) ** 2),
        ]
    )
    vMapCheck = vRayStart.copy()
    vRayLength1D = np.array([0.0, 0.0])
    vStep = np.array([0, 0])

    for i in range(2):
        vStep[i] = -1 if vRayDir[i] < 0 else 1
        # vRayLength1D[i] = vRayUnitStepSize[i]
        # vRayLength1D[i] = (
        #     (vRayStart[i] - vMapCheck[i]) * vRayUnitStepSize[i]
        #     if vRayDir[i] < 0
        #     else ((vMapCheck[i] + 1) - vRayStart[i]) * vRayUnitStepSize[i]
        # )

    bTileFound = False
    fMaxDistance = depth
    fDistance = 0
    while not bTileFound and fDistance < fMaxDistance:
        if vRayLength1D[0] < vRayLength1D[1]:
            vMapCheck[0] += vStep[0]
            fDistance = vRayLength1D[0]
            vRayLength1D[0] += vRayUnitStepSize[0]
        elif vRayLength1D[0] > vRayLength1D[1]:
            vMapCheck[1] += vStep[1]
            fDistance = vRayLength1D[1]
            vRayLength1D[1] += vRayUnitStepSize[1]
        elif vRayLength1D[0] == vRayLength1D[1]:
            vMapCheck += vStep
            vRayLength1D += vRayUnitStepSize

        if 0 <= vMapCheck[0] < w and 0 <= vMapCheck[1] < h:
            if binary_map[int(vMapCheck[1]), int(vMapCheck[0])] == 1:
                bTileFound = True
        else:
            break

    return angle, fDistance, bTileFound


@jit(nopython=True)
def loop_rays(x, y, angle, depth, w, h, binary_map, STEP_ANGLE, CASTED_RAYS):
    return [
        ray_cast_dda(x, y, angle + i * STEP_ANGLE, depth, w, h, binary_map)
        for i in range(CASTED_RAYS)
    ]


class Game:
    def __init__(self, tomlfile: str = "docs/game.toml"):
        start = time.time()
        self.model = Model(tomlfile=tomlfile)
        end = time.time()
        logging.info(f"DFP takes {end-start} s to output map.")
        self._initialise_game_engine()
        self.view = View(self.model).register_window(self.win)
        self.control = Controller(self.model)

    def _initialise_game_engine(self):
        pygame.init()
        self.win = pygame.display.set_mode(
            (self.model.SCREEN_WIDTH, self.model.SCREEN_HEIGHT)
        )
        pygame.display.set_caption("Deep Floorplan Raycasting")
        self.clock = pygame.time.Clock()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)

            # 3d background
            self.view.draw_3d_env()
            self.control.player_control()
            self.view.draw_dfp()
            self.view.show_fps(str(int(self.clock.get_fps())))
            self.view.draw_player_loc()

            # calculate player rays
            wallangles = loop_rays(
                self.model.player_x,
                self.model.player_y,
                self.model.player_angle - self.model.HALF_FOV,
                self.model.MAX_DEPTH,
                self.model.w,
                self.model.h,
                self.model.result,
                self.model.STEP_ANGLE,
                self.model.args.casted_rays,
            )

            for idx, (angle, fdist, iswall) in enumerate(wallangles):
                self.view.draw_player_rays(angle, fdist)
                wall_color = int(
                    255 * (1 - (3 * fdist / self.model.MAX_DEPTH) ** 2)
                )
                wall_color = max(min(wall_color, 255), 30)
                fdist *= math.cos(self.model.player_angle - angle)
                wall_height = 21000 / (fdist + 0.00001)

                if iswall:
                    self.view.draw_3d_wall(idx, wall_color, wall_height)
            pygame.display.flip()

            self.clock.tick()
        pygame.quit()


if __name__ == "__main__":
    env = Game()
    env.run()
