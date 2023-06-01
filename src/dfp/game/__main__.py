import logging
import math
import random
import sys
import time
from argparse import Namespace
from typing import Tuple, Union

import numpy as np
import pygame
from numba import jit, njit

from ..deploy import main
from ..utils.settings import overwrite_args_with_toml

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
    FOV = math.pi / 3
    HALF_FOV = FOV / 2
    WALL_COLOR = [100, 100, 100]
    FLOOR_COLOR = [200, 200, 200]
    RAY_COLOR = (0, 255, 0)
    TILE_SIZE = 1
    dangle = 0.01
    dpos = 1
    GAME_TEXT_COLOR = pygame.Color("coral")

    def __init__(self, tomlfile: str = "docs/game.toml"):
        self.tomlfile = tomlfile
        args = Namespace(tomlfile=tomlfile)
        self.args = overwrite_args_with_toml(args)
        self.STEP_ANGLE = self.FOV / self.args.casted_rays
        self._initialise_map()
        self._initialise_player_pose()
        self._initialise_game_engine()

    def _initialise_map(self):
        start = time.time()
        self.result = main(self.args)
        end = time.time()
        logging.info(f"DFP takes {end-start} s to output map.")
        self.h, self.w = self.result.shape
        self.bg = self.rgb_im = np.zeros((self.h, self.w, 3))
        self.bg[self.result != 10] = self.FLOOR_COLOR
        self.bg[self.result == 10] = self.WALL_COLOR
        self.result[self.result != 10] = 0
        self.result[self.result == 10] = 1
        self.bg = np.transpose(self.bg, (1, 0, 2))
        self.MAX_DEPTH = max(self.h, self.w)
        self.SCREEN_HEIGHT = self.h
        self.SCREEN_WIDTH = self.w * 2
        self.SCALE = (self.SCREEN_WIDTH / 2) / self.args.casted_rays
        self.surf = pygame.surfarray.make_surface(self.bg)

    def _initialise_player_pose(self):
        posy, posx = np.where(self.result != 1)
        posidx = random.randint(0, len(posy) - 1)
        self.player_x, self.player_y = posx[posidx], posy[posidx]
        self.player_angle = math.pi

    def _initialise_game_engine(self):
        pygame.init()
        self.win = pygame.display.set_mode(
            (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        )
        pygame.display.set_caption("Deep Floorplan Raycasting")
        self.clock = pygame.time.Clock()

    def draw_dfp(self):
        self.win.blit(self.surf, (0, 0))

    def show_fps(self):
        font = pygame.font.SysFont("Arial", 18)
        fps = str(int(self.clock.get_fps()))
        fps_text = font.render(fps, 1, self.GAME_TEXT_COLOR)
        self.win.blit(fps_text, (10, 0))

    def player_control(self):
        keys = pygame.key.get_pressed()
        angle, x, y = self.player_angle, self.player_x, self.player_y
        origx, origy = x, y
        if keys[pygame.K_LEFT]:
            angle -= self.dangle
        if keys[pygame.K_RIGHT]:
            angle += self.dangle
        if keys[pygame.K_w]:
            x -= math.sin(angle) * self.dpos
            y += math.cos(angle) * self.dpos
        if keys[pygame.K_s]:
            x += math.sin(angle) * self.dpos
            y -= math.cos(angle) * self.dpos
        if keys[pygame.K_d]:
            x += math.sin(angle - math.pi / 2) * self.dpos
            y -= math.cos(angle - math.pi / 2) * self.dpos
        if keys[pygame.K_a]:
            x -= math.sin(angle - math.pi / 2) * self.dpos
            y += math.cos(angle - math.pi / 2) * self.dpos
        if keys[pygame.K_q]:
            pygame.quit()
            sys.exit(0)
        if x < 0:
            x = 0
        if x >= self.w:
            x = self.w - 1
        if y < 0:
            y = 0
        if y >= self.h:
            y = self.h - 1
        if self.result[int(y), int(x)] == 1:
            self.player_angle, self.player_x, self.player_y = (
                angle,
                origx,
                origy,
            )
            return
        self.player_angle, self.player_x, self.player_y = angle, x, y

    def draw_player_rays(
        self,
        angle: float,
        depth: Union[float, int],
    ):
        pygame.draw.line(
            self.win,
            self.RAY_COLOR,
            (self.player_x, self.player_y),
            (
                self.player_x - math.sin(angle) * depth,
                self.player_y + math.cos(angle) * depth,
            ),
            3,
        )

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)

            # 3d background
            pygame.draw.rect(
                self.win, (100, 100, 100), (self.w, self.h / 2, self.w, self.h)
            )
            pygame.draw.rect(
                self.win, (0, 150, 200), (self.w, -self.h / 2, self.w, self.h)
            )

            self.player_control()
            self.draw_dfp()
            self.show_fps()

            # draw player
            pygame.draw.circle(
                self.win, (255, 0, 0), (self.player_x, self.player_y), 8
            )

            # calculate player rays
            wallangles = loop_rays(
                self.player_x,
                self.player_y,
                self.player_angle - self.HALF_FOV,
                self.MAX_DEPTH,
                self.w,
                self.h,
                self.result,
                self.STEP_ANGLE,
                self.args.casted_rays,
            )

            for idx, (angle, fdist, iswall) in enumerate(wallangles):
                self.draw_player_rays(angle, fdist)
                wall_color = int(255 * (1 - (3 * fdist / self.MAX_DEPTH) ** 2))
                wall_color = max(min(wall_color, 255), 30)
                fdist *= math.cos(self.player_angle - angle)
                wall_height = 21000 / (fdist + 0.00001)

                if iswall:
                    pygame.draw.rect(
                        self.win,
                        (wall_color, wall_color, wall_color),
                        (
                            int(self.w + idx * self.SCALE),
                            int((self.h / 2) - wall_height / 2),
                            self.SCALE,
                            wall_height,
                        ),
                    )
            pygame.display.flip()

            self.clock.tick()
        pygame.quit()


if __name__ == "__main__":
    env = Game()
    env.run()
