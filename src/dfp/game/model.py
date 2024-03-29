import math
import random
from argparse import Namespace

import numpy as np
import pygame

from ..deploy import main
from ..utils.settings import overwrite_args_with_toml


class Model:
    FOV = math.pi / 3
    HALF_FOV = FOV / 2
    WALL_COLOR = [100, 100, 100]
    FLOOR_COLOR = [200, 200, 200]
    RAY_COLOR = (0, 255, 0)
    SKY_COLOR = (0, 150, 200)
    GRD_COLOR = (100, 100, 100)
    PLAYER_COLOR = (255, 0, 0)
    TILE_SIZE = 1
    dangle = 0.01
    dpos = 1
    GAME_TEXT_COLOR = pygame.Color("coral")
    # surf = None

    def __init__(self, tomlfile: str = "docs/game.toml"):
        self.tomlfile = tomlfile
        args = Namespace(tomlfile=tomlfile)
        self.args = overwrite_args_with_toml(args)
        self.STEP_ANGLE = self.FOV / self.args.casted_rays
        self._initialise_map()
        self._initialise_player_pose()

    def _initialise_map(self):
        self.result = main(self.args)
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
