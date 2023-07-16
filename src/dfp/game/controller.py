import math
import sys

import numpy as np
import pygame
from scipy.ndimage.morphology import distance_transform_edt

from .model import Model


class Controller:
    def __init__(self, model: Model):
        self.model = model
        self.get_repulsive_field()

    def get_repulsive_field(self):
        tmp_result = self.model.result.copy()
        tmp_result[0, :] = 1
        tmp_result[-1, :] = 1
        tmp_result[:, 0] = 1
        tmp_result[:, -1] = 1
        self.navigate_repulsive_field = distance_transform_edt(1 - tmp_result)
        self.navigate_repulsive_field /= np.max(self.navigate_repulsive_field)
        self.navigate_repulsive_field = 1 / (
            self.navigate_repulsive_field**2 + 0.1
        )
        self.navigate_repulsive_field *= self.model.nu_auto_navigate

    def get_attractive_field(self):
        h, w = self.model.result.shape
        self.x, self.y = np.meshgrid(
            np.linspace(1, w, w), np.linspace(1, h, h)
        )
        self.navigate_attractive_field = 10 * (
            (self.x - self.model.goal[0]) ** 2
            + (self.y - self.model.goal[1]) ** 2
        )
        self.navigate_attractive_field += (
            self.x - self.model.player_x
        ) ** 2 + (self.y - self.model.player_y) ** 2
        self.navigate_attractive_field /= np.max(
            self.navigate_attractive_field
        )
        self.navigate_attractive_field *= self.model.xi_auto_navigate

    def get_navigate_field(self):
        self.model.navigate_field = np.transpose(
            self.navigate_attractive_field + self.navigate_repulsive_field,
            (1, 0),
        )
        # self.model.navigate_field = np.transpose(
        #     self.navigate_attractive_field,
        #     (1, 0),
        # )
        # self.model.navigate_field = np.transpose(
        #     + self.navigate_repulsive_field,
        #     (1, 0),
        # )
        self.model.navigate_field /= np.max(self.model.navigate_field)
        self.model.navigate_field *= 255
        self.model.navigate_field = np.tile(
            np.expand_dims(self.model.navigate_field, axis=2), (1, 1, 3)
        )

    def player_control(self):
        keys = pygame.key.get_pressed()
        if True in keys:
            self.model.auto_navigate = False
        state = pygame.mouse.get_pressed()
        angle, x, y = (
            self.model.player_angle,
            self.model.player_x,
            self.model.player_y,
        )
        origx, origy = x, y
        if keys[pygame.K_LEFT]:
            angle -= self.model.dangle
        if keys[pygame.K_RIGHT]:
            angle += self.model.dangle
        if keys[pygame.K_w]:
            x -= math.sin(angle) * self.model.dpos
            y += math.cos(angle) * self.model.dpos
        if keys[pygame.K_s]:
            x += math.sin(angle) * self.model.dpos
            y -= math.cos(angle) * self.model.dpos
        if keys[pygame.K_d]:
            x += math.sin(angle - math.pi / 2) * self.model.dpos
            y -= math.cos(angle - math.pi / 2) * self.model.dpos
        if keys[pygame.K_a]:
            x -= math.sin(angle - math.pi / 2) * self.model.dpos
            y += math.cos(angle - math.pi / 2) * self.model.dpos
        if keys[pygame.K_q]:
            pygame.quit()
            sys.exit(0)
        if state[0]:
            destination_pos = pygame.mouse.get_pos()
            if (
                destination_pos[0] <= self.model.result.shape[1]
                and destination_pos[1] <= self.model.result.shape[0]
            ):
                self.model.auto_navigate = True
            if self.model.auto_navigate:
                if self.model.goal != destination_pos:
                    self.model.goal = destination_pos  # x,y
                    # get new navigation map
                    self.get_attractive_field()
                    self.get_navigate_field()
                    self.model.navigate_surf = pygame.surfarray.make_surface(
                        self.model.navigate_field
                    ).convert_alpha()

                self.model.goal = destination_pos  # x,y
        if x < 0:
            x = 0
        if x >= self.model.w:
            x = self.model.w - 1
        if y < 0:
            y = 0
        if y >= self.model.h:
            y = self.model.h - 1
        if self.model.result[int(y), int(x)] == 1:
            (
                self.model.player_angle,
                self.model.player_x,
                self.model.player_y,
            ) = (
                angle,
                origx,
                origy,
            )
            return
        self.model.player_angle, self.model.player_x, self.model.player_y = (
            angle,
            x,
            y,
        )
