import math
import sys

import pygame

from .model import Model


class Controller:
    def __init__(self, model: Model):
        self.model = model

    def player_control(self):
        keys = pygame.key.get_pressed()
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
