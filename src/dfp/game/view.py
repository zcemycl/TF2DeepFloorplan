from __future__ import annotations

import math
from typing import Union

import pygame

from .model import Model


class View:
    def __init__(self, model: Model):
        self.model = model

    def register_window(self, win: pygame.Surface) -> View:
        self.win = win
        return self

    def draw_dfp(self):
        self.win.blit(self.model.surf, (0, 0))
        if self.model.auto_navigate and self.model.navigate_surf:
            self.win.blit(self.model.navigate_surf, (0, 0))

    def show_fps(self, fps: str):
        font = pygame.font.SysFont("Arial", 18)
        fps_text = font.render(fps, 1, self.model.GAME_TEXT_COLOR)
        self.win.blit(fps_text, (10, 0))

    def draw_player_loc(self):
        pygame.draw.circle(
            self.win,
            self.model.PLAYER_COLOR,
            (self.model.player_x, self.model.player_y),
            self.model.PLAYER_SIZE,
        )

    def draw_player_rays(
        self,
        angle: float,
        depth: Union[float, int],
    ):
        pygame.draw.line(
            self.win,
            self.model.RAY_COLOR,
            (self.model.player_x, self.model.player_y),
            (
                self.model.player_x - math.sin(angle) * depth,
                self.model.player_y + math.cos(angle) * depth,
            ),
            3,
        )

    def draw_3d_env(self):
        pygame.draw.rect(
            self.win,
            self.model.GRD_COLOR,
            (self.model.w, self.model.h / 2, self.model.w, self.model.h),
        )
        pygame.draw.rect(
            self.win,
            self.model.SKY_COLOR,
            (self.model.w, -self.model.h / 2, self.model.w, self.model.h),
        )

    def draw_3d_wall(
        self, idx: int, wall_color: int, wall_height: Union[int, float]
    ):
        pygame.draw.rect(
            self.win,
            (wall_color, wall_color, wall_color),
            (
                int(self.model.w + idx * self.model.SCALE),
                int((self.model.h / 2) - wall_height / 2),
                self.model.SCALE,
                wall_height,
            ),
        )
