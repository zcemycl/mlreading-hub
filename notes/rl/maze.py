from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import numpy as np
import pygame


class GridBox:
    def __init__(self, x: int, y: int, w: int, h: int, i: int, j: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.rect = pygame.Rect(x, y, w, h)
        self.i = i
        self.j = j


class Game:
    h = 600
    w = 800
    nx = 12
    ny = 10

    def __init__(self):
        self.box_size = min(self.w // self.nx, self.h // self.ny)
        self.maze = np.zeros((self.nx, self.ny))
        # -1: hover (green)
        # 0: available (white)
        # 1: start (yellow)
        # 2: end (purple)
        # 3: wall (brown)
        # 4: fire (red)
        # 5: water (blue)
        self.state2color_width = {
            -1: [0, 255, 0, self.box_size],
            0: [0, 0, 0, 5],
            1: [255, 255, 0, self.box_size],
            2: [128, 0, 128, self.box_size],
            3: [150, 75, 0, self.box_size],
            4: [255, 0, 0, self.box_size],
            5: [0, 0, 255, self.box_size],
        }
        self.state2reward = {
            1: -3,
            2: -10,
            3: 1000,
            4: -20000,
            5: -20
        }
        pygame.init()
        self.hover_state = 0
        self.startOpt = False
        self.font_color = pygame.Color("coral")
        self.win = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Maze")
        self.clock = pygame.time.Clock()
        self.buttons = []
        for i in range(self.nx):
            for j in range(self.ny):
                self.buttons.append(
                    GridBox(
                        self.box_size * i,
                        self.box_size * j,
                        self.box_size,
                        self.box_size,
                        i,
                        j,
                    )
                )

    def _reset_settings(self):
        self.maze = np.zeros((self.nx, self.ny))
        self.startOpt = False
        self.hover_state = 0

    def reset_bg(self):
        pygame.draw.rect(self.win, (255, 255, 255), (0, 0, self.w, self.h))
        pygame.draw.rect(
            self.win,
            (128, 128, 128),
            (self.box_size * self.nx, 0, self.w, self.h),
        )

    def show_fps(self):
        font = pygame.font.SysFont("Arial", 18)
        fps = str(int(self.clock.get_fps()))
        fps_text = font.render(fps, 1, self.font_color)
        self.win.blit(fps_text, (10, 0))

    def get_player_control(self):
        keys = pygame.key.get_pressed()
        state = pygame.mouse.get_pressed()

        if not self.startOpt:
            if keys[pygame.K_q]:
                pygame.quit()
                sys.exit(0)
            elif keys[pygame.K_0]:
                self.hover_state = 0
            elif keys[pygame.K_1]:
                self.hover_state = 1
            elif keys[pygame.K_2]:
                self.hover_state = 2
            elif keys[pygame.K_3]:
                self.hover_state = 3
            elif keys[pygame.K_4]:
                self.hover_state = 4
            elif keys[pygame.K_5]:
                self.hover_state = 5
            elif keys[pygame.K_RETURN]:
                self.startOpt = True

        if keys[pygame.K_ESCAPE]:
            self._reset_settings()

        if not self.startOpt and state[0]:
            pos = pygame.mouse.get_pos()
            for i, button in enumerate(self.buttons):
                if button.rect.collidepoint(pos):
                    if self.hover_state in [1, 2]:
                        self.maze[self.maze == self.hover_state] = 0
                    self.maze[button.i, button.j] = self.hover_state

    def run(self):
        tar_i = -1
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                elif event.type == pygame.MOUSEMOTION:
                    for tar_i, button in enumerate(self.buttons):
                        if button.rect.collidepoint(event.pos):
                            break

            self.get_player_control()
            self.reset_bg()
            for j, button in enumerate(self.buttons):
                if j == tar_i:
                    color_width = self.state2color_width[-1]
                else:
                    color_width = self.state2color_width[
                        self.maze[button.i, button.j]
                    ]
                pygame.draw.rect(
                    self.win, color_width[:3], button.rect, color_width[3]
                )
            self.show_fps()

            pygame.display.flip()
            self.clock.tick()

        pygame.quit()


if __name__ == "__main__":
    env = Game()
    env.run()
