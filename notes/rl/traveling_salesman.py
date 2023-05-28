import math
import sys
import random
import logging
import numpy as np
import pygame
from numba import jit, njit
from typing import Dict, Tuple, Union


class Game:
    def __init__(self):
        self.h = 600
        self.w = 800
        self.font_color = pygame.Color("coral")
        pygame.init()
        self.win = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Travelling Salesman Game")
        self.clock = pygame.time.Clock()
        self._reset_settings()

    def _reset_settings(self):
        self.startOpt = False
        self.coordinates = set()
        self.c2i = dict()
        self.i2c = dict()

    def show_instructions(self):
        texts = [
            "Left Click: set node location", 
            "Right Click: start optimisation",
            "Key Q: quit game",
            "Key Esc: reset nodes"
        ]
        font = pygame.font.SysFont("Arial", 18)
        for i, text in enumerate(texts):
            screen_text = font.render(text, 1, self.font_color)
            self.win.blit(screen_text, (10, 15*(i+1)))

    def reset_bg(self):
        pygame.draw.rect(self.win, (0, 0, 0), 
            (0, 0, self.w, self.h))
    
    def show_fps(self):
        font = pygame.font.SysFont("Arial", 18)
        fps = str(int(self.clock.get_fps()))
        fps_text = font.render(fps, 1, self.font_color)
        self.win.blit(fps_text, (10, 0))

    def get_player_control(self):
        keys = pygame.key.get_pressed()
        state = pygame.mouse.get_pressed()
        if keys[pygame.K_q]:
            pygame.quit()
            sys.exit(0)
        elif keys[pygame.K_ESCAPE]:
            self._reset_settings()
            
        if state[0] and not self.startOpt:
            pos = pygame.mouse.get_pos()
            if pos not in self.coordinates:
                self.coordinates.add(pos)
                self.c2i[pos] = len(self.c2i)
                self.i2c[len(self.i2c)] = pos
        elif state[2] and not self.startOpt:
            self.startOpt = True


    def draw_travel_routes(self, order):
        for i in range(len(order)):
            pygame.draw.line(self.win, (10, 100, 100),
                self.i2c[order[i]], self.i2c[order[i-1]], 
                width=2)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)

            self.get_player_control()
            self.reset_bg()
            for coor in self.coordinates:
                pygame.draw.circle(self.win, (0, 200, 200), coor, 5)
            if not self.startOpt:
                order = list(range(len(self.coordinates)))
            self.draw_travel_routes(order)
            self.show_fps()
            self.show_instructions()

            pygame.display.flip()
            self.clock.tick()

        pygame.quit()




if __name__ == "__main__":
    env = Game()
    env.run()