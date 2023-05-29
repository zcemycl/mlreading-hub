from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import numpy as np
import pygame


class Game:
    h = 600
    w = 800
    nx = 10
    ny = 10

    def __init__(self):
        pygame.init()
        self.font_color = pygame.Color("coral")
        self.win = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Maze")
        self.clock = pygame.time.Clock()

    def reset_bg(self):
        pygame.draw.rect(self.win, (0, 0, 0), (0, 0, self.w, self.h))

    def show_fps(self):
        font = pygame.font.SysFont("Arial", 18)
        fps = str(int(self.clock.get_fps()))
        fps_text = font.render(fps, 1, self.font_color)
        self.win.blit(fps_text, (10, 0))

    # def 


    def run(self):
        box_size = min(self.w//self.nx, self.h//self.ny)
        buttons = []
        colors = []
        for i in range(self.nx):
            for j in range(self.ny):
                rect = pygame.Rect(box_size*i, box_size*j, box_size, box_size)
                buttons.append(rect)
                colors.append((255, 255, 255))
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                elif event.type == pygame.MOUSEMOTION:
                    for i, button in enumerate(buttons):
                        if button.collidepoint(event.pos):
                            colors[i] = (0, 255, 0)
                        else:
                            colors[i] = (255, 255, 255)

            self.reset_bg()
            for i, button in enumerate(buttons):
                pygame.draw.rect(self.win, colors[i], button, 10)
            
            self.show_fps()

            pygame.display.flip()
            self.clock.tick()
        
        pygame.quit()

if __name__ == "__main__":
    env = Game()
    env.run()
            