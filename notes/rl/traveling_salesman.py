from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import numpy as np
import pygame


class Gene(object):
    mixp = 0.5
    mutp = 0.1

    def __init__(self, length: int):
        self.length = length
        self.dist = np.inf
        self.seq = list(range(1, length))
        np.random.shuffle(self.seq)
        self.seq.append(0)

    def __repr__(self):
        return f"<Gene seq: {self.seq}, dist: {self.dist}>"

    def __str__(self):
        return f"<Gene seq: {self.seq}, dist: {self.dist}>"

    @classmethod
    def mix(cls, dna1: List[int], dna2: List[int]) -> Gene:
        new = cls(len(dna1))
        new.seq = dna1.copy()
        for i in range(new.length - 1):
            if np.random.rand() <= new.mixp:
                prev = new.seq[i]
                inx = new.seq.index(dna2[i])
                new.seq[inx] = prev
                new.seq[i] = dna2[i]

        copyarr = new.seq.copy()
        for i in range(new.length - 1):
            if np.random.rand() <= new.mutp:
                newn = np.random.randint(1, new.length)
                # np.random.randint()
                inx = copyarr.index(newn)
                prev = copyarr[i]
                copyarr[inx] = prev
                copyarr[i] = newn
        new.seq = copyarr

        return new

    @classmethod
    def mutate(cls, length: int) -> Gene:
        new = cls(length)
        copyarr = new.seq.copy()
        for i in range(new.length - 1):
            if np.random.rand() <= new.mutp:
                newn = np.random.randint(1, new.length)
                # np.random.randint()
                inx = copyarr.index(newn)
                prev = copyarr[i]
                copyarr[inx] = prev
                copyarr[i] = newn
        new.seq = copyarr

        return new


class GeneticAlgo:
    def __init__(
        self,
        length: int,
        i2c: Dict[int, Tuple[int, int]],
        size=20,
        survive_ratio=0.5,
        mix_ratio=0.25,
    ):
        self.length = length
        self.size = size
        self.i2c = i2c
        self.maxDist = np.inf
        self.survive_ratio = survive_ratio
        self.mix_ratio = mix_ratio
        self.population = [Gene(self.length) for _ in range(self.size)]
        self.calc_fitness()
        self.population = sorted(self.population, key=lambda x: x.dist)

    def compute_dist(self, seq: List[int]) -> int:
        dist = 0
        prev = seq[-1]
        for i in range(len(seq)):
            cur = seq[i]
            dist += np.sqrt(
                (self.i2c[cur][0] - self.i2c[prev][0]) ** 2
                + (self.i2c[cur][1] - self.i2c[prev][1]) ** 2
            )
            prev = cur
        return dist

    def calc_fitness(self):
        for i in range(self.size):
            tmp_route = self.population[i]
            tmp_route.dist = self.compute_dist(self.population[i].seq)
            self.population[i] = tmp_route

    def step(self):
        remain_size = int(self.survive_ratio * self.size)
        self.population = self.population[:remain_size]
        for i in range(int(self.mix_ratio * self.size)):
            n1 = np.random.randint(0, remain_size)
            n2 = np.random.randint(0, remain_size)
            while n1 == n2:
                n2 = np.random.randint(0, remain_size)
            self.population.append(
                Gene.mix(self.population[n1].seq, self.population[n2].seq)
            )

        while len(self.population) <= self.size:
            self.population.append(Gene.mutate(self.length))
        self.calc_fitness()
        self.population = sorted(self.population, key=lambda x: x.dist)


class Game:
    def __init__(self, size=20, survive_ratio=0.5, mix_ratio=0.25):
        self.h = 600
        self.w = 800
        self.size = size
        self.survive_ratio = survive_ratio
        self.mix_ratio = mix_ratio
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
            "Key Esc: reset nodes",
        ]
        font = pygame.font.SysFont("Arial", 18)
        for i, text in enumerate(texts):
            screen_text = font.render(text, 1, self.font_color)
            self.win.blit(screen_text, (10, 15 * (i + 1)))

    def reset_bg(self):
        pygame.draw.rect(self.win, (0, 0, 0), (0, 0, self.w, self.h))

    def show_fps(self):
        font = pygame.font.SysFont("Arial", 18)
        fps = str(int(self.clock.get_fps()))
        fps_text = font.render(fps, 1, self.font_color)
        self.win.blit(fps_text, (10, 0))

    def show_dist(self, dist: int):
        font = pygame.font.SysFont("Arial", 18)
        screen_text = font.render(f"{dist}", 1, self.font_color)
        self.win.blit(screen_text, (self.w - 100, 0))

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
            self.policy = GeneticAlgo(
                len(self.coordinates),
                self.i2c,
                size=self.size,
                survive_ratio=self.survive_ratio,
                mix_ratio=self.mix_ratio,
            )

    def draw_travel_routes(self, order):
        for i in range(len(order)):
            pygame.draw.line(
                self.win,
                (10, 100, 100),
                self.i2c[order[i]],
                self.i2c[order[i - 1]],
                width=2,
            )

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
            else:
                self.policy.step()
                order = self.policy.population[0].seq
                self.show_dist(self.policy.population[0].dist)
            self.draw_travel_routes(order)
            self.show_fps()
            self.show_instructions()

            pygame.display.flip()
            self.clock.tick()

        pygame.quit()


if __name__ == "__main__":
    env = Game(size=100, survive_ratio=0.2, mix_ratio=0.4)
    env.run()
