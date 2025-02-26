import pygame
import neat
import os
import random
from pygame.locals import *
from game import flappy_bird_utils
from itertools import cycle

FPS = 30
# 游戏设置
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
# PIPE_HEIGHT = 400
GAP = 100  # pipe gap

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird AI')

IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
BASEY = SCREEN_HEIGHT * 0.79

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])
# 游戏中的鸟类
class Bird:
    def __init__(self):

        self.x = int(SCREEN_WIDTH * 0.2)
        self.y = int((SCREEN_HEIGHT - PLAYER_HEIGHT) / 2)
        self.vel = 0
        self.height = PLAYER_HEIGHT
        self.width = PLAYER_WIDTH
        self.is_jumping = False
        self.index = 0
        self.loopIter = 0

    def move(self):
        if self.is_jumping:
            self.vel = -10
            self.is_jumping = False
        self.vel += 1  # gravity
        self.y += self.vel

        if self.y > SCREEN_HEIGHT - self.height:
            self.y = SCREEN_HEIGHT - self.height
            self.vel = 0
        if self.y < 0:
            self.y = 0
            self.vel = 0

        if (self.loopIter + 1) % 3 == 0:
            self.index = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30

    def jump(self):
        self.is_jumping = True

    def get_state(self, pipes):
        closest_pipe = None
        for pipe in pipes:
            if pipe.x + pipe.width > self.x:
                closest_pipe = pipe
                break
        return closest_pipe

    def colision(self, pipe):

        uPipeRect = pygame.Rect(pipe.x, pipe.top - PIPE_HEIGHT, PIPE_WIDTH, PIPE_HEIGHT)
        lPipeRect = pygame.Rect(pipe.x, pipe.bottom, PIPE_WIDTH, PIPE_HEIGHT)

        playerRect = pygame.Rect(self.x, self.y, self.width, self.height)

        # player and upper/lower pipe hitmasks
        pHitMask = HITMASKS['player'][self.index]
        uHitmask = HITMASKS['pipe'][0]
        lHitmask = HITMASKS['pipe'][1]

        # if bird collided with upipe or lpipe
        uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
        lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

        if uCollide or lCollide:
            return True
        return False

# 游戏中的管道
class Pipe:
    def __init__(self):
        self.x = SCREEN_WIDTH
        self.height = random.randint(100, 300)
        self.gap = GAP
        self.width = PIPE_WIDTH
        self.top = self.height
        self.bottom = self.height + self.gap

    def move(self):
        self.x -= 5

    def is_offscreen(self):
        return self.x < -self.width

    def is_need_new(self):
        return self.x < SCREEN_WIDTH / 2


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False
    # print(rect,rect1,rect2)
    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0  # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREEN_WIDTH - totalWidth) / 2

    for digit in scoreDigits:
        screen.blit(IMAGES['numbers'][digit], (Xoffset, SCREEN_HEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()

# 创建环境并运行
def run_game(genomes, config):
    nets = []
    ge = []
    birds = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird())
        ge.append(genome)
        genome.fitness = 0

    pipes = [Pipe()]
    c_pipe = pipes[0]
    f_pipe = pipes[0]
    clock = pygame.time.Clock()
    score = 0
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == QUIT:
                done = True

        for i, bird in enumerate(birds):
            state = bird.get_state(pipes)
            output = nets[i].activate((bird.y, state.top, state.bottom, state.x - bird.x))
            if output[0] > 0:
                bird.jump()

            bird.move()
            ge[i].fitness += 0.1  # Increase fitness over time

        if c_pipe.is_need_new():
            c_pipe = Pipe()
            pipes.append(c_pipe)
        if f_pipe.x + f_pipe.width / 2 < birds[0].x:
            f_pipe = c_pipe
            score += 1
        for pipe in pipes:
            pipe.move()
            if pipe.is_offscreen():
                pipes.remove(pipe)
                # score += 1
                # print(score)

        # Check collisions
        for i, bird in enumerate(birds):
            state = bird.get_state(pipes)
            if state and bird.colision(state):
                ge[i].fitness -= 1
                birds.pop(i)
                ge.pop(i)
                nets.pop(i)
        print(len(birds))
        if len(birds) == 0:
            done = True

        # 游戏画面更新
        screen.fill((0, 0, 0))
        screen.blit(IMAGES['background'], (0, 0))
        for pipe in pipes:
            screen.blit(IMAGES['pipe'][0], (pipe.x, pipe.top - PIPE_HEIGHT))
            screen.blit(IMAGES['pipe'][1], (pipe.x, pipe.bottom))

        for bird in birds:
            screen.blit(IMAGES['player'][bird.index],
                        (bird.x, bird.y))

        showScore(score)

        pygame.display.update()
        clock.tick(FPS)

# 配置NEAT
def run_neat(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    population = neat.Population(config)
    population.run(run_game, 50)  # Run for 50 generations

if __name__ == "__main__":
    config_path = os.path.join(os.getcwd(), 'config-feedforward')  # Create your own NEAT config file
    run_neat(config_path)
