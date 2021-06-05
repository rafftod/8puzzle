"""
Most of the code pertaining to the graphical interface comes from Brice PETIT,
Dumitru NEGRU, and Maxence HUWART
"""

import os
import random
import sys
import time

import gym
import numpy as np
import pygame

from Agent import DQNAgent

FPS = 60


class SlidePuzzle(gym.Env):
    def __init__(self, gs, ts, ms, difficulty = 6):
        """
        Init the game.
        
        :param gs: The grid size
        """

        # PyGame part
        self.gs, self.ts, self.ms = gs, ts, ms
        self.tiles_len = gs[0] * gs[1] - 1
        self.tiles = [(x, y) for y in range(gs[1]) for x in range(gs[0])]
        self.winCdt = [(x, y) for y in range(gs[1]) for x in range(gs[0])]

        # actual pos on the screen
        self.tilepos = [(x * (ts + ms) + ms, y * (ts + ms) + ms) for y in range(gs[1]) for x in range(gs[0])]

        # the place they slide to
        self.tilePOS = {(x, y): (x * (ts + ms) + ms, y * (ts + ms) + ms) for y in range(gs[1]) for x in range(gs[0])}

        self.rect = pygame.Rect(0, 0, gs[0] * (ts + ms) + ms, gs[1] * (ts + ms) + ms)
        self.speed = 3

        self.prev = None
        self.nb_move = 0
        self.nb_games = 0

        self.pic = pygame.transform.smoothscale(pygame.image.load('image.png'), self.rect.size)

        self.images = []
        font = pygame.font.Font(None, 120)
        for i in range(self.tiles_len):
            x, y = self.tilepos[i]
            image = self.pic.subsurface(x, y, ts, ts)
            text = font.render(str(i + 1), 2, (0, 0, 0))
            w, h = text.get_size()
            image.blit(text, ((ts - w) / 2, (ts - h) / 2))
            self.images += [image]

        self.difficulties = [(2, 500, 3), (4, 1500, 6), (6, 4000, 9), (8, 10000, 12),
                             (10, 20000, 15)]  # list of (nb_shuffles, episode_cap, nb_tries)
        self.testing_difficulty = difficulty
        self.max_tries = round(self.testing_difficulty*1.5)
        # gym part

        # Reward is :
        # + 10 if state is a solution to the problem
        # 0 otherwise
        # -50 if tile could not be moved
        self.reward_range = (-50, 10)

        # Action space:
        # We can move one piece of the board from its coordinates
        # Each coordinate is encoded as integer from 0 to 3
        # 0 is left, 1 is right, 2 is up, 3 is down
        # see step() for decoding
        self.action_space = gym.spaces.Discrete(4)

        # Observation space:
        # We encode the board as an array of 9 values, ordered as on the board
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(72,), dtype=np.int32)

        # Create agent
        self.agent = DQNAgent(self)

    def getBlank(self):
        """
        Get the blank tile.

        :return: Return the last tile.
        """
        return self.tiles[-1]

    def setBlank(self, pos):
        """
        Set the blank tile.

        :param pos: The position of the blank tile.
        """
        self.tiles[-1] = pos

    # The blank tile.
    opentile = property(getBlank, setBlank)

    def isWin(self):
        """
        Check if the player win the game.

        :return: Return True if the player win the game, otherwise False.
        """

        return self.tiles == self.winCdt

    def isLost(self):
        return self.nb_move >= self.max_tries

    def sliding(self):
        """
        Check if there are tiles that are sliding.
        :return: Return True if a tile is sliding, otherwise nothing.
        """
        for i in range(self.tiles_len):
            x, y = self.tilepos[i]  # current pos
            X, Y = self.tilePOS[self.tiles[i]]  # target pos
            if x != X or y != Y:
                return True

    def switch(self, tile):
        """
        Switch the current tile with the blank.

        :param tile: The current tile.

        :return:     Break the switch function if a tile is sliding.
        """
        self.tiles[self.tiles.index(tile)], self.opentile, self.prev = self.opentile, tile, self.opentile
        self.nb_move += 1

    def in_grid(self, tile):
        """
        Check if the tile is in the grid.

        :param tile: The tile to check.

        :return:     Return true if the tile is in the grid, otherwise false.
        """
        return 0 <= tile[0] < self.gs[0] and 0 <= tile[1] < self.gs[1]

    def adjacent(self):
        """
        Give the positions of the tiles adjacent to the blank tile.

        :return: Return positions of the tiles adjacent to the blank tile.
        """
        x, y = self.opentile
        return (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)  # left, right, up, down

    def random(self):
        """
        Choose randomly an action.
        """
        adj = self.adjacent()
        self.switch(random.choice([pos for pos in adj if self.in_grid(pos) and pos != self.prev]))

    def random_adjacent_tile(self):
        """
        Chooses a tile at random from blank adjacency
        """
        adj = self.adjacent()
        pos_list = [pos for pos in adj if self.in_grid(pos) and pos != self.prev]
        return random.choice(pos_list)

    def shuffle(self, difficulty=None):
        """
        Shuffle tiles.
        """
        if difficulty is None:
            while not self.isSolvable():
                random.shuffle(self.tiles)
        else:
            while self.manhattan_distance() < difficulty:
                self.random()

    def isSolvable(self):
        """
        Check if the game is solvable.

        :return: Return True if yes, otherwise False.
        """
        tiles = []
        for i in range(len(self.tiles)):
            for j in range(len(self.tiles)):
                if self.tiles[j][1] * 3 + self.tiles[j][0] + 1 == i + 1:
                    tiles.append(j + 1)
        count = 0
        for i in range(len(tiles) - 1):
            for j in range(i + 1, len(tiles)):
                if tiles[i] > tiles[j] and tiles[i] != 9:
                    count += 1
        return count % 2 == 0 and count != 0

    def update(self, dt):
        """
        Update the view.
        :param dt: Derived time.
        """
        # If the value between the current and target is less than speed, we can just let it jump right into place.
        # Otherwise, we just need to add/sub in direction.
        s = self.speed * dt
        for i in range(self.tiles_len):
            x, y = self.tilepos[i]  # current pos
            X, Y = self.tilePOS[self.tiles[i]]  # target pos
            dx, dy = X - x, Y - y

            self.tilepos[i] = (X if abs(dx) < s else x + s if dx > 0 else x - s), (
                Y if abs(dy) < s else y + s if dy > 0 else y - s)

    def draw(self, screen):
        """
        Draw the game with the number of move.
        :param screen: The current screen.
        """
        for i in range(self.tiles_len):
            x, y = self.tilepos[i]
            screen.blit(self.images[i], (x, y))
        self.draw_text(screen, "Moves : " + str(self.nb_move), 40, 500, 10, 255, 255, 255, False)

    def draw_text(self, screen, text, size, x, y, R, G, B, center):
        """
        Draw text.
        :param screen: The screen.
        :param text:   The text to draw on the screen.
        :param size:   The size of the text.
        :param x:      The x position of the text.
        :param y:      The y position of the text.
        :param R:      The R color.
        :param G:      The G color.
        :param B:      The B color.
        :param center: If the text need to be in the center.
        """
        font = pygame.font.Font(None, size)
        text = font.render(text, True, (R, G, B))
        if center:
            text_rect = text.get_rect()
            text_rect.midtop = (x, y)
            screen.blit(text, text_rect)
        else:
            screen.blit(text, (x, y))

    def drawShortcuts(self, screen, is_player):
        """
        Draw in game shortcuts
        :param screen:    The screen.
        :param is_player: Check if it is a player because, shorcuts are different in
                          the player mode or in the AI mode.
        """
        self.draw_text(screen, "Shortcuts", 40, 500, 40, 255, 255, 255, False)
        self.draw_text(screen, "Pause : Escape", 40, 500, 70, 255, 255, 255, False)
        if is_player:
            self.draw_text(screen, "Move up : z", 40, 500, 100, 255, 255, 255, False)
            self.draw_text(screen, "Move down : s", 40, 500, 130, 255, 255, 255, False)
            self.draw_text(screen, "Move left : q", 40, 500, 160, 255, 255, 255, False)
            self.draw_text(screen, "Move right : d", 40, 500, 190, 255, 255, 255, False)
            self.draw_text(screen, "Random move : Space", 40, 500, 220, 255, 255, 255, False)
            self.draw_text(screen, "AI move : a", 40, 500, 250, 255, 255, 255, False)

    def playEvents(self, event):
        """
        Catch events from the mouse and the keyboard.
        Binded keys:
            - z moves the tile upwards
            - s moves the tile downwards
            - q moves the tile to the left
            - d moves the tile to the right
            - space choose a random mouvement
        :param event: The current event.
        """
        mouse = pygame.mouse.get_pressed()
        mpos = pygame.mouse.get_pos()
        # If we use the left click
        if mouse[0]:
            # We convert the position of the mouse according to the grid position and the margin
            x, y = mpos[0] % (self.ts + self.ms), mpos[1] % (self.ts + self.ms)
            if x > self.ms and y > self.ms:
                tile = mpos[0] // self.ts, mpos[1] // self.ts
                if self.in_grid(tile) and tile in self.adjacent():
                    self.switch(tile)

        if event.type == pygame.KEYDOWN:
            for key, dx, dy in ((pygame.K_s, 0, -1), (pygame.K_z, 0, 1), (pygame.K_d, -1, 0), (pygame.K_q, 1, 0)):
                if event.key == key:
                    x, y = self.opentile
                    tile = x + dx, y + dy
                    if self.in_grid(tile):
                        self.switch(tile)
            # Move randomly a tile.
            if event.key == pygame.K_SPACE:
                self.random()
            if event.key == pygame.K_a:
                action = self.agent.play(self.format_tiles())
                self.step(action)

    def catchGameEvents(self, is_player, fpsclock, screen):
        """
        Catchs event during the game.
        :param is_player: A boolean value to check if it is a game with a player or with an AI.
        :param fpsclock:  Track time.
        :param screen:    The screen.

        :return:          Return True if the player want to quit the game.
                          Otherwise, False.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit()
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return self.pauseMenu(fpsclock, screen)
            if is_player:
                self.playEvents(event)
        return False

    def playGame(self, fpsclock, screen):
        """
        Play the game.
        :param fpsclock: Track time.
        :param screen:   The screen.
        """
        wantToQuitGame = False
        finished = False
        self.reset()
        while not finished and not wantToQuitGame:
            dt = fpsclock.tick(FPS)
            screen.fill((0, 0, 0))
            self.draw(screen)
            self.drawShortcuts(screen, True)
            pygame.display.flip()
            wantToQuitGame = self.catchGameEvents(True, fpsclock, screen)
            self.update(dt)
            finished = self.checkGameState(fpsclock, screen)


    def checkGameState(self, fpsclock, screen):
        """
        Check if the game is won. If it is won, we ask to the player if he want
        the play again, quit the game or want to go to the main menu.

        :param fpsclock: Track time.
        :param screen:   The screen.
        :return:         Return False if the game is won or if the player want
                         to play again. Otherwise, False.
        """
        if self.isWin() or self.isLost():
            if self.exitMenu(fpsclock, screen):
                return True
        return False

    def selectPlayerMenu(self, fpsclock, screen):
        """
        Ask to the player if he wants to play or if he wants an AI to play.
        :param fpsclock: Track time.
        :param screen:   The screen.
        :return:         Return the choice of the player.
        """
        screen.fill((0, 0, 0))
        self.draw_text(screen, "Press h to play", 40, 400, 150, 255, 255, 255, True)
        self.draw_text(screen, "Press a to train the AI", 40, 400, 300, 255, 255, 255, True)
        pygame.display.flip()
        while True:
            dt = fpsclock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h:
                        self.reset()
                        return "human"
                    if event.key == pygame.K_a:
                        return "AI"

    def pauseMenu(self, fpsclock, screen):
        """
        Ask to the player if he want to continue the game or if he want to go to the main menu.
        :param fpsclock: Track time.
        :param screen:   The screen.
        :return:         Return True if the player want to go to the main menu.
                         Otherwise, False.
        """
        screen.fill((0, 0, 0))
        self.draw_text(screen, "Do you want to go back", 40, 400, 100, 255, 255, 255, True)
        self.draw_text(screen, "to the main menu ?", 40, 400, 140, 255, 255, 255, True)
        self.draw_text(screen, "Press y for yes", 40, 400, 250, 255, 255, 255, True)
        self.draw_text(screen, "Press n for no (or escape)", 40, 400, 290, 255, 255, 255, True)
        pygame.display.flip()
        while True:
            dt = fpsclock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        return True
                    if event.key == pygame.K_n or event.key == pygame.K_ESCAPE:
                        return False

    def exitMenu(self, fpsclock, screen):
        """
        The menu to exit, restart the game or go to the main menu.
        :param fpsclock: Track time.
        :param screen:   The screen.
        :return:         Return True if the player wants to go to the main menu.
                         Otherwise, False.
        """
        screen.fill((0, 0, 0))
        self.rect = pygame.Rect(0, 0, self.gs[0] * (self.ts + self.ms) + self.ms,
                                self.gs[1] * (self.ts + self.ms) + self.ms)
        self.pic = pygame.transform.smoothscale(pygame.image.load('bluredImage.png'), self.rect.size)
        screen.blit(self.pic, self.rect)
        if self.isWin():
            self.draw_text(screen, "You won !", 50, 250, 80, 0, 0, 0, True)
            self.draw_text(screen, "Congratulations !", 50, 250, 160, 0, 0, 0, True)
        else:
            self.draw_text(screen, "You lost !", 50, 250, 80, 0, 0, 0, True)
            self.draw_text(screen, "Better luck next time !", 50, 250, 160, 0, 0, 0, True)
        self.draw_text(screen, "Moves : " + str(self.nb_move), 40, 500, 10, 255, 255, 255, False)
        self.draw_text(screen, "Shortcuts", 40, 500, 40, 255, 255, 255, False)
        self.draw_text(screen, "Restart : y", 40, 500, 70, 255, 255, 255, False)
        self.draw_text(screen, "Quit : n", 40, 500, 100, 255, 255, 255, False)

        pygame.display.flip()
        while True:
            fpsclock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        self.reset()
                        return False
                    if event.key == pygame.K_n:
                        self.exit()

    def exit(self):
        """
        Exit the application.
        """
        sys.exit()

    """
    gym environment functions
    """

    def format_tiles(self):
        formatted_tiles = [0] * 72
        for i in range(3):
            for j in range(3):
                pos = self.tiles.index((j, i))
                if pos != 8:
                    formatted_tiles[pos * 9 + i * 3 + j] = 1

        return formatted_tiles

    def step(self, action):
        # action is the coordinates of the tiles we want to move
        # pos_dict = {0:"Left", 1:"Right", 2:"Up", 3:"Down"}
        # print(f"Blank pos: {self.tiles[8]}, Action: {pos_dict[action]} |", end = " ")
        moved_tile = self.adjacent()[action]
        if self.in_grid(moved_tile) and moved_tile != self.prev:
            self.switch(moved_tile)
            reward = 10 if self.isWin() else -self.manhattan_distance()
        else:
            reward = -50  # illegal move is punished
        obs = self.format_tiles()
        done = self.isWin()
        return obs, reward, done, {"moves": self.nb_move}

    def reset(self):
        """
        gym environment reset
        """
        self.tiles = self.winCdt[:]
        """for difficulty, episode_cap, _ in self.difficulties:
            if self.nb_games < episode_cap:
                self.shuffle(difficulty)
                break
        else:
            self.shuffle()  # shuffle the board state"""
        self.shuffle(difficulty=self.testing_difficulty)

        self.nb_move = 0  # reset number of moves
        self.prev = None
        self.nb_games += 1
        return self.format_tiles()  # return new board state

    def render(self, mode='human'):
        """
        Gym renderer is a placeholder since rendering is handled by the pygame engine
        """

        pass

    def manhattan_distance(self):
        dist = 0
        for target, tile in zip(self.winCdt[:-1], self.tiles[:-1]):
            dist += abs(target[0] - tile[0]) + abs(target[1] - tile[1])
        return dist

def launchWithGUI(program, fpsclock, screen):
    program.playGame(fpsclock, screen)

def trainAI(program):
    program.nb_games = program.agent.episode_number
    program.agent.start()

def main():
    """
    The main function to run the game.
    """
    pygame.init()
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.display.set_caption('8-Puzzle game')
    screen = pygame.display.set_mode((800, 500))
    fpsclock = pygame.time.Clock()
    program = SlidePuzzle((3, 3), 160, 5, difficulty=6)  # program is also the gym environment
    choice = program.selectPlayerMenu(fpsclock, screen)
    if choice == "AI":
        pygame.display.quit()
        trainAI(program)
    elif choice == "human":
        launchWithGUI(program, fpsclock, screen)
    del program




if __name__ == '__main__':
    main()
