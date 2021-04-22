import gym
import numpy as np

import random
import sys

from Agent import DQNAgent


class SlidePuzzle(gym.Env):
    def __init__(self, gs, training=False):
        """
        Init the game.
        
        :param gs: The grid size
        """

        # PyGame part
        self.gs = gs
        self.tiles_len = gs[0] * gs[1] - 1
        self.tiles = [(x, y) for y in range(gs[1]) for x in range(gs[0])]
        self.winCdt = [(x, y) for y in range(gs[1]) for x in range(gs[0])]

        self.prev = None

        self.nb_move = 0

        self.training = training

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
        self.observation_space = gym.spaces.Box(low=0, high=8, shape=(9,), dtype=np.int32)

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
        win = False
        if self.tiles == self.winCdt:
            win = True
        return win

    def switch(self, tile):
        """
        Switch the current tile with the blank.

        :param tile: The current tile.

        :return:     Break the switch function if a tile is sliding.
        """
        # Since we can keep moving tiles while others are sliding, we should stop that from happening.
        # We attempt this using the sliding function.
        if not self.training:
            return
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
        return (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)

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

    def shuffle(self):
        """
        Shuffle tiles.
        """
        while not self.isSolvable():
            random.shuffle(self.tiles)

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
        return True if (count % 2 == 0 and count != 0) else False

    def playAIGame(self):
        """
        Play the game with AI.

        """

        agent = DQNAgent(self)
        agent.train()

    def exit(self):
        """
        Exit the application.
        """
        sys.exit()

    """
    gym environment functions
    """

    def step(self, action):
        # action is the coordinates of the tiles we want to move
        moved_tile = self.adjacent()[action]
        if self.in_grid(moved_tile) and moved_tile != self.prev:
            self.switch(moved_tile)
            reward = 10 if self.isWin() else 0
        else:
            reward = -50  # illegal move is punished
        obs = [self.tiles.index((j, i)) + 1 for i in range(3) for j in range(3)]
        done = self.isWin()
        return obs, reward, done, {"moves": self.nb_move}

    def reset(self, episode):
        """
        gym environment reset
        """
        if episode < 30:
            self.tiles = self.winCdt[:]
            self.random()
        elif episode < 100:
            self.tiles = self.winCdt[:]
            self.random()
            self.random()
        elif episode < 200:
            self.tiles = self.winCdt[:]
            self.random()
            self.random()
            self.random()
        else:
            self.shuffle()  # shuffle the board state
        self.nb_move = 0  # reset number of moves
        self.prev = None
        return [self.tiles.index((j, i)) + 1 for i in range(3) for j in range(3)]  # return new board state

    def render(self, mode='human'):
        """
        Gym renderer is a placeholder since rendering is handled by the pygame engine
        """

        pass


def main():
    """
    The main function to run the game.
    """
    program = SlidePuzzle((3, 3), training=True)  # program is also the gym environment

    program.playAIGame()

    del program

if __name__ == '__main__':
    main()
