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
        self.nb_games = 0

        self.training = training
        self.difficulties = [(8, 9300, 14), (10, 20000, 18)] # list of (nb_shuffles, episode_cap, nb_tries)
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
        return True if (count % 2 == 0 and count != 0) else False

    def playAIGame(self):
        """
        Play the game with AI.

        """

        agent = DQNAgent(self)
        self.nb_games = agent.episode_number
        #agent.start()
        for i in range (8, 11):
            agent.do_predictions(500, i)

    def exit(self):
        """
        Exit the application.
        """
        sys.exit()

    """
    gym environment functions
    """

    def format_tiles(self):
        formatted_tiles = [0]*72
        for i in range(3):
            for j in range(3):
                pos = self.tiles.index((j, i))
                if pos != 8:
                    formatted_tiles[pos*9 + i*3 + j] = 1

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

    def step_predict(self, action):
        # action is the coordinates of the tiles we want to move
        # pos_dict = {0:"Left", 1:"Right", 2:"Up", 3:"Down"}
        # print(f"Blank pos: {self.tiles[8]}, Action: {pos_dict[action]} |", end = " ")
        moved_tile = self.adjacent()[action]
        if self.in_grid(moved_tile) and moved_tile != self.prev:
            self.switch(moved_tile)
            reward = 10 if self.isWin() else -self.manhattan_distance()
        else:
            self.random()
            reward = -50  # illegal move is punished
        obs = self.format_tiles()
        done = self.isWin()
        return obs, reward, done, {"moves": self.nb_move}

    def reset(self):
        """
        gym environment reset
        """
        self.tiles = self.winCdt[:]
        for difficulty, episode_cap, _ in self.difficulties:
            if self.nb_games < episode_cap:
                self.shuffle(difficulty)
                break
        else:
            self.shuffle()  # shuffle the board state

        self.nb_move = 0  # reset number of moves
        self.prev = None
        self.nb_games += 1
        return self.format_tiles() # return new board state


    def reset_predict(self, difficulty):
        """
        gym environment reset
        """
        self.tiles = self.winCdt[:]
        self.shuffle(difficulty)

        self.nb_move = 0  # reset number of moves
        self.prev = None
        return self.format_tiles() # return new board state

    def render(self, mode='human'):
        """
        Gym renderer is a placeholder since rendering is handled by the pygame engine
        """

        pass

    def manhattan_distance(self):
        dist = 0
        for target, tile in zip(self.winCdt[:-1], self.tiles[:-1]):
            dist += abs(target[0]-tile[0]) + abs(target[1]-tile[1])
        return dist

def main():
    """
    The main function to run the game.
    """
    program = SlidePuzzle((3, 3), training=True)  # program is also the gym environment

    program.playAIGame()

    del program

if __name__ == '__main__':
    main()
