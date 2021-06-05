# 8 puzzle
This project implements deep Q-learning for the 8-puzzle problem.

## Running the code
You can start the game and choose either a human or AI played game with the following command :
```
python3 EightPuzzleGame.py
```

## Architecture

We will describe the most important parts and functions of the code here under.

### Agent.py

This file is the implementation of the Q-learning agent.

#### train()
This method is used to train the deep neural network that estimates the Q-function. It makes use of a replay memory to enable experience replay and break the correlation between consecutive samples.

#### play()
This is used to get the prediction of the agent model.


#### start()

Starts the training over 20000 episodes (if no models are saved).

### EightPuzzleGame.py

This implements the UI, the behaviour of the game engine and the gym environment it is built onto.

The gym action and state spaces are defined in the constructor of the EightPuzzleGame class.

We will briefly summarize what is particular to this project, which is the gym functions.

#### step()

This performs an action (moving a tile) on the environment, and computes the reward of the agent. It also indicates if the game is won or not.

#### reset()

This is used to reset the environment by shuffling the tiles and setting the number of moves to 0.
