import os
import random
from collections import deque
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from numpy.random import default_rng
import gym
import glob
import re

class DQNAgent:
    """
    Class that represents the Q-Learning agent with neural network.

    """

    def __init__(self, env: gym.Env, discount_rate=0.9, learning_rate=0.001, max_memory=3000):
        # Store environment
        self.env = env
        # Store shapes for convenience
        self.action_space_size = env.action_space.n
        self.observation_space_size = env.observation_space.shape[0]
        # This memory allows to randomly sample from previous tries
        # and avoid being biased by using only latest samples
        self.memory = deque(maxlen=max_memory)
        # We will minimize the MSE in the model,
        # which means minimizing
        # (r + gamma * argmax_a'(Q_hat(s, a') - Q(s,a))²
        # Discount rate (gamma), future reward depreciation factor
        self.discount_rate = discount_rate
        # Learning rate (r)
        self.learning_rate = learning_rate
        # Epsilon-greedy variable and its bounds
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # This model is used to predict actions to take
        self.model = self.create_model()

        list_of_files = glob.glob('./*.model')
        try:
            latest_file = max(list_of_files, key=os.path.getctime)
            self.load_model(latest_file)
            self.file_number = int(re.findall(r'\d+', latest_file)[0])
        except ValueError:
            self.file_number=0
        # This target model is used to control what actions the model should take
        self.model_target = self.create_model()
        # This double-model mode of function is required to improve convergence
        # as we are training while exploring and testing

    def create_model(self) -> tf.keras.models.Sequential:
        """
        Creates a NN model for this object environment.

        :return: the created model.
        """
        tf.compat.v1.enable_eager_execution()  # to allow numpy() function
        model = tf.keras.models.Sequential()
        # model.add(tf.keras.Input(shape=(self.observation_space_size,)))
        # Input layer with input size of observation_space_size and output size of 24
        model.add(tf.keras.layers.Dense(24, input_dim=self.observation_space_size, activation="relu"))
        # Hidden layers
        #model.add(tf.keras.layers.Dense(24, activation="relu"))
        model.add(tf.keras.layers.Dense(24, activation="relu"))
        # Output layer that has action_space_size outputs
        model.add(tf.keras.layers.Dense(self.action_space_size, activation="linear"))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, new_state, done: bool) -> None:
        """
        Adds a state - action context to memory.


        :param state: Initial state, before acting
        :param action: Action taken
        :param reward: Reward earned
        :param new_state: New state reached
        :param done: If game is finished or not
        """
        self.memory.append((state, action, reward, new_state, done))

    def replay(self) -> None:
        """
        Trains the model following target model q-values.

        """
        # batch_size represents the number of samples we take from memory
        batch_size = 32
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.model_target.predict(state)
            if done:
                # reward is not discounted as it is the actual reward of the state
                target[0][action] = reward
            else:
                # the reward is the current reward + what we can get later (discounted)
                q_future = max(self.model_target.predict(new_state)[0])
                target[0][action] = reward + q_future * self.discount_rate
            # finally train the model (NOT model_target) to fit that input-output combination
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self) -> None:
        """
        Turns the target model into the trained model.
        This update is called less frequently to improve stability.
        """
        weights = self.model.get_weights()
        target_weights = weights[:]  # copy values
        self.model_target.set_weights(target_weights)

    def get_action(self, state) -> int:
        # slowly decrease epsilon
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            # take a random action
            return self.env.action_space.sample()
        # else, return what the model predicts
        return int(np.argmax(self.model.predict(state)[0]))

    def train(self, episodes=10000) -> None:
        """
        Trains the model on episodes
        :param episodes: number of games the agent will play
        """
        # eye candy progress bar
        progress_bar = tqdm(range(self.file_number+1, episodes))
        for episode in progress_bar:
            # reset for each episode
            state = self.env.reset(episode)
            # reshape to feed the NN
            state = np.reshape(state, [1, self.observation_space_size])
            # run while game is not solved
            done = False
            moves = 0
            while not done and moves < 50:
                # decide what action to take
                moves += 1
                if moves % 10 == 0:
                    print(f"moves: {moves}")
                action = self.get_action(state)
                # act
                """print(f"moves: {moves}")
                print(f"old state: {state}")
                print(f"action: {action}")"""
                new_state, reward, done, _ = self.env.step(action)

                """print(f"new state: {new_state}")
                print(f"reward: {reward}")"""
                new_state = np.reshape(new_state, [1, self.observation_space_size])
                # remember consequences of our acts
                self.remember(state, action, reward, new_state, done)
                # train the model
                self.replay()
                # update the target
                self.target_train()
                # go into new state
                state = new_state
            # update progress bar
            progress_bar.set_description(f"Episode {episode} - Succeeded after {moves} moves")
            #  print(f"Episode {episode} completed in {moves} moves.")
            self.save_model(f"episode_{episode}.model")

    def save_model(self, filename: str):
        self.model.save(filename)

    def load_model(self, filename):
        print(f"loading models from: {filename}")
        self.model = tf.keras.models.load_model(filename)
        self.model_target = tf.keras.models.load_model(filename)
