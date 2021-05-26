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
MIN_REPLAY_MEMORY_SIZE = 32  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)


class DQNAgent:
    """
    Class that represents the Q-Learning agent with neural network.

    """
    def __init__(self, env: gym.Env, discount_rate=0.9, learning_rate=0.001, max_memory=1500):
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
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # This model is used to predict actions to take
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0

        list_of_files = glob.glob('./*.model')
        try:
            latest_file = max(list_of_files, key=os.path.getctime)
            self.load_model(latest_file)
            self.episode_number = int(re.findall(r'\d+', latest_file)[0])
        except ValueError:
            self.episode_number=0
        self.epsilon = max(self.epsilon_decay**self.episode_number, self.epsilon_min)
        # This target model is used to control what actions the model should take
        # Target network
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
        #model.add(tf.keras.layers.Dense(250, input_dim=self.observation_space_size, activation="relu"))
        model.add(tf.keras.layers.Flatten(input_shape=(self.observation_space_size,)))
        # Hidden layers
        #model.add(tf.keras.layers.Dense(16, activation="relu"))
        model.add(tf.keras.layers.Dense(200, activation="relu"))
        model.add(tf.keras.layers.Dense(200, activation="relu"))

        # Output layer that has action_space_size outputs
        model.add(tf.keras.layers.Dense(self.action_space_size, activation="linear"))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        print(model.summary())
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount_rate * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > 5:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, len(state)))[0]

    def save_model(self, filename: str):
        self.model.save(filename)

    def load_model(self, filename):
        print(f"loading models from: {filename}")
        self.model = tf.keras.models.load_model(filename)
        self.target_model = tf.keras.models.load_model(filename)

    def play(self, current_state):
        """if np.random.random() > 0.1:
            # Get action from Q table
            action = np.argmax(self.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, self.action_space_size)"""
        # action = np.argmax(self.get_qs(current_state))
        action = np.random.randint(0, self.action_space_size)
        print(action)
        return action


    def start(self):
        # Iterate over episodes
        successes = 0
        progress_bar = tqdm(range(self.episode_number+1, 10000 + 1), ascii=True, unit='episodes')
        for episode in progress_bar:
            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = self.env.reset()

            # Reset flag and start iterating until episode ends
            done = False
            difficulty = 35
            for elem in self.env.difficulties:
                if episode < elem[1]:
                    difficulty = elem[2]
                    break
            rewards_list = []
            while not done and step < difficulty:
                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > self.epsilon:
                    # Get action from Q table
                    action = np.argmax(self.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, self.action_space_size)

                new_state, reward, done, _ = self.env.step(action)
                rewards_list.append(reward)
                if reward == 10:
                    successes += 1
                progress_bar.set_description(f"Success {successes}, Success rate: {round(100*successes/(episode-5000), 2)}%, Epsilon: {round(self.epsilon, 2)}", refresh=True)
                progress_bar.set_postfix_str(f"Rewards: {rewards_list}", refresh=True)

                # Every step we update replay memory and train main network
                self.update_replay_memory((current_state, action, reward, new_state, done))
                self.train(done)

                current_state = new_state
                step += 1
            if episode % 25 == 0:
                self.save_model(f"episode_{episode}.model")
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                # if episode == difficulty[1]:
                #     self.epsilon = 1
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon)
