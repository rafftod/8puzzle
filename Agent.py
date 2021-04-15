import random
from collections import deque
import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import Sequential
from numpy.random import default_rng
import gym


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
        model = tf.keras.model.Sequential()
        # model.add(tf.keras.Input(shape=(self.observation_space_size,)))
        # Input layer with input size of observation_space_size and output size of 24
        model.add(tf.keras.layers.Dense(24, input_shape=self.observation_space_size, activation="relu"))
        # Hidden layers
        model.add(tf.keras.layers.Dense(48, activation="relu"))
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
        return np.argmax(self.model.predict(state)[0])[0]

    def train(self, env: gym.Env, episodes=10000):
        pass
        # # random number generator
        # rng = default_rng()
        # # this memory queue will store :
        # # - action history
        # # - state history
        # # - new state observed history
        # # - reward history
        # # - done history (that indicates if the game ended)
        # memory = deque(maxlen=100000)
        # # Number of moves while we always take a random action
        # epsilon_random_moves = 500
        # # Number of moves for exploration
        # epsilon_greedy_frames = 10000
        # # Action count before starting training
        # actions_before_update = 100
        # # Network update rate
        # network_update_rate = 10000
        # # epsilon-greedy
        # epsilon = 1.0
        #
        # for i_episode in range(episodes):
        #     state = np.array(env.reset())
        #     episode_reward = 0
        #     state = np.reshape(state, [1, self.observation_space_size])
        #     moves_count = 0  # number of moves done in this episode
        #
        #     while True:
        #         moves_count += 1
        #
        #         if moves_count < epsilon_random_moves or epsilon > np.random.rand(1)[0]:
        #             # take a random action and see what happens
        #             action = np.random.randint(self.action_space_size)
        #         else:
        #             # predict Q-value from current state
        #             state_tensor = tf.convert_to_tensor(state)
        #             state_tensor = tf.expand_dims(state_tensor, 0)
        #             action_probs = self.model(state_tensor, training=False)
        #             action = tf.argmax(action_probs[0]).numpy()
        #
        #         # TODO
        #         # print("e:", eps, " cnt: ", cnt, " ga: ", DISCOUNT)
        #         if rng.random() > eps:
        #             action = tf.argmax(action_probs[0]).numpy()
        #         else:
        #             action = rng.integers(0, env.action_space.n)
        #
        #         # perform action on enviroment
        #         print(action)
        #         print(state)
        #         newState, reward, done, info = env.step(action)
        #         print(newState)
        #         newState = np.reshape(newState, [1, self.observation_space_size])
        #         # newState = tf.expand_dims(tf.convert_to_tensor(newState), 0)
        #         memory.append((state, action, reward, newState, done))
        #
        #         histo = []
        #         # if cnt % 5 == 0 and len(memory) > 32:
        #         if len(memory) > 32:
        #             # batch = memory[-16:]
        #             batch = random.sample(memory, 32)
        #             states = []
        #             targets = []
        #             for stat, actio, rewar, next_stat, don in batch:
        #                 # print(state, action, reward, next_state, done)
        #                 if don:
        #                     target = -1
        #                     # print("shoulnd happen!", stat, actio, rewar, don)
        #                 else:
        #                     target = rewar + self.discount_rate * np.amax(
        #                         model_target.predict(next_stat)[0]
        #                     )
        #                 # target = R(s,a) + gamma * max Q`(s`,a`)
        #                 # target (max Q` value) is output of Neural Network which takes s` as an input
        #                 # amax(): flatten the lists (make them 1 list) and take max value
        #
        #                 train_target = self.model.predict(stat)
        #                 # s --> NN --> Q(s,a)=train_target
        #                 train_target[0][actio] = target
        #                 states.append(stat[0])
        #                 targets.append(train_target[0])
        #
        #             hist = self.model.fit(
        #                 np.array(states), np.array(targets), epochs=1, verbose=0
        #             )
        #             histo.append(hist.history["loss"])
        #             # verbose: dont show loss and epoch
        #         state = newState
        #
        #         # print("rew: ", reward, done)
        #
        #         if done and i_episode % 10 == 0:
        #             self.model.save(f"saved/cartpoletest-{i_episode}", save_format="tf")
        #
        #         if done and i_episode % 5 == 0:
        #             model_target.set_weights(self.model.get_weights())
        #
        #         if done:
        #             print(
        #                 f"{i_episode} - done: resolved after: {cnt}, {eps} - loss: {np.mean(histo)}"
        #             )
        #             cnt_l.append(cnt)
        #             break
        #
        #     # # print(cnt_l)
        #     # if len(cnt_l) > 100:
        #     #     cnt_l.pop(0)
        #
        #     # if sum(cnt_l) > 100 * 185:
        #     #     print("Learned, lets test")
        #     #     state = env.reset()
        #     #     while True:
        #     #         env.render()  # if running RL comment this out
        #     #         discreteState = get_discrete_state(state[features], bins, len(bins))
        #     #         action = np.argmax(qTable[discreteState])
        #     #         state, reward, done, info = env.step(action)

    def restore_model(self, filename: str):
        self.model.load_weights(filename)

    def test(self, env: gym.Env):
        pass
