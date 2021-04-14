import random
from collections import deque

import numpy as np
import tensorflow as tf
from numpy.random._generator import default_rng
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import gym


class QNNAgent:
    """
    Class that represents the Q-Learning agent with neural network.

    """

    def __init__(self, env: gym.Env, discount_rate=0.9, learning_rate=0.1):
        self.action_space_size = env.action_space.n
        self.observation_space_size = env.observation_space.shape[0]
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.create_model()

    def create_model(self):
        tf.compat.v1.enable_eager_execution()
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.Input(shape=(self.observation_space_size,)))
        self.model.add(tf.keras.layers.Dense(24, activation="relu"))
        self.model.add(tf.keras.layers.Dense(24, activation="relu"))
        self.model.add(tf.keras.layers.Dense(self.action_space_size, activation="linear"))  # one output for each tile
        self.model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))


    def get_action(self, env):
        return random.randint(0, 8)

    def train(self, env: gym.Env, episodes=10000, filename="weights.h5f"):
        model_target = tf.keras.models.clone_model(self.model)
        rng = default_rng()
        cnt_l = []
        memory = deque(maxlen=3000)
        eps = 1

        for i_episode in range(episodes):
            # print("nepi: ", i_episode)
            state = env.reset()
            state = np.reshape(state, [1, self.observation_space_size])
            # state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            cnt = 0  # how may movements cart has made

            while True:
                print(cnt)
                # env.render()  # if running RL comment this out
                cnt += 1
                eps -= 0.9 / 5000
                eps = max(eps, 0.1)
                # Get action from Q NN
                # print("st: ", state_tensor)
                action_probs = self.model(state, training=False)

                # print("e:", eps, " cnt: ", cnt, " ga: ", DISCOUNT)
                if rng.random() > eps:
                    action = tf.argmax(action_probs[0]).numpy()
                else:
                    action = rng.integers(0, env.action_space.n)

                # perform action on enviroment
                print(action)
                print(state)
                newState, reward, done, info = env.step(action)
                print(newState)
                newState = np.reshape(newState, [1, self.observation_space_size])
                # newState = tf.expand_dims(tf.convert_to_tensor(newState), 0)
                memory.append((state, action, reward, newState, done))

                histo = []
                # if cnt % 5 == 0 and len(memory) > 32:
                if len(memory) > 32:
                    # batch = memory[-16:]
                    batch = random.sample(memory, 32)
                    states = []
                    targets = []
                    for stat, actio, rewar, next_stat, don in batch:
                        # print(state, action, reward, next_state, done)
                        if don:
                            target = -1
                            # print("shoulnd happen!", stat, actio, rewar, don)
                        else:
                            target = rewar + self.discount_rate * np.amax(
                                model_target.predict(next_stat)[0]
                            )
                        # target = R(s,a) + gamma * max Q`(s`,a`)
                        # target (max Q` value) is output of Neural Network which takes s` as an input
                        # amax(): flatten the lists (make them 1 list) and take max value

                        train_target = self.model.predict(stat)
                        # s --> NN --> Q(s,a)=train_target
                        train_target[0][actio] = target
                        states.append(stat[0])
                        targets.append(train_target[0])

                    hist = self.model.fit(
                        np.array(states), np.array(targets), epochs=1, verbose=0
                    )
                    histo.append(hist.history["loss"])
                    # verbose: dont show loss and epoch
                state = newState

                # print("rew: ", reward, done)

                if done and i_episode % 10 == 0:
                    self.model.save(f"saved/cartpoletest-{i_episode}", save_format="tf")

                if done and i_episode % 5 == 0:
                    model_target.set_weights(self.model.get_weights())

                if done:
                    print(
                        f"{i_episode} - done: resolved after: {cnt}, {eps} - loss: {np.mean(histo)}"
                    )
                    cnt_l.append(cnt)
                    break

            # # print(cnt_l)
            # if len(cnt_l) > 100:
            #     cnt_l.pop(0)

            # if sum(cnt_l) > 100 * 185:
            #     print("Learned, lets test")
            #     state = env.reset()
            #     while True:
            #         env.render()  # if running RL comment this out
            #         discreteState = get_discrete_state(state[features], bins, len(bins))
            #         action = np.argmax(qTable[discreteState])
            #         state, reward, done, info = env.step(action)

    def restore_model(self, filename: str):
        self.model.load_weights(filename)

    def test(self, env: gym.Env):
        self.dqn.test(env, visualize=False)
