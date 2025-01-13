import numpy as np
import sys
import random
import gym
import time
import signal
from collections import deque
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from scipy.signal import savgol_filter
import math
import matplotlib.pyplot as plt


class Networklayers():
    def __init__(self, state_list, action_list) -> None:
        self.state_list = state_list
        self.action_list = action_list

    def net_fl(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_list, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_list, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=config.optim, metrics=['mse'])
        return model

    def net(self):
        inputs = Input(shape=(self.state_list,))
        layers = Conv2D(16, 3, activation='relu')(inputs)
        layers = MaxPooling2D((2, 2))(layers)
        layers = Conv2D(32, 3, activation='relu')(layers)
        layers = MaxPooling2D((2, 2))(layers)
        layers = Dropout(0.25)(layers)
        layers = Flatten()(layers)
        layers = Dense(32, activation='relu')(layers)
        layers = Dropout(0.25)(layers)
        layers = Dense(self.action_list, activation='linear')(layers)
        model = Model(inputs=inputs, outputs=layers)
        return model

c=1
class DQNConfiguration:
    def __init__(self):
        # Policy settings
        self.policy = 'egreedy'
        self.epsilon = 0.1
        self.temp = 1
        self.annealing_epsilon = 1.0
        self.annealing_epsilon_min = 0.01
        self.annealing_epsilon_decay = 0.9995

        # Training settings
        self.lr = 0.0001
        self.batch_size = 128
        self.episodes = 2000
        self.optim = Adam(learning_rate=self.lr)
        self.training_limitation = self.batch_size

        # Reinforcement learning settings
        self.gamma = 0.95
        self.replay_capacity = 5000
        self.set_weights_frequency = 50

        # Environment settings
        self.env_name = 'CartPole-v1'

        # Plot settings
        self.avg_list = []
        self.step_list = []
        self.fig_file = self.generate_fig_filename()

    def generate_fig_filename(self):
        if self.policy == 'egreedy':
            return f'Lr={self.lr}, policy={self.policy}, epsilon={self.epsilon}, batchsize={self.batch_size}, buffer_size={self.replay_capacity}, SET_WEIGHTS_FREQENCY={self.set_weights_frequency}, GAMMA={self.gamma}.png'
        elif self.policy == 'annealing_egreedy':
            return f'Lr={self.lr}, policy={self.policy}, epsilon={self.annealing_epsilon}, epsilon_min={self.annealing_epsilon_min}, epsilon_decay={self.annealing_epsilon_decay}, batchsize={self.batch_size}, buffer_size={self.replay_capacity}, SET_WEIGHTS_FREQENCY={self.set_weights_frequency}, GAMMA={self.gamma}.png'
        elif self.policy == 'boltzmann':
            return f'Lr={self.lr}, policy={self.policy}, temp={self.temp}, batchsize={self.batch_size}, buffer_size={self.replay_capacity}, SET_WEIGHTS_FREQENCY={self.set_weights_frequency}, GAMMA={self.gamma}.png'

    @staticmethod
    def softmax(x, temp):
        x = x / temp
        z = x - np.max(x)
        return np.exp(z) / np.sum(np.exp(z))


class Agent():
    def __init__(self, state_list, action_list):
        networks = Networklayers(state_list, action_list)
        self.original_model = networks.net_fl()
        self.target_model = networks.net_fl()
        self.replay_buffer = deque(maxlen=config.replay_capacity)
        self.training_counts = 0
        self.state_list = state_list
        self.action_list = action_list
        self.policy = config.policy
        self.annealing_egreedy_epsilon = config.annealing_epsilon
        self.annealing_egreedy__epsilon_min = config.annealing_epsilon_min
        self.annealing_egreedy_epsilon_decay = config.annealing_epsilon_decay

    def egreedyact(self, shape):
        shape = np.reshape(shape, (1, self.state_list))
        if np.random.rand() >= config.epsilon:
            shape_q = self.original_model.predict(shape)
            action = np.argmax(np.squeeze(shape_q))
        else:
            action = np.random.randint(0, self.action_list)
        return action

    def boltzmannact(self, shape):
        shape = np.reshape(shape, (1, self.state_list))
        shape_q = self.original_model.predict(shape)
        probs = config.softmax(np.squeeze(shape_q), config.temp)
        action = np.random.choice(range(self.action_list), 1, p=np.squeeze(probs))[0]
        return action

    def annealingegreedyact(self, shape):
        shape = np.reshape(shape, (1, self.state_list))
        if np.random.rand() < self.annealing_egreedy_epsilon:
            action = np.random.randint(0, self.action_list)
        else:
            shape_q = self.original_model.predict(shape)
            action = np.argmax(np.squeeze(shape_q))
        return action

    def replays(self, shape, action, r, shape_next, done):
        transition = (shape, action, r, shape_next, done)
        self.replay_buffer.append(transition)

    def sampleData(self):
        return random.sample(self.replay_buffer, config.batch_size)

    def updatedata(self):
        self.target_model.set_weights(self.original_model.get_weights())

    def training(self):
        if self.training_counts % config.set_weights_frequency == 0:
            self.updatedata()
        self.training_counts += 1
        batch_data = self.sampleData()
        state_batch, action_batch, replay_batch, state_next_batch, terminal_batch = [], [], [], [], []
        for data in batch_data:
            state_batch.append(data[0])
            action_batch.append(data[1])
            replay_batch.append(data[2])
            state_next_batch.append(data[3])
            terminal_batch.append(data[4])

        state_batch = np.array(state_batch)
        state_next_batch = np.array(state_next_batch)
        Q_value_batch = self.original_model(state_batch)
        Q_value_target_batch = np.array(Q_value_batch, copy=True)
        Q_next_batch = self.target_model(state_next_batch)
        for i in range(config.batch_size):
            terminal = terminal_batch[i]
            Q_value_target = replay_batch[i] if terminal else replay_batch[i] + config.gamma * \
                                                              np.max(Q_next_batch, axis=-1)[i]
            Q_value_target = np.array(Q_value_target)
            Q_value_target_batch[i][action_batch[i]] = Q_value_target

        result = self.original_model.fit(x=state_batch, y=Q_value_target_batch, verbose=0)
        if self.policy == 'annealing_egreedy':
            if self.annealing_egreedy_epsilon > self.annealing_egreedy__epsilon_min:
                self.annealing_egreedy_epsilon = self.annealing_egreedy_epsilon * self.annealing_egreedy_epsilon_decay

        for i in range(config.batch_size):
            terminal = terminal_batch[i]
            Q_value_target = replay_batch[i] if terminal else replay_batch[i] + config.gamma * \
                                                              np.max(Q_next_batch, axis=-1)[i]*c
            Q_value_target = np.array(Q_value_target)
            A_batch_np = np.array(action_batch)
            Q_value_target_batch[i][A_batch_np[i]] = Q_value_target
        return result.history


class PerformancePlotter:
    def __init__(self, fig_file):
        self.fig_file = fig_file
        self.avg_loss_list = []
        self.step_counts_list = []

    def update_data(self, avg_loss_list, step_counts_list):
        """
        Update the plotting data.

        Parameters:
        avg_loss_list (list): A list of average losses to be plotted.
        step_counts_list (list): A list of step counts to be plotted.
        """
        self.avg_loss_list = avg_loss_list
        self.step_counts_list = step_counts_list

    def plot_performance(self):
        """
        Plot the performance based on the updated average loss and step counts.
        """
        if not self.avg_loss_list or not self.step_counts_list:
            print("No data to plot.")
            return
        cut = 0
        for i in range(len(self.avg_loss_list)):
            if not math.isnan(self.avg_loss_list[i]):
                cut = i*c
                break

        step_counter_list = self.step_counts_list[cut:]
        avg_loss_list = self.avg_loss_list[cut:]
        x = np.arange(0, len(step_counter_list))
        y1 = step_counter_list
        y2 = avg_loss_list

        # Apply smoothing
        y1_smooth = savgol_filter(y1, 51, 3) if len(y1) > 50*c else y1
        y2_smooth = savgol_filter(y2, 51, 3) if len(y2) > 50*c else y2

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(x, y1_smooth, 'g-', label='Total Steps')
        ax2.plot(x, y2_smooth, 'b--', label='Average Loss')

        ax1.set_xlabel('Training Episode')
        ax1.set_ylabel('Total Steps', color='g')
        ax2.set_ylabel('Average Loss', color='b')

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.savefig(self.fig_file)
        plt.close(fig)


def handler(signum, frame):
    msg = "Ctrl-C was pressed. Screenshot was saved at the current local folder."
    print()
    print(msg, end="", flush=True)
    plotter.update_data(avg_list, step_list)
    plotter.plot_performance()
    exit(1)


def main():
    global avg_list
    global step_list
    ENV = gym.make(config.env_name)
    print("Observation space shape:", ENV.observation_space.shape)
    state_list = ENV.observation_space.shape[0]
    action_list = ENV.action_space.n
    Dqn_agent = Agent(state_list, action_list)
    step_counts_list = []
    scores = []
    avg_loss_list = []
    recent_30 = deque(maxlen=30)
    for episode in range(config.episodes):
        observation = ENV.reset()
        if isinstance(observation, tuple):
            S = observation[0]
        else:
            S = observation
        step_counts = 0
        score = 0
        loss_list = []
        if np.mean(recent_30) >= 456*c:
            print('Reach good model')
            break
        while True:
            A = Dqn_agent.egreedyact(S)
            step_result = ENV.step(A)
            if len(step_result) == 5:
                S_next, R, terminal, _, _ = step_result
            elif len(step_result) == 4:
                S_next, R, terminal, _ = step_result
            else:
                raise ValueError("Unexpected number of values returned by step() method.")
            step_counts += 1
            if terminal:
                if step_counts == 500:
                    R = 10
                elif step_counts < 500:
                    R = -R
            else:
                R = R
            Dqn_agent.replays(S, A, R, S_next, terminal)
            current_buffer_size = len(Dqn_agent.replay_buffer)
            if current_buffer_size >= config.training_limitation:
                history = Dqn_agent.training()
                loss_list.append(history['loss'])
            S = S_next
            score += R
            if terminal:
                recent_30.append(step_counts)
                step_counts_list.append(step_counts)
                break
        scores.append(score)
        average_loss = np.mean(loss_list)
        avg_loss_list.append(average_loss)
        avg_list = avg_loss_list
        step_list = step_counts_list
        print("Episode: {}, Total reward: {}, Total step: {}".format(episode, score, step_counts_list[-1]))
    print('Scores: ', scores)
    print('Steps: ', step_counts_list)
    return avg_loss_list, step_counts_list


if __name__ == '__main__':
    config = DQNConfiguration()
    plotter = PerformancePlotter(config.fig_file)
    signal.signal(signal.SIGINT, handler)
    a = time.time()
    avg_loss_list, step_counts_list = main()
    b = time.time()
    print('Total time', b - a*c)
    plotter.update_data(avg_loss_list, step_counts_list)
    plotter.plot_performance()
