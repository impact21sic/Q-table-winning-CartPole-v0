import gym
import numpy as np
import time
import os 
from matplotlib import pyplot as plt


env = gym.make("CartPole-v0")
env.reset()
episodes = 20000
show_every = 2000
alpha = 0.1
epsilon = 0.1
#epsilon_decaying = epsilon / (1-(episodes//2)) if uncomented - epsilon should be 1
gamma = 0.9
action_space = env.action_space.n
obs_space = len(env.observation_space.high)
# Discretization of 2 observation values which are infinite
num_bins = 12
bins = [
    np.linspace(-4.8, 4.8, num_bins),
    np.linspace(-5, 5, num_bins),
    np.linspace(-.418, .418, num_bins),
    np.linspace(-5, 5, num_bins)
]


def discretize_states(observation, bins, obs_space):
    state = []
    for i in range(obs_space):
        state.append(int(np.digitize(observation[i], bins[i]) - 1))

    return tuple(state)


def create_q_table(num_bins, bins, action_space, obs_space):
    q_table = np.zeros([num_bins]*obs_space+[action_space])

    return q_table


def train(q_table, render = False, plot = False):
    global epsilon
    previous_moves = []  # Array of all scores over runs
    metrics = {'ep': [], 'avg': [], 'min': [], 'max': []}  # Metrics recorded for graph

    for episode in range(episodes):
        dis_s = discretize_states(env.reset(), bins, obs_space)
        done = False
        moves = 0 # How many moves the cart has made

        while not done:
            if episode % show_every == 0 and render:
                env.render()

            moves += 1
            # Take action from the Q table 
            if np.random.random() > epsilon:
                action = np.argmax(q_table[dis_s])
            # Take random action
            else:
                action = np.random.randint(0, action_space)

            s1, r, done, _ = env.step(action) # Perform an action on the environment
            dis_s1 = discretize_states(s1, bins, obs_space)
            max_q = np.max(q_table[dis_s1]) # Estimation of optimal future reward
            current_q = q_table[dis_s + (action,)] # The old Q value

            # Poll fell, the card went out of the bounds
            if done and moves < 200:
                r = -100

            # Bellman equation for all Q values
            new_q = (1-alpha)*current_q+alpha*(r+gamma*max_q)
            q_table[dis_s + (action,)] = new_q # Update the Q table
            dis_s = dis_s1

        previous_moves.append(moves)

        """
        if 0.1 < epsilon <= 1:
            if episodes//2 >= episode >= 1:
                epsilon += epsilon_decaying
        else:
            epsilon = 0.1
        """

        if episode % 100 == 0:
            latest_runs = previous_moves[-100:]
            average_moves = sum(latest_runs) / len(latest_runs)
            metrics['ep'].append(episode)
            metrics['avg'].append(average_moves)
            metrics['min'].append(min(latest_runs))
            metrics['max'].append(max(latest_runs))

            print("Run:", episode, "Average:", average_moves, "Min:", min(latest_runs), "Max:", max(latest_runs))

    env.close()

    if plot:
        plt.plot(metrics['ep'], metrics['avg'], label="average rewards")
        plt.plot(metrics['ep'], metrics['min'], label="min rewards")
        plt.plot(metrics['ep'], metrics['max'], label="max rewards")
        plt.legend(loc=4)
        plt.show()

    return q_table


def test_main():
    q_table = create_q_table(num_bins, bins, action_space, obs_space)
    discrete_state = discretize_states(env.reset(), bins, obs_space)
    print(discrete_state)
    print("Bins' shape: ", np.array(bins).shape)
    print("Q table's shape: ", q_table.shape)
    print("State's shape: ", np.array(discrete_state).shape)
    print(q_table[discrete_state])
    print(q_table[discrete_state + (0,)])
    print(q_table[(1,1,1,1,0)])


def main():
    q_table = train(create_q_table(num_bins, bins, action_space, obs_space),render=True,plot=True)


if __name__ == "__main__":
    main()
