import os
import numpy as np
import math
import time
from collections import deque


def train_agent(env, using_tkinter, agent, method, gamma, learning_rate, eps_start, eps_end, eps_decay,
                window_success, threshold_success, returns_list, steps_counter_list, info_training,
                max_nb_episodes, max_nb_steps, sleep_time, folder_name=""):
    """

    :param using_tkinter: [bool] to display the environment, or not
    :param agent: [brain object]
    :param method: [string] value-based learning method - either sarsa or q-learning
    :param gamma: [float between 0 and 1] discount factor
    If gamma is closer to one, the agent will consider future rewards with greater weight,
    willing to delay the reward.
    :param learning_rate: [float between 0 and 1] - Non-constant learning rate must be used?
    :param eps_start: [float]
    :param eps_end: [float]
    :param eps_decay: [float]
    :param window_success: [int]
    :param threshold_success: [float] to solve the env, = average score over the last x scores, where x = window_success
    :param returns_list: [list of float]
    :param steps_counter_list: [list of int]
    :param info_training: [dict]
    :param max_nb_episodes: [int] limit of training episodes
    :param max_nb_steps: [int] maximum number of timesteps per episode
    :param sleep_time: [int] sleep_time between two steps [ms]
    :param folder_name: [string] to distinguish between runs during hyper-parameter tuning
    :return: [list] returns_list - to be displayed
    """
    returns_window = deque(maxlen=window_success)  # last x scores, where x = window_success

    # probability of random choice for epsilon-greedy action selection
    greedy_epsilon = eps_start

    # record for each episode:
    # steps_counter_list = []  # number of steps in each episode - look if some get to max_nb_steps
    # returns_list = []  # return in each episode
    best_trajectories_list = []

    # track maximum return
    max_return = -math.inf  # to be set low enough (setting max_nb_steps * max_cost_per_step should do it)
    max_window = -np.inf

    # initialize updated variable
    current_action = None
    next_observation = None

    # measure the running time
    time_start = time.time()
    nb_episodes_seen = max_nb_episodes
    #
    for episode_id in range(max_nb_episodes):  # limit the number of episodes during training
        # while episode_id < max_nb_episodes
        # episode_id = episode_id + 1

        # reset metrics
        step_counter = max_nb_steps  # length of episode
        return_of_episode = 0  # = score
        trajectory = []  # sort of replay-memory, just for debugging
        rewards = []
        actions = []
        changes_in_state = 0
        reward = 0
        next_action = None

        # reset the environment for a new episode
        current_observation, masked_actions_list = env.reset()  # initial observation = initial state

        # run episodes
        for step_id in range(max_nb_steps):
            # ToDo: how to penalize the agent that does not terminate the episode?

            # fresh env
            if using_tkinter:
                env.render(sleep_time)

            current_action = agent.choose_action(current_observation, masked_actions_list, greedy_epsilon)
            next_observation, reward, termination_flag, masked_actions_list = env.step(current_action)
            return_of_episode += reward

            agent.learn(current_observation, current_action, reward, next_observation, termination_flag,
                            gamma, learning_rate)

            current_observation = next_observation

            if termination_flag:  # if done
                step_counter = step_id
                steps_counter_list.append(step_id)
                returns_list.append(return_of_episode)
                # agent.compare_reference_value()

                break

            # log
            trajectory.append(current_observation)
            trajectory.append(current_action)

            # monitor actions, states and rewards are not constant
            rewards.append(reward)
            actions.append(current_action)
            if not (next_observation[0] == current_observation[0]
                    and next_observation[1] == current_observation[1]):
                changes_in_state = changes_in_state + 1

        # At this point, the episode is terminated
        # decay epsilon
        greedy_epsilon = max(eps_end, eps_decay * greedy_epsilon)

        # log
        trajectory.append(next_observation)  # final state
        returns_window.append(return_of_episode)  # save most recent score
        if episode_id % 100 == 0:
            time_intermediate = time.time()
            print('\n --- Episode={} ---\n eps={}\n Average Score in returns_window = {:.2f} \n duration={:.2f}'.format(
                episode_id, greedy_epsilon, np.mean(returns_window), time_intermediate - time_start))
            # agent.print_q_table()

        if episode_id % 20 == 0:
            print('Episode {} / {}. Eps = {}. Total_steps = {}. Return = {}. Max return = {}, Top 10 = {}'.format(
                episode_id+1, max_nb_episodes, greedy_epsilon, step_counter, return_of_episode, max_return,
                sorted(returns_list, reverse=True)[:10]))

        if return_of_episode == max_return:
            if trajectory not in best_trajectories_list:
                best_trajectories_list.append(trajectory)
        elif return_of_episode > max_return:
            del best_trajectories_list[:]
            best_trajectories_list.append(trajectory)
            max_return = return_of_episode

        if np.mean(returns_window) > max_window:
            max_window = np.mean(returns_window)

        # test success
        if np.mean(returns_window) >= threshold_success:
            time_stop = time.time()
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}, duration={:.2f} [s]'.format(
                episode_id - window_success, np.mean(returns_window), time_stop - time_start))
            info_training["nb_episodes_to_solve"] = episode_id - window_success
            nb_episodes_seen = episode_id
            break

    time_stop = time.time()
    info_training["duration"] = int(time_stop - time_start)
    info_training["nb_episodes_seen"] = nb_episodes_seen
    info_training["final_epsilon"] = greedy_epsilon
    info_training["max_window"] = max_window
    info_training["reference_values"] = agent.compare_reference_value()

    # where to save the weights
    folder = os.path.join("results_qlearning/" + folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    agent.save_q_table(folder)

    print('End of training')
    print('Best return : %s --- with %s different trajectory(ies)' % (max_return, len(best_trajectories_list)))
    for trajectory in best_trajectories_list:
        print(trajectory)

    if using_tkinter:
        env.destroy()

    # return returns_list, steps_counter_list