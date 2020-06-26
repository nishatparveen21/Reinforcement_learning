def train_agent(using_tkinter, agent, method, gamma, learning_rate, eps_start, eps_end, eps_decay,
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

        # for sarsa - agent selects next action based on observation
        if (method == "sarsa") or (method == "sarsa_lambda"):
            current_action = agent.choose_action(current_observation, masked_actions_list, greedy_epsilon)
            if method == "sarsa_lambda":
                # for sarsa_lambda - initial all zero eligibility trace
                agent.reset_eligibility_trace()

        if method_used == "mc_control":
            # generate an episode by following epsilon-greedy policy
            episode = []
            current_observation, _ = env.reset()
            for step_id in range(max_nb_steps):  # while True
                current_action = agent.choose_action(tuple(current_observation), masked_actions_list, greedy_epsilon)
                next_observation, reward, termination_flag, masked_actions_list = env.step(current_action)
                return_of_episode += reward

                # a tuple is hashable and can be used in defaultdict
                episode.append((tuple(current_observation), current_action, reward))
                current_observation = next_observation

                if termination_flag:
                    step_counter = step_id
                    steps_counter_list.append(step_id)
                    returns_list.append(return_of_episode)
                    break

            # update the action-value function estimate using the episode
            # print("episode = {}".format(episode))
            # agent.compare_reference_value()
            agent.learn(episode, gamma, learning_rate)

        else:  # TD
            # run episodes
            for step_id in range(max_nb_steps):
                # ToDo: how to penalize the agent that does not terminate the episode?

                # fresh env
                if using_tkinter:
                    env.render(sleep_time)

                if (method == "sarsa") or (method == "sarsa_lambda"):
                    next_observation, reward, termination_flag, masked_actions_list = env.step(current_action)
                    return_of_episode += reward
                    if not termination_flag:  # if done
                        # Online-Policy: Choose an action At+1 following the same e-greedy policy based on current Q
                        # ToDo: here, we should read the masked_actions_list associated to the next_observation
                        masked_actions_list = env.masking_function(next_observation)
                        next_action = agent.choose_action(next_observation, masked_actions_list=masked_actions_list,
                                                          greedy_epsilon=greedy_epsilon)

                        # agent learn from this transition
                        agent.learn(current_observation, current_action, reward, next_observation, next_action,
                                    termination_flag, gamma, learning_rate)
                        current_observation = next_observation
                        current_action = next_action

                    if termination_flag:  # if done
                        agent.learn(current_observation, current_action, reward, next_observation, next_action,
                                    termination_flag, gamma, learning_rate)
                        # ToDo: check it ignore next_observation and next_action
                        step_counter = step_id
                        steps_counter_list.append(step_id)
                        returns_list.append(return_of_episode)
                        break

                elif (method == "q") or (method == "expected_sarsa") or (method == "simple_dqn_pytorch"):
                    current_action = agent.choose_action(current_observation, masked_actions_list, greedy_epsilon)
                    next_observation, reward, termination_flag, masked_actions_list = env.step(current_action)
                    return_of_episode += reward

                    if method == "q":
                        # agent learn from this transition
                        agent.learn(current_observation, current_action, reward, next_observation, termination_flag,
                                    gamma, learning_rate)

                    elif method == "simple_dqn_pytorch":
                        agent.step(current_observation, current_action, reward, next_observation, termination_flag)

                    elif method == "expected_sarsa":
                        agent.learn(current_observation, current_action, reward, next_observation, termination_flag,
                                    greedy_epsilon, gamma, learning_rate)

                    else:  # DQN with tensorflow
                        # New: store transition in memory - subsequently to be sampled from
                        agent.store_transition(current_observation, current_action, reward, next_observation)

                        # if the number of steps is larger than a threshold, start learn ()
                        if (step_id > 5) and (step_id % 5 == 0):  # for 1 to T
                            # print('learning')
                            # pick up some transitions from the memory and learn from these samples
                            agent.learn()

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
#     parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(parent_dir, "results/simple_road/" + folder_name)
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

def test_agent(using_tkinter_test, agent, returns_list, nb_episodes=1, max_nb_steps=20, sleep_time=0.001,
               weight_file_name="q_table.pkl"):
    """
    load weights and show one run
    :param using_tkinter_test: [bool]
    :param agent: [brain object]
    :param returns_list: [float list] - argument passed by reference
    :param nb_episodes: [int]
    :param max_nb_steps: [int]
    :param sleep_time: [float]
    :param weight_file_name: [string]
    :return: -
    """
    grand_parent_dir_test = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weight_file = os.path.abspath(grand_parent_dir_test + "/results/simple_road/" + weight_file_name)
    if agent.load_q_table(weight_file):

        for episode_id in range(nb_episodes):
            trajectory = []
            # reset the environment for a new episode
            current_observation, masked_actions_list = env.reset()  # initial observation = initial state
            print("{} = initial_observation".format(current_observation))
            score = 0  # initialize the score
            step_id = 0
            while step_id < max_nb_steps:
                step_id += 1
                # fresh env
                if using_tkinter_test:
                    env.render(sleep_time)

                # agent choose current_action based on observation
                greedy_epsilon = 0
                current_action = agent.choose_action(current_observation, masked_actions_list, greedy_epsilon)

                next_observation, reward, termination_flag, masked_actions_list = env.step(current_action)

                score += reward  # update the score

                trajectory.append(current_observation)
                trajectory.append(current_action)
                trajectory.append(reward)
                trajectory.append(termination_flag)

                # update state
                current_observation = next_observation
                print("\r {}, {}, {}.".format(current_action, reward, termination_flag), end="")
                sys.stdout.flush()
                if termination_flag:  # exit loop if episode finished
                    trajectory.append(next_observation)
                    break

            returns_list.append(score)
            print("\n{}/{} - Return: {}".format(episode_id, nb_episodes, score))
            print("\nTrajectory = {}".format(trajectory))
            # Best trajectory= [[0, 3], 'no_change', [3, 3], 'no_change', [6, 3], 'no_change', [9, 3], 'slow_down',
            # [11, 2], 'no_change', [13, 2], 'speed_up', [16, 3], 'no_change', [19, 3]]

        print("---")
        print("{} = average return".format(np.mean(returns_list)))
    else:
        print("cannot load weight_file at {}".format(weight_file))