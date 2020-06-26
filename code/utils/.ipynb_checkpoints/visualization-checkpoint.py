import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time  # to time the learning process
import json  # to get the configuration of the environment
from collections import deque
import math
from utils.logger import Logger

def display_results(agent, method_used_to_plot, returns_to_plot, smoothing_window, threshold_success,
                    steps_counter_list_to_plot, display_flag=True, folder_name=""):
    """
    Use to SAVE + plot (optional)
    :param agent:
    :param method_used_to_plot:
    :param returns_to_plot:
    :param smoothing_window:
    :param threshold_success:
    :param steps_counter_list_to_plot:
    :param display_flag:
    :param folder_name:
    :return:
    """
    # where to save the plots
#     parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(parent_dir, "results/simple_road/" + folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # plot step_counter for each episode
    plt.figure()
    plt.grid(True)
    plt.xlabel('Episode')
    plt.title("Episode Step_counts over Time (Smoothed over window size {})".format(smoothing_window))
    plt.ylabel("Episode step_count (Smoothed)")
    steps_smoothed = pd.Series(steps_counter_list_to_plot).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(steps_counter_list_to_plot, linewidth=0.5)
    plt.plot(steps_smoothed, linewidth=2.0)
    plt.savefig(folder + "step_counter.png", dpi=800)
    if display_flag:
        plt.show()

    plt.figure()
    plt.grid(True)
    returns_smoothed = pd.Series(returns_to_plot).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(returns_to_plot, linewidth=0.5)
    plt.plot(returns_smoothed, linewidth=2.0)
    plt.axhline(y=threshold_success, color='r', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Episode Return(Smoothed)")
    plt.title(f"Episode Return over Time (Smoothed over window size {smoothing_window})")
    plt.savefig(folder + "return.png", dpi=800)
    if display_flag:
        plt.show()

    # bins = range(min(returns_to_plot), max(returns_to_plot) + 1, 1)
    plt.figure()
    plt.hist(returns_to_plot, norm_hist=True, bins=100)
    plt.ylabel('reward distribution')
    if display_flag:
        plt.show()

    agent.print_q_table()
    if method_used_to_plot != "simple_dqn_tensorflow":
        agent.plot_q_table(folder, display_flag)
        agent.plot_optimal_actions_at_each_position(folder, display_flag)