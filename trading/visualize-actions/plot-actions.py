import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/.."))

import datetime
from misc import utils
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator
import numpy as np

filenames = [
    #"mgvt_1c",
    #"t2vt_1c",
    #"source_2014",
    "source_2015",
    "source_2016",
    "source_2017",
    #"2015",
    #"2016",
    #"2017",
    #"2018"
]

for filename in filenames:
    results = utils.load_object("visualize-actions/" + filename)
    days = results[0]
    actions = results[1]
    rewards = results[2]

    # transpose actions matrix
    #actions = np.transpose(actions)

    # rewards cumulative sum
    rewards = np.sum(rewards, axis = 1)
    rewards = np.cumsum(rewards)

    def format_time(value, tick_number):
        hours = str(int(2 + value // 60))
        minutes = int(value % 60)
        if minutes < 10:
            minutes = "0" + str(minutes)
        else:
            minutes = str(minutes)
        return hours + ":" + minutes

    def format_date(value, tick_number):
        if value < 0 or value > len(days):
            return ""
        origin = datetime.date(year = 1900, month = 1, day = 1)
        delta = datetime.timedelta(days = int(days[int(value)]) - 2)
        date = origin + delta
        return str(date.day) + "/" + str(date.month)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 8), gridspec_kw={'height_ratios': [1, 2]})

    final_reward = np.around(rewards[-1], decimals = 4)

    ax1.plot(rewards)
    ax1.set_title("Cumulative sum of rewards - " + filename + " (final reward: " + str(final_reward) + ")")
    ax1.set(xlabel = "Date", ylabel = "Cumulative reward")
    ax1.set_xbound(lower=0, upper=len(days))
    ax1.xaxis.set_major_locator(MultipleLocator(50))
    ax1.xaxis.set_major_formatter(FuncFormatter(format_date))
    ax1.yaxis.set_major_locator(MaxNLocator(5))
    ax1.grid()

    plot = ax2.imshow(actions, cmap='GnBu')
    ax2.set_title("Actions heatmap - " + filename)
    ax2.set(xlabel = "Time of day", ylabel = "Date")
    ax2.xaxis.set_major_locator(MultipleLocator(300))
    ax2.xaxis.set_major_formatter(FuncFormatter(format_time))
    ax2.yaxis.set_major_locator(MultipleLocator(50))
    ax2.yaxis.set_major_formatter(FuncFormatter(format_date))
    fig.colorbar(plot, ticks=[-1, 0, 1], fraction=0.0109, pad=0.01)

    plt.savefig("visualize-actions/" + filename + ".png", format='png')

    plt.show()