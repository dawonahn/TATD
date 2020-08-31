import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(path):

    df = pd.read_csv(path, sep = '\t')

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(1, 1, 1)

    markers = ["o", "^", "^", "o", "4"]
    color = ('dodgerblue', 'seagreen', 'seagreen', '')
    lines = ('--', '-', '-.', ':')

    ms = 16
    mew = 2.5

    j = 0
    c_dot = [[ii, df.iloc[ii][1], df.iloc[ii][2]] for ii in range(df.shape[0]) if ii % 500 == 0]
    c_dot = np.transpose(c_dot)
    ax.plot(c_dot[1], c_dot[2],
            color=color[j], linewidth=3, linestyle=lines[j], marker=markers[j],
            markerfacecolor='None', markersize=ms, markeredgewidth=mew, )
    j += 1

    c_dot = [[ii, df.iloc[ii][1], df.iloc[ii][4]] for ii in range(df.shape[0]) if ii % 500 == 0]
    c_dot = np.transpose(c_dot)
    ax.plot(c_dot[1], c_dot[2],
            color=color[j], linewidth=3, linestyle=lines[j], marker=markers[j],
            markerfacecolor='None', markersize=ms, markeredgewidth=mew, )

    plt.yticks(size=21)
    plt.xticks(size=21)
    plt.xlabel('Running Time (seconds)', size=25)
    plt.ylabel('Validation RMSE', size=25)

    labels = list(['Train', 'Validation'])
    legend = plt.legend(labels, fontsize=14,
                        bbox_to_anchor=(0, 1, 1, 0.2),
                        ncol=2, frameon=False, loc='center')
    plt.subplots_adjust(hspace=0.1, bottom=0.2, left=0.2)
    # plt.savefig(f'../../../../fig_temp/{name}.pdf', dpi = 500, tight = True)
