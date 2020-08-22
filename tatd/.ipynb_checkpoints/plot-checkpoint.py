import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style ='ticks', 
        rc={'figure.figsize':(8, 15), 'figure.dpi': 400, 
            'font.size': 20.0 ,'font.weight': 'normal',
            'axes.labelsize': 18.0,"lines.linewidth": 0.3,
            'axes.labelweight': 'normal'})


color_code = ['#498C8A','#F5BA2A', '#F64D2A', '#8BBB21', '#2E95EC', '#E2711C']


def plot_sparsity1(data, name, dtype):
    
    markers = ["o","v","x","*",'X','d',"P","4", "3", "P", "*", 'X', 'd']

    g = sns.catplot(data = data, x = 'percent', y = 'Error', 
                    hue = 'Model',
                    margin_titles=True, dodge=True,
                    markers = markers, kind = 'point', 
                    height=5, aspect=1.2,
                    palette= color_code, scale = 5.5,
                    linestyles=["--", "--","--", "--", "--"],
                    hue_order = ['standard','kernel','1d', 
                                 'attention_dot', 'standalone'],
                   sharex=False, sharey=False, legend = False)
    (g.set_axis_labels("", "")
     .set_xticklabels(size=18)
     .set_yticklabels(size=18)
     .set_titles("{col_name} {col_var}", size = 20)
     .set_axis_labels('Sparsity', 'Train RMSE'))


    g.savefig(f'../best/{name}_sparsity_{dtype}.pdf', dpi = 300)
    