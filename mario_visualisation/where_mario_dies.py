import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def run():
    df_baseline = pd.read_csv(
        "../results/5000eps/20240428-17.08.30_baseline_original/logs/explain/20240428-17.08.30_baseline_original_episodes.csv",
        header=None)
    # add column names
    df_baseline.columns = ['timestamp', 'name', 'seed', 'total_steps', 'episode', 'episode_steps', 'reward',
                            'distance', 'velocity', 'game_time', 'game_score', 'flag', 'loss', 'epsilon', 'dataset']
    plt.figure(figsize=(60, 6))
    sns.histplot(data=df_baseline, x='reward', bins=3500, color='black')
    plt.show()

    df_T1 = pd.read_csv(
        "../results/20240511-11.23.06_T1_ext_ni_episodes.csv",
        header=None)
    # add column names
    df_T1.columns = ['timestamp', 'name', 'seed', 'total_steps', 'episode', 'episode_steps', 'reward',
                            'distance', 'velocity', 'game_time', 'game_score', 'flag', 'loss', 'epsilon', 'dataset']
    plt.figure(figsize=(60, 6))
    sns.histplot(data=df_T1, x='reward', bins=3500, color='black')
    plt.show()

if __name__ == '__main__':
    print(os.getcwd())
    run()
    print("Prep done")