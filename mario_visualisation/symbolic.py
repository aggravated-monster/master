import glob
import os

import pandas as pd
from pandas.errors import EmptyDataError


def prep_train_timing(csv_folder):
    df_result = pd.DataFrame()
    # list all target models in directory
    files = glob.glob(csv_folder + '*train*')
    print(files)
    for file in files:
        try:
            df_xxx = pd.read_csv(file, header=None)
            # add column names
            df_xxx.columns = ['timestamp', 'task', 'name', 'seed', 'duration']
            df_result = pd.concat([df_result, df_xxx])
        except EmptyDataError:
            continue

    df_result.to_csv("timing_train.csv", index=False)


def prep_step_timing(csv_folder):
    df_result = pd.DataFrame()
    # list all target models in directory
    files = glob.glob(csv_folder + '*step*')
    print(files)
    for file in files:
        try:
            df_xxx = pd.read_csv(file, header=None)
            # add column names
            df_xxx.columns = ['timestamp', 'task', 'name', 'seed', 'duration']
            df_result = pd.concat([df_result, df_xxx])
        except EmptyDataError:
            continue

    df_result.to_csv("timing_step.csv", index=False)


def prep_symbolic_timing(csv_folder):
    df_result = pd.DataFrame()
    # list all target models in directory
    files = glob.glob(csv_folder + '*')
    print(files)
    for file in files:
        try:
            df_xxx = pd.read_csv(file, header=None)
            # add column names
            if "induction" in file:
                df_xxx.columns = ['timestamp', 'task', 'name', 'seed', 'attempt', 'episode', 'duration']
                # drop attempt and episode
                df_xxx.drop(['attempt', 'episode'], axis=1, inplace=True)
            else:
                df_xxx.columns = ['timestamp', 'task', 'name', 'seed', 'duration']
            df_result = pd.concat([df_result, df_xxx])
        except EmptyDataError:
            continue

    df_result.to_csv("timing_symbolic.csv", index=False)


if __name__ == '__main__':
    print(os.getcwd())
    prep_train_timing(csv_folder="../results/timing_experiments/logs/timing/")
    prep_step_timing(csv_folder="../results/timing_experiments/logs/timing/")
    prep_symbolic_timing(csv_folder="../results/timing_experiments/logs/timing/symbolic/")
    print("Prep done")
