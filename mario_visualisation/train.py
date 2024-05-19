import glob
import os

import pandas as pd


def prep_comparison(csv_folder):
    df_result = pd.DataFrame()
    # list all target models in directory
    files = glob.glob(csv_folder + '*episodes*')
    print(files)
    for file in files:
        file_name = os.path.basename(file)
        model_name = os.path.splitext(file_name)[0]
        parts = model_name.split("_")

        df_xxx = pd.read_csv(file, header=None)
        # add column names
        df_xxx.columns = ['timestamp', 'name', 'seed', 'total_steps', 'episode', 'episode_steps', 'reward', 'distance',
                          'velocity', 'game_time', 'game_score', 'flag', 'loss', 'epsilon', 'model_id']
        df_xxx['ewm_reward'] = df_xxx['reward'].ewm(alpha=0.0001).mean()
        df_xxx['ewm_wins'] = df_xxx['flag'].ewm(alpha=0.0001).mean()

        df_result = pd.concat([df_result, df_xxx])

    df_result.to_csv("train.csv", index=False)



if __name__ == '__main__':
    print(os.getcwd())
    prep_comparison(csv_folder="../results/train/")
    print("Prep done")