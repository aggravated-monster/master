import glob
import os

import pandas as pd


def prep_comparison(csv_folder):
    df_result = pd.DataFrame()
    # list all target models in directory
    files = glob.glob(csv_folder + '*target*')
    print(files)
    for file in files:
        file_name = os.path.basename(file)
        model_name = os.path.splitext(file_name)[0]
        parts = model_name.split("_")

        df_xxx = pd.read_csv(file, header=None)
        # add column names
        df_xxx.columns = ['timestamp', 'name', 'seed', 'total_steps', 'episode', 'episode_steps', 'reward', 'distance',
                          'velocity', 'game_time', 'game_score', 'flag', 'loss', 'epsilon', 'model_id']
        df_xxx["bin"] = int(parts[-5])/100000
        if "B1" in parts[-6]:
            df_xxx["model"] = "B1"
            df_xxx["display_name"] = str(int(int(parts[-5])/100000)) + "K"
        elif "B2" in parts[-6]:
            df_xxx["model"] = "B2"
            df_xxx["display_name"] = str(int(int(parts[-5])/100000)) + "K"
        elif "T1" in model_name:
            df_xxx["model"] = "T1 discr ni"
            df_xxx["display_name"] = str(int(int(parts[-5])/100000)) + "K"

        df_result = pd.concat([df_result, df_xxx])

    df_result.to_csv("comp.csv", index=False)



if __name__ == '__main__':
    print(os.getcwd())
    prep_comparison(csv_folder="../results/comp/")
    print("Prep done")