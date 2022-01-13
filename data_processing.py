import os
import sys
sys.path.insert(0, os.path.abspath(os.getcwd()) + '/OpenTraj/opentraj') # Anaconda python can't find the toolkit path without this for some reason

from toolkit.loaders.loader_edinburgh import load_edinburgh
from matplotlib import pyplot as plt
import numpy as np

def read_edinburgh_data(num_steps=5):
    opentraj_root = './OpenTraj/'
    selected_day = '01Sep' # 3 days of data in total, ['01Jul', '01Aug', '01Sep']
    edinburgh_path = os.path.join(opentraj_root, 'datasets/Edinburgh/annotations', 'tracks.%s.txt' % selected_day)
    traj_dataset = load_edinburgh(edinburgh_path, title="Edinburgh", 
                                  use_kalman=False, scene_id=selected_day, sampling_rate=4)

    data = traj_dataset.data

    # Removing any agents which don't have enough steps for model fitting and predicting
    agent_ids = data.agent_id.unique()
    for agent_id in agent_ids:
        if len(data[data.agent_id == agent_id]) < 2 * num_steps:
            data = data[data.agent_id != agent_id]
    agent_ids = data.agent_id.unique()

    # 'Normalize' the data so that all trajectories will begin at x=0, y=0
    for agent_id in agent_ids:
        first_x = data[data.agent_id == agent_id]['pos_x'].iloc[0]
        first_y = data[data.agent_id == agent_id]['pos_y'].iloc[0]

        data.loc[data.agent_id == agent_id, 'pos_x'] = data[data.agent_id == agent_id]['pos_x'] - first_x
        data.loc[data.agent_id == agent_id, 'pos_y'] = data[data.agent_id == agent_id]['pos_y'] - first_y
        
    return data, agent_ids