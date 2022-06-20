import pandas as pd

# Helper functions for reading the OpenTraj datasets with padnas
def load_eth(dataset_root='./raw_data'):
    csv_columns = ["frame_id", "agent_id", "pos_x", "pos_z", "pos_y", "vel_x", "vel_z", "vel_y"]
    data = pd.read_csv(dataset_root + "/ETH/seq_eth/obsmat.txt", sep=r"\s+", header=None, names=csv_columns)

    return data

def load_hotel(dataset_root='./raw_data'):
    csv_columns = ["frame_id", "agent_id", "pos_x", "pos_z", "pos_y", "vel_x", "vel_z", "vel_y"]
    data = pd.read_csv(dataset_root + "/ETH/seq_hotel/obsmat.txt", sep=r"\s+", header=None, names=csv_columns)

    return data
