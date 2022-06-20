import numpy as np
import json
import cv2
import os
import argparse
import time
import generative_model

from data_processing import load_eth, load_hotel
from pyqt.opentrajui import OpenTrajUI

def error_msg(msg):
    print('Error:', msg)
    exit(-1)

default_params = {
    'NOISE': 0.025, 
    'NO_OF_TRAJECTORIES': 200, 
    'CONST_VEL_MODEL_PROB': 0.5, 
    'STOP_PROB': 0.025, 
    'DISCOUNT_AVG_PROB': 1.0, 
    'DISCOUNT_LOWER_BOUND': 0.15, 
    'VELOCITY_CHANGE_PROB': 0.1,
    'VELOCITY_CHANGE_NOISE': 0.1, 
    'ANGLE_CHANGE_PROB': 0.2, 
    'ANGLE_CHANGE_NOISE': 2, 
    'GROUP_PERCENTAGES': [0.2, 0.48], 
    'GROUP_CLUSTER_COUNT': [1, 3]
}

class Play:
    def __init__(self, model_params, recording):
        self.qtui = OpenTrajUI(reserve_n_agents=100)
        self.agent_index = -1
        self.model_params = model_params
        self.recording = recording

    def is_a_video(self, filename):
        return '.mp4' in filename or '.avi' in filename

    def to_image_frame(self, Hinv, loc):
        """
        Given H^-1 and world coordinates, returns (u, v) in image coordinates.
        """
        if loc.ndim > 1:
            locHomogenous = np.hstack((loc, np.ones((loc.shape[0], 1))))
            loc_tr = np.transpose(locHomogenous)
            loc_tr = np.matmul(Hinv, loc_tr)  # to camera frame
            locXYZ = np.transpose(loc_tr/loc_tr[2])  # to pixels (from millimeters)
            return locXYZ[:, :2].astype(int)
        else:
            locHomogenous = np.hstack((loc, 1))
            locHomogenous = np.dot(Hinv, locHomogenous)  # to camera frame
            locXYZ = locHomogenous / locHomogenous[2]  # to pixels (from millimeters)
            return locXYZ[:2].astype(int)

    def set_background_im(self, im, timestamp=-1):
        self.bg_im = im.copy()
        self.qtui.update_im(im)
        self.qtui.erase_paths()
        self.qtui.erase_circles()
        if timestamp >= 0:
            self.qtui.setTimestamp(timestamp)

    def draw_trajectory(self, ll, color, width):
        self.qtui.draw_path(ll[..., ::-1], color, [width])

        if self.recording == True:
            for tt in range(ll.shape[0] - 1):
                cv2.line(self.bg_im, (ll[tt][1], ll[tt][0]), (ll[tt + 1][1], ll[tt + 1][0]), tuple(reversed(color)), width, cv2.LINE_AA)

    def draw_agent(self, pos, radius, color, width):
        self.qtui.draw_circle(pos, radius, color, width)

        if self.recording == True:
            cv2.circle(self.bg_im, (pos[0], pos[1]), radius, tuple(reversed(color)), 0, cv2.LINE_AA)

    def play(self, traj_dataset, Hinv, media_file):
        frame_ids = sorted(traj_dataset['frame_id'].unique())

        if os.path.exists(media_file):
            print("media file exists")
            if self.is_a_video(media_file):
                cap = cv2.VideoCapture(media_file)
            else:
                ref_im = cv2.imread(media_file)

        if self.recording == True:
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
            size = (width, height)
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, size)

        agent_ids = []
        t = frame_ids[0]
        pause = False
        predicted_trajectories_prev = []

        # Get the entire historical trajectories
        #all_trajs_whole = traj_dataset.groupby('agent_id')
        #all_trajs_whole = [v[['pos_x', 'pos_y']].to_numpy() for k, v in all_trajs_whole]

        while True:
            if self.is_a_video(media_file) and not pause:
                cap.set(cv2.CAP_PROP_POS_FRAMES, t-1)
                ret, ref_im = cap.read()
            self.set_background_im(ref_im, t)

            # used for showing the preds when paused
            if not pause and t % 4 == 0:
                predicted_trajectories_prev = []

            # Get the historical trajectories for last 50 frames
            all_trajs = traj_dataset[
                            (traj_dataset['frame_id'] <= t) &
                            (traj_dataset['frame_id'] > t - 50)
                            ].groupby('agent_id')
            
            agent_ids = [k for k, v in all_trajs]
            all_trajs = [v[['pos_x', 'pos_y']].to_numpy() for k, v in all_trajs]

            for i, id in enumerate(agent_ids):
                # draw the historical trajectory for the last 50 frames
                traj_i = all_trajs[i]
                TRAJ_i = self.to_image_frame(Hinv, traj_i)
                self.draw_trajectory(TRAJ_i, (255, 255, 0), 2)

                # draw the entire history
                #traj_whole_i = all_trajs_whole[i]
                #TRAJ_whole_i = self.to_image_frame(Hinv, traj_whole_i)
                #self.draw_trajectory(TRAJ_whole_i, (255, 255, 0), 1)

                predicted_trajectories = []
                if len(traj_i) >= 3 and not pause and t%4 == 0:
                    x_coords = [point[0] for point in traj_i]
                    y_coords = [point[1] for point in traj_i]
                    pred_result = generative_model.predict(x_coords, y_coords, self.model_params, trajectory_length=8)
                    pred_xs, pred_ys = pred_result[0], pred_result[1]

                    for pred_no in range(len(pred_xs)):
                        one_pred_x = pred_xs[pred_no]
                        one_pred_y = pred_ys[pred_no]
                        predicted_trajectories.append(np.column_stack((one_pred_x, one_pred_y)))
                    predicted_trajectories_prev += predicted_trajectories

                # draw the current location
                xy_i = all_trajs[i][-1]
                UV_i = self.to_image_frame(Hinv, xy_i)
                self.draw_agent((UV_i[1], UV_i[0]), 5, (255, 255, 255), 2)

            # draw predictions
            for predicted_trajectory in predicted_trajectories_prev:
                PRED_i = self.to_image_frame(Hinv, predicted_trajectory)
                self.draw_trajectory(PRED_i, (124,252,0), 1)

            if not agent_ids and not pause:
                print('No agent')

            if not pause and t < frame_ids[-1]:
                t += 1
                if self.recording == True:
                    out.write(self.bg_im)

            delay_ms = 20
            self.qtui.processEvents()
            pause = self.qtui.pause
            time.sleep(delay_ms/1000.)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='OpenTraj - Human Trajectory Dataset Package')
    argparser.add_argument('--data-root', '--data-root', default='./raw_data',
                           help='the root address of OpenTraj datasets')
    argparser.add_argument('--dataset', '--dataset',
                           default='eth',
                           choices=['eth',
                                    'hotel'],
                           help='select dataset'
                                '(default: "eth")')
    argparser.add_argument('--model-params', '--model-params',
                        help='path of the .json file of the parameters for the prediction model')

    argparser.add_argument('--record', '--record', action='store_true', default='False')

    args = argparser.parse_args()
    data_root = args.data_root
    traj_dataset = None
    recording = args.record

    params_json_path = args.model_params
    if params_json_path == None:
        model_params = default_params
    else:
        with open(params_json_path, "r") as params_file:
            model_params = json.load(params_file)

    if args.dataset == 'eth':
        annot_file = os.path.join(data_root, 'ETH/seq_eth/obsmat.txt')
        traj_dataset = load_eth(data_root)
        homog_file = os.path.join(data_root, 'ETH/seq_eth/H.txt')
        media_file = os.path.join(data_root, 'ETH/seq_eth/video.avi')

    elif args.dataset == 'hotel':
        annot_file = os.path.join(data_root, 'ETH/seq_hotel/obsmat.txt')
        traj_dataset = load_hotel(data_root)
        homog_file = os.path.join(data_root, 'ETH/seq_hotel/H.txt')
        media_file = os.path.join(data_root, 'ETH/seq_hotel/video.avi')

    else:
        error_msg('Unsupported dataset')

    Homog = (np.loadtxt(homog_file)) if os.path.exists(homog_file) else np.eye(3)
    Hinv = np.linalg.inv(Homog)

    play = Play(model_params, recording)
    play.play(traj_dataset, Hinv, media_file)
    # qtui.app.exe()
