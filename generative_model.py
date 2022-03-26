from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
import scipy.stats as st
from sklearn.cluster import KMeans
from tqdm import tqdm

def get_model(params):
    if np.random.rand() < params['CONST_VEL_MODEL_PROB']:
        return 'CONST_VEL'
    else:
        return 'CONST_VEL_W_ROTATION'
    
def get_action(params):
    if np.random.rand() < params['STOP_PROB']:
        return 'STOP'
    if np.random.rand() < params['VELOCITY_CHANGE_PROB']:
        return 'VELOCITY_CHANGE'
    if np.random.rand() < params['ANGLE_CHANGE_PROB']:
        return 'ANGLE_CHANGE'
    return None

# Decide whether to use average or discounted average as const velocity
def get_const_vel(params, sample_vel_x, sample_vel_y):
    if np.random.rand() < params['DISCOUNT_AVG_PROB']:
        discount = np.random.uniform(low=params['DISCOUNT_LOWER_BOUND'])
        const_vel_x = (discount**3*sample_vel_x[0] + discount**2*sample_vel_x[1] + discount*sample_vel_x[2] + sample_vel_x[3]) / (discount**3 + discount**2 + discount + 1)
        const_vel_y = (discount**3*sample_vel_y[0] + discount**2*sample_vel_y[1] + discount*sample_vel_y[2] + sample_vel_y[3]) / (discount**3 + discount**2 + discount + 1)
    else:
        const_vel_x = np.mean(sample_vel_x)
        const_vel_y = np.mean(sample_vel_y)
        
    return const_vel_x, const_vel_y

# Decide whether to use average or discounted average angle
def get_angle(params, sample_vel_x, sample_vel_y):
    all_angles = []
    for i in range(1, len(sample_vel_x)):
        prev_vel = [sample_vel_x[i-1], sample_vel_y[i-1]]
        curr_vel = [sample_vel_x[i], sample_vel_y[i]]
        one_angle = np.math.atan2(np.linalg.det([prev_vel, curr_vel]),np.dot(prev_vel, curr_vel))   
        all_angles.append(one_angle)
    
    if np.random.rand() < params['DISCOUNT_AVG_PROB']:
        discount = np.random.uniform(low=params['DISCOUNT_LOWER_BOUND'])
        angle = (discount**2*all_angles[0] + discount*all_angles[1] + all_angles[2]) / (discount**2 + discount + 1)
    else:
        angle = np.mean(all_angles)
    return angle

def calculate_FDE(pred_x, pred_y, test_x, test_y):

    final_displacement_x = pred_x[-1] - test_x[-1]
    final_displacement_y = pred_y[-1] - test_y[-1]
    FDE = np.sqrt(final_displacement_x**2 + final_displacement_y**2)
    
    return FDE

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def generate_trajectory(sample_x, sample_y, params, length=5):
    # calculate velocity data
    sample_vel_x = [(sample_x[i] - sample_x[i-1]) + np.random.normal(0, params['NOISE']) for i in range(1, len(sample_x))]
    sample_vel_y = [(sample_y[i] - sample_y[i-1]) + np.random.normal(0, params['NOISE']) for i in range(1, len(sample_y))]
    
    const_vel_x, const_vel_y = get_const_vel(params, sample_vel_x, sample_vel_y)
    angle = get_angle(params, sample_vel_x, sample_vel_y)
    
    # start predicting
    pred_x = []
    pred_y = []
    model = get_model(params)
    for i in range(length):
        action = get_action(params)
        if action == 'STOP':
            if len(pred_x) == 0:
                pred_x.append(sample_x[-1])
                pred_y.append(sample_y[-1])
            else:
                pred_x.append(pred_x[-1])
                pred_y.append(pred_y[-1])
            continue
            
        elif action == 'VELOCITY_CHANGE':
            const_vel_x = const_vel_x + np.random.normal(0, params['VELOCITY_CHANGE_NOISE'])
            const_vel_y = const_vel_y + np.random.normal(0, params['VELOCITY_CHANGE_NOISE'])
        elif action == 'ANGLE_CHANGE':
            angle = angle + np.random.normal(0, params['ANGLE_CHANGE_NOISE'])
        
        if model == 'CONST_VEL':
            if len(pred_x) == 0:
                pred_x.append(sample_x[-1] + const_vel_x)
                pred_y.append(sample_y[-1] + const_vel_y)
            else:
                pred_x.append(pred_x[-1] + const_vel_x)
                pred_y.append(pred_y[-1] + const_vel_y)
        elif model == 'CONST_VEL_W_ROTATION':
            if len(pred_x) == 0:
                prev_x = sample_x[-2]
                prev_y = sample_y[-2]
                cur_x = sample_x[-1] + const_vel_x
                cur_y = sample_y[-1] + const_vel_y
            else:
                prev_x = pred_x[-1]
                prev_y = pred_y[-1]
                cur_x = pred_x[-1] + const_vel_x
                cur_y = pred_y[-1] + const_vel_y
            rot_x, rot_y = rotate((prev_x, prev_y), (cur_x, cur_y), angle)
            pred_x.append(rot_x)
            pred_y.append(rot_y)
            
            # redefine the average velocity as it now has a new heading
            const_vel_x = rot_x - prev_x
            const_vel_y = rot_y - prev_y
        
    return pred_x, pred_y
            
def run_KMeans(pred_x_list, pred_y_list, no_of_clusters):
    final_points = []
    # this can be optimized as this is done in the parent method already
    for i in range(len(pred_x_list)):
        final_points.append([pred_x_list[i][-1], pred_y_list[i][-1]])
    
    Kmean = KMeans(n_clusters=no_of_clusters)
    Kmean.fit(final_points)

    cluster_avg_x, cluster_avg_y, no_of_elements_per_cluster = get_clustered_averages(pred_x_list, pred_y_list, no_of_clusters, Kmean.labels_)
    # Should probably also return labels or do the averaged clusters already contain them?
    return cluster_avg_x, cluster_avg_y, no_of_elements_per_cluster

def get_clustered_averages(all_pred_x, all_pred_y, no_of_clusters, cluster_labels):
    no_of_elements_per_cluster = []
    for i in range(no_of_clusters):
        no_of_elements_per_cluster.append((cluster_labels == i).sum())
    
    clustered_preds_x = [[] for _ in range(no_of_clusters)]
    clustered_preds_y = [[] for _ in range(no_of_clusters)]

    for idx, cluster_label in enumerate(cluster_labels):
        clustered_preds_x[cluster_label].append(all_pred_x[idx])
        clustered_preds_y[cluster_label].append(all_pred_y[idx])
        
    cluster_avg_x = [[] for _ in range(no_of_clusters)]
    cluster_avg_y = [[] for _ in range(no_of_clusters)]
    
    for i in range(no_of_clusters):
        cluster_avg_x[i] = np.mean(clustered_preds_x[i], axis=0)
        cluster_avg_y[i] = np.mean(clustered_preds_y[i], axis=0)
        
    return cluster_avg_x, cluster_avg_y, no_of_elements_per_cluster

def predict(sample_x, sample_y, params, trajectory_length=5):
    all_pred_x, all_pred_y = [], []
    all_final_x, all_final_y = [], []
    
    for i in range(params['NO_OF_TRAJECTORIES']):
        pred_x, pred_y = generate_trajectory(sample_x, sample_y, params, trajectory_length)
        all_pred_x.append(pred_x)
        all_pred_y.append(pred_y)
        
        all_final_x.append(pred_x[-1])
        all_final_y.append(pred_y[-1])
    
    # run kernel density estimate
    values = np.vstack([all_final_x, all_final_y])
    kernel = st.gaussian_kde(values)
    # evaluate trajectories
    evaluated = kernel.evaluate(values)
    # find the sorting order for the trajectories based on KDE pdf
    sorting_order = evaluated.argsort()[::-1] # Note: the sorting order is ascending by default, [::-1] reverses the order (might be too slow though?)
    
    # sort predictions by KDE density
    sorted_all_pred_x = np.array(all_pred_x)[sorting_order]
    sorted_all_pred_y = np.array(all_pred_y)[sorting_order]
    
    # distribute the data to representative sets
    no_of_traj = len(sorted_all_pred_x)

    
    # find the closest TOP% (first cluster size) of trajectories for the highest density trajectory
    highest_density_x = sorted_all_pred_x[0]
    highest_density_y = sorted_all_pred_y[0]
    largest_distance = None
    closest_indexes = np.array([], dtype=np.int8)
    closest_distances = np.array([])
    
    for idx in range(len(sorted_all_pred_x)):
        distance = calculate_FDE(highest_density_x, highest_density_y, sorted_all_pred_x[idx], sorted_all_pred_y[idx])
        
        if len(closest_indexes) < no_of_traj*params['GROUP_PERCENTAGES'][0]:
            closest_indexes = np.append(closest_indexes, idx)
            closest_distances = np.append(closest_distances, distance)
            if largest_distance == None or largest_distance < distance:
                largest_distance = distance
        else:
            if largest_distance > distance:
                index_max = np.argmax(closest_distances)
                closest_indexes[index_max] = idx
                closest_distances[index_max] = distance
                largest_distance = np.amax(closest_distances)
    
    closest_x = sorted_all_pred_x[closest_indexes]
    closest_y = sorted_all_pred_y[closest_indexes]

    # remove the closest trajectories from the data...
    sorted_all_pred_x = np.delete(sorted_all_pred_x, closest_indexes, axis=0)
    sorted_all_pred_y = np.delete(sorted_all_pred_y, closest_indexes, axis=0)
    
    # ...and move them to the front of the array
    sorted_all_pred_x = np.append(closest_x, sorted_all_pred_x, axis=0)
    sorted_all_pred_y = np.append(closest_y, sorted_all_pred_y, axis=0)
    
    # Return values will be in a format of [pred_xs: list, pred_ys: list, pred_weigths: list]
    return_values = [[], [], []]
    
    
    ## Loop over the representative groups and run K-means clustering on each
    ## (if group should return more than 1 representative trajectory)
    
    prev_group_size_end = 0
    group_size_ends = params['GROUP_PERCENTAGES']
    for group_idx, group_size_end in enumerate(group_size_ends):
        group_cluster_count = params['GROUP_CLUSTER_COUNT'][group_idx]
        
        group_x = sorted_all_pred_x[int(no_of_traj*prev_group_size_end):int(no_of_traj*group_size_end)]
        group_y = sorted_all_pred_y[int(no_of_traj*prev_group_size_end):int(no_of_traj*group_size_end)]
        
        # No need to run k-means clustering if the group is supposed to have just 1 representative cluster
        if group_cluster_count == 1:
            representative_x = np.mean(group_x, axis=0)
            representative_y = np.mean(group_y, axis=0)
            trajectory_weight = len(representative_x)/no_of_traj
            
            return_values[0].append(representative_x)
            return_values[1].append(representative_y)
            return_values[2].append(trajectory_weight)
        else:
            group_data = run_KMeans(group_x, group_y, no_of_clusters=group_cluster_count)
            group_pred_xs, group_pred_ys, group_no_of_trajs = group_data
            
            return_values[0] = return_values[0] + [*group_pred_xs]
            return_values[1] = return_values[1] + [*group_pred_ys]
            return_values[2] = return_values[2] + [*[i/no_of_traj for i in group_no_of_trajs]]
        
        prev_group_size_end = group_size_end
   
    # Return (all_x_predictions, all_y_predictions, all_weights)
    return return_values
    