from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
import scipy.stats as st
from sklearn.cluster import KMeans

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

    
    # find the closest 20% of trajectories for the highest density trajectory
    highest_density_x = sorted_all_pred_x[0]
    highest_density_y = sorted_all_pred_y[0]
    largest_distance = None
    closest_indexes = np.array([], dtype=np.int8)
    closest_distances = np.array([])
    
    for idx in range(len(sorted_all_pred_x)):
        distance = calculate_FDE(highest_density_x, highest_density_y, sorted_all_pred_x[idx], sorted_all_pred_y[idx])
        
        if len(closest_indexes) < no_of_traj*0.2:
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

    # cluster proportions of the whole trajectories
    first_cluster_end = params['CLUSTER_PERCENTAGES'][0]
    second_cluster_end = params['CLUSTER_PERCENTAGES'][1]
    third_cluster_end = params['CLUSTER_PERCENTAGES'][2]
    fourth_cluster_end = params['CLUSTER_PERCENTAGES'][3]

    
    # first cluster is a special case, it will contain top 20% of trajectories in a single cluster
    first_cluster_x = sorted_all_pred_x[0:int(no_of_traj*first_cluster_end)]
    first_cluster_y = sorted_all_pred_y[0:int(no_of_traj*first_cluster_end)]
      
    second_cluster_x = sorted_all_pred_x[int(no_of_traj*first_cluster_end):int(no_of_traj*second_cluster_end)]
    second_cluster_y = sorted_all_pred_y[int(no_of_traj*first_cluster_end):int(no_of_traj*second_cluster_end)]
    
    third_cluster_x = sorted_all_pred_x[int(no_of_traj*second_cluster_end):int(no_of_traj*third_cluster_end)]
    third_cluster_y = sorted_all_pred_y[int(no_of_traj*second_cluster_end):int(no_of_traj*third_cluster_end)]
    
    fourth_cluster_x = sorted_all_pred_x[int(no_of_traj*third_cluster_end):int(no_of_traj*fourth_cluster_end)]
    fourth_cluster_y = sorted_all_pred_y[int(no_of_traj*third_cluster_end):int(no_of_traj*fourth_cluster_end)]
    
    second_cluster_data = run_KMeans(second_cluster_x, second_cluster_y, no_of_clusters=5)
    third_cluster_data = run_KMeans(third_cluster_x, third_cluster_y, no_of_clusters=5)
    fourth_cluster_data = run_KMeans(fourth_cluster_x, fourth_cluster_y, no_of_clusters=5)
    
    # assign the mean trajectories and weights
    # Note: first cluster only has 1 trajectory and 1 weight
    first_c_pred_x = np.mean(first_cluster_x, axis=0)
    first_c_pred_y = np.mean(first_cluster_y, axis=0)
    first_c_no_of_traj = len(first_cluster_x)/no_of_traj
    
    # Rest of the clusters have 5 trajectories and a weight for each of them
    second_c_pred_x, second_c_pred_y, second_c_no_of_traj = second_cluster_data
    third_c_pred_x, third_c_pred_y, third_c_no_of_traj = third_cluster_data
    fourth_c_pred_x, fourth_c_pred_y, fourth_c_no_of_traj = fourth_cluster_data
   
    # Return (all_x_predictions, all_y_predictions, all_weights)
    return (
        [first_c_pred_x, *second_c_pred_x, *third_c_pred_x, *fourth_c_pred_x],
        [first_c_pred_y, *second_c_pred_y, *third_c_pred_y, *fourth_c_pred_y],
        [first_c_no_of_traj, *[i/no_of_traj for i in second_c_no_of_traj], *[i/no_of_traj for i in third_c_no_of_traj], *[i/no_of_traj for i in fourth_c_no_of_traj]]
    )
    
