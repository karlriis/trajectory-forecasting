# trajectory-forecasting
This repository contains the code for my Master's thesis on the topic of forecasting human trajectories https://comserv.cs.ut.ee/ati_thesis/datasheet.php?language=en

The new trajectory forecasting method resides in `generative_model.py`.
The main function for using the method is `predict()`. Its input parameters are shown in the table below (non-compulsory parameters with default values are marked with \*).

| Parameter name  | Parameter value | Meaning |
| ------------- | ------------- | ------------- |
| sample_x  | List of floats, e.g. [1.0, 1.8, 2.3, ...]  | The x-coordinates of an observed historical trajectory |
| sample_y  | List of floats, e.g. [1.0, 1.8, 2.3, ...]  | The y-coordinates of an observed historical trajectory |
| params  | Dictionary (possible values brought out separately below)  | The parameters for our method which manipulate the inner workings of the method |
| trajectory_length*  | Integer (default 5)  | Marks the length of the predicted trajectories |
| clustering_method*  | String ('KMeans' or 'KMedoids', default 'Kmeans') | The clustering method used for choosing representative predictions |
| smoothing*  | Boolean (default True) | Applies smoothing to representative predictions to reduce sudden turns |

The possible parameter values for the method are as follows:
```
params = {
    'NOISE': 0.05, 
    'NO_OF_TRAJECTORIES': 500, 
    'CONST_VEL_MODEL_PROB': 0.5, 
    'STOP_PROB': 0.025, 
    'DISCOUNT_AVG_PROB': 1.0, 
    'DISCOUNT_LOWER_BOUND': 0.15, 
    'VELOCITY_CHANGE_PROB': 0.1,
    'VELOCITY_CHANGE_NOISE': 0.1, 
    'ANGLE_CHANGE_PROB': 0.2, 
    'ANGLE_CHANGE_NOISE': 2, 
    'GROUP_PERCENTAGES': [0.1, 0.5, 0.75, 1.0], 
    'GROUP_CLUSTER_COUNT': [1, 9, 6, 4]
}
```

The method outputs multiple trajectory predictions with probabilities as follows:
```
[
  [[0.4, 0.6, 0.7, ...], [1.3, 1.4, 1.6, ...], ...], # The x-coordinate values for each future trajectory 
  [[0.9, 0.8, 0.5, ...], [1.1, 0.8, 0.4, ...], ...], # The y-coordinate values for each future trajectory
  [0.2, 0.11, 0.07, ...]                             # The probability/weight of each future trajectory
]
```
The main experiment regarding the thesis is in the jupyter notebook `our_method_results_comparison.ipynb`.

Some other experiments require a library from OpenTraj (ones which use data from the Edinburgh dataset). 
To run these, follow the following steps beforehand:
1. clone https://github.com/crowdbotp/OpenTraj into the current folder  
2. pip install pykalman (required for the dataset loaders toolkit)  
