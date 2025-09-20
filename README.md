# Test_Waymo

[waymo_open_dataset_motion_v_1_1_0 dataset](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/validation;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%5B%5D%22))&inv=1&invt=Ab5Rgw&prefix=&forceOnObjectsSortingFiltering=false)


## Log Data
- [MTR's Guideline](#MTR's-Guideline)
- [Train log data (12/8/25)](log_data\log_train_20250812-153151.txt)
- [Test result of Best_model (12/8/25)](log_data/log_eval_20250815-185212.txt)
- [Test result of check point at 30th epoch (12/8/25)](log_data\log_eval_20250815-190435.txt)
- [Test result of Latest Model (12/8/25)](log_data\log_eval_20250815-Latest_Model.txt)
- [Best Evaluation at epoch 21 (12/8/25)](log_data\best_eval_record.txt)




## Training Loss
  - sampling with 100 loss data per epoch (1-29 epoch)
![Training Loss with Tread](log_data/Training_loss_with_trend%20(Cleaned).svg)
- Total iterations: 2900 ()
- Loss range: -242.095000 to 113454.273000
- Mean loss: 448.810496
- Std loss: 3142.328200

## Performance on the validation set of Waymo Open Motion Dataset
|  Model  |  Training Set | minADE | minFDE | Miss Rate | mAP |
|---------|----------------|--------|--------|--------|--------|
|MTR_mtr+100%_data | 100%           | 0.6046 | 1.2251 | 0.1366 | 0.4164 |
|self_train+100%_data| 100%           | 1.6081  | 3.3691 | 0.4232 | 0.1715 |


## Self-train Models Testing
| Metric | best_model | latest_model | at 30_epoch |
|--------|----------------|-------------------|------------------|
| minADE | 1.6081          | 1.4284            | 3.4769           |
| minFDE | 3.3691          | 3.0869            | 6.0130           |
| MissRate | 0.4232          | 0.3954            | 0.8072           |
| mAP    | 0.1715          | 0.1808            | 0.0946           |


## Input Data Structure
based on preprocessing script of tfrecord to .pkl 
1. decode_tracks_from_proto [mtr/datasets/waymo/data_preprocess.py]
```
  : Output track_infos = 
    - 'object_id': List of agent IDs
    - 'object_type': List of agent types (vehicle, pedestrian, etc.)
    - 'trajs': Array of shape (num_objects, num_timestamps, 10)
  
    For each agent at each timestep: 
    * where 10 is info related to center agent
    trajectory_point = [
        center_x,     # global X position (meters)
        center_y,     # global Y position (meters) 
        center_z,     # global Z position (meters)
        length,       # bounding box length (meters)
        width,        # bounding box width (meters)
        height,       # bounding box height (meters)
        heading,      # orientation angle (radians, global frame)
        velocity_x,   # velocity in X direction (m/s, global frame)
        velocity_y,   # velocity in Y direction (m/s, global frame)
        valid         # flag (1.0 = valid observation, 0.0 = missing/invalid data)
    ]
```

2. decode_map_features_from_proto [mtr/datasets/waymo/data_preprocess.py]
    - map_infos = {'all_polylines': np.array([...]),  # Shape: (total_map_points, 7)
```
    * Categorized map elements with metadata:
      'lane': [{'id':,'speed_limit_mph':,'type':,'interpolating':,'entry_lanes':,'exit_lanes':,   
               'left_boundary':,'right_boundary':,'polyline_index':}, ...],
      'road_line': [{'id':,'type':,'polyline_index':}, ...],
      'road_edge': [{'id':,'type':, 'polyline_index':}, ...],
      'stop_sign': [{'id':,'lane_ids':,'position':'polyline_index':}, ...],
      'crosswalk': [{'id':,'polyline_index':}, ...],
      'speed_bump': [{'id':,'polyline_index':}, ...]}

    Each point in all_polylines array:
    map_point = [
        x,           # Global X position (meters)
        y,           # Global Y position (meters)
        z,           # Global Z position (meters)
        dir_x,       # Direction vector X component (normalized)
        dir_y,       # Direction vector Y component (normalized) 
        dir_z,       # Direction vector Z component (normalized)
        global_type  # Integer type ID (from waymo_types.py polyline_type)
    ]
```

3. process_waymo_data_with_scenario_proto [mtr/datasets/waymo/data_preprocess.py]
```
  : Main processing function for each .tfrecord containing multiple scenarios to individual .pkl files per scenario
    - Reads TensorFlow records
    - Parses scenario protobuf data
    - Extracts tracks, map, and metadata
    - Saves each scenario as individual .pkl file
    - Output: sample_SCENARIO_ID.pkl files
```
```
TREE : 
  Preprocessing (.tfrecord → .pkl):
  ├── Agent trajectories: GLOBAL coordinates (10D)
  ├── Map polylines: GLOBAL coordinates (7D)
  └── Metadata: scenario info, prediction targets

  Training Pipeline (waymo_dataset.py):
  ├── Transform to CENTER AGENT coordinates
  ├── Agent features: 10D → 29D (add time, type, heading encoding)
  ├── Map features: 7D → 9D (add prev_x, prev_y)
  └── Feed to model
```
## Single Scenerio Structure
```
scenario_data = {
    #  METADATA 
    'scenario_id': 'scenario_123',
    'timestamps_seconds': [0.0, 0.1, 0.2, ..., 9.0],  # 91 timesteps (9.1 seconds)
    'current_time_index': 10,  # Split: 0-10 past, 11-90 future
    'sdc_track_index': 0,  # Which agent is the self-driving car
    'objects_of_interest': [],  # Special objects (usually empty)

    # PREDICTION TARGETS
    'tracks_to_predict': {
        'track_index': [1, 2, 5],  # Which agents to predict
        'difficulty': [1, 2, 1],   # Prediction difficulty level (1=easy, 5=hard)
        'object_type': ['TYPE_VEHICLE', 'TYPE_VEHICLE', 'TYPE_PEDESTRIAN']
    },

     # AGENT TRAJECTORIES 
    'track_infos': {
        'object_id': [101, 102, 103, ...],  
        'object_type': ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', ...],
        'trajs': np.array([...])  # (num_agents, 91_timesteps, 10)
        # GLOBAL coordinates: [x, y, z, length, width, height, heading, vx, vy, valid]
    },

    # HD MAP DATA
    'map_infos': {
        'all_polylines': np.array([...]),  # (num_points, 7)
        # GLOBAL coordinates: [x, y, z, dir_x, dir_y, dir_z, global_type]
        
        'lane': [{'id': ..., 'speed_limit_mph': ..., 'type': ..., 'polyline_index': (start, end)}, ...],
        'road_line': [{'id': ..., 'type': ..., 'polyline_index': (start, end)}, ...],
        'road_edge': [{'id': ..., 'type': ..., 'polyline_index': (start, end)}, ...],
        'stop_sign': [{'id': ..., 'position': [x, y, z], 'lane_ids': [...], 'polyline_index': (start, end)}, ...],
        'crosswalk': [{'id': ..., 'polyline_index': (start, end)}, ...],
        'speed_bump': [{'id': ..., 'polyline_index': (start, end)}, ...]
    },
    # DYNAMIC TRAFFIC SIGNALS
    'dynamic_map_infos': {
        'lane_id': [...],    # Traffic signal lane IDs per timestep
        'state': [...],      # Signal states per timestep  
        'stop_point': [...]  # Stop point positions per timestep
    }
}
```



### MTR's Guideline
```bash
# --------------------------------- PRE-PROCRESS ---------------------------------
- save raw data of .tfrecord in /home/ioon/MTR/data/waymo
- should be installing Waymo API first, if already then skip
- run this command to pre-process raw data of .tfcord (tensorflow binary format) into .pkl (more suitable file for python) "cd mtr/datasets/waymo
python data_preprocess.py ../../../data/waymo/scenario/  ../../../data/waymo"
- you should have the more structeral data and cleaner data in /home/ioon/MTR/data
(contains : processed_scenarios_training, processed_scenarios_validation, processed_scenarios_training_infos.pkl, processed_scenarios_val_infos.pkl)
- processed_scenarios_training keeps each scenario for tainning process
- processed_scenarios_validation keeps each scenario for validation step in each epoch of training process

# --------------------------------- TRAIN (in /tools)-----------------------------
(we're having 1 GPU that handles batch_size of 1, dist_test.sh is not needed, run straight from train.py)
- an error might causes by mis-typo in train.py "--local_rank" (probably should be "--local-rank")
- to fix, in train.py -> parser.add_argument('--local_rank', '--local-rank', type=int, default=None, help='local rank for distributed training')
- to avoid the conflict, and gpu size use command "cd tools
python train.py --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size 1 --epochs 30 --extra_tag my_first_exp", this should start training the "processed_scenarios_training" folder
- train.py handles setup and configuration, train_utils.py contains core training loops.

---------- VALIDATION (happens during training) ---------
- when each epoch is done, it will auto-validate with "processed_scenarios_validation" folder just to monitor the progress if it's getting any better
- train.py calls fucntion in train_utils.py which are train one epoch, save checkpoint, run validation and save best model. DETAILED CODE IN "def train_model()" IN TRAIN_UTILS.PY

# --------------------------------- MODEL ----------------------------------------
- we have brain-less model in model.py (in mtr/model folder) at fisrt
- after done training, model with learned weights should be in /home/ioon/MTR/output/waymo/mtr+100_percent_data/my_first_exp/ckpt, focus on "best_model.pth" (the last epoch isn't always the smartest)
- best_model is saved if (current_mAP > best_mAP)

# --------------------------------- TEST ------------------------------------------
(handling the GPU as same as the train step)

- use this "python test.py \
  --cfg_file cfgs/waymo/mtr+100_percent_data.yaml \
  --ckpt ../output/waymo/mtr+100_percent_data/my_first_exp/ckpt/best_model.pth \
  --batch_size 1"
- in test.py, usually just loads the "best_model.pth" into the model code but there are some other ways
- in test.py, 1. eval_single_ckpt() (only best_model.pth) gives final performance of how well of the trained model
2. repeat_eval_ckpt() (multiple checkpoint) shows how better the model gets over checkpoints (only do this if wanting to debug the way you train)
- sometimes best_model.pth does not give the best result at mAP, it could be the latest_model that is the highest, just haven't been evaluated yet
	      
```