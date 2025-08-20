# Test_Waymo

[waymo_open_dataset_motion_v_1_1_0 dataset](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/validation;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%5B%5D%22))&inv=1&invt=Ab5Rgw&prefix=&forceOnObjectsSortingFiltering=false)


## Log Data
- [MTR's Guideline](#MTR's-Guideline)
- [Train log data (12/8/25)](log_data\log_train_20250812-153151.txt)
- [Test result of Best_model (12/8/25)](log_data/log_eval_20250815-185212.txt)
- [Test result of check point at 30th epoch (12/8/25)](log_data\log_eval_20250815-190435.txt)
- [Test result of Latest Model (12/8/25)](log_data\log_eval_20250815-Latest_Model.txt)
- [Best Evaluation at epoch 21 (12/8/25)](log_data\best_eval_record.txt)


## Performance on the validation set of Waymo Open Motion Dataset
|  Model  |  Training Set | minADE | minFDE | Miss Rate | mAP |
|---------|----------------|--------|--------|--------|--------|
|[MTR_mtr+20%_data](model\original_from_mtr\mtr+20_percent_data.yaml) | 20%            | 0.6697 | 1.3712 | 0.1668 | 0.3437 |
|self_train+20%_data | 20%            | 0.6697 | 1.3712 | 0.1668 | 0.3437 |
|[MTR_mtr+100%_data](model\original_from_mtr\mtr+100_percent_data.yaml)| 100%           | 0.6046 | 1.2251 | 0.1366 | 0.4164 |
|self_train+100%_data (still didn't)| 100%           | 0.6046 | 1.2251 | 0.1366 | 0.4164 |


## Self-train Testing
| Metric | best_model | latest_model | at 30_epoch |
|--------|----------------|-------------------|------------------|
| minADE | 1.6081          | 1.4284            | 3.4769           |
| minFDE | 3.3691          | 3.0869            | 6.0130           |
| MissRate | 0.4232          | 0.3954            | 0.8072           |
| mAP    | 0.1715          | 0.1808            | 0.0946           |


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

- use this "python test.py \\
  --cfg_file cfgs/waymo/mtr+100_percent_data.yaml \\
  --ckpt ../output/waymo/mtr+100_percent_data/my_first_exp/ckpt/best_model.pth \\
  --batch_size 1"
- in test.py, usually just loads the "best_model.pth" into the model code but there are some other ways
- in test.py, 1. eval_single_ckpt() (only best_model.pth) gives final performance of how well of the trained model
2. repeat_eval_ckpt() (multiple checkpoint) shows how better the model gets over checkpoints (only do this if wanting to debug the way you train)
		  
```