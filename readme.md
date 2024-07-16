## Code for AnDi 2

### Environment

Run the following command to create our conda environment:
```
conda env create -f environment.yml
```
However, conda cannnot correctly install GPU version PyTorch, so install it with pip:
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

For track 1 video trajectory linking, we use MATLAB and TrackMate.

### Reproducing results for Challenge Phase

#### Track 2

Our model checkpoints are put in the folder `output` with different time stamps. First, change the variable `public_data_path` in test_for_submit_file_convert.py to the challenge dataset path. Then, make sure the model path `addre` is valid. To reproduce the results of track 2, run the following command to get network predictions.
```
cd code/andi_2
python3 test_for_submit_file_convert.py
```
Then, we can get the track 2 single and ensemble results:
```
python3 track2_single.py
python3 track2_ens.py
```
The final results are in the directory `challenge_results/track_2`

#### Track 1

First, use TrackMate to link trajectories from videos in MATLAB. Run `main.m` in folder `code/traj_link` and change line 7 to challenge dataset path.

Change line 179 to the dataset path and run `main.py` to convert the files into our data format:

```
cd traj_link
python3 main.py
```

Then, follow the same procedure in Track 2. To reproduce the results of track 1, run the following command to get network predictions. For vip particle prediction, change the `task='vip'` in line 89 of file `test_for_submit_file_convert_track1_single.py`. For ensemble prediction, change the `task='all'` in line 88 to predict all particles' outputs. 

```
cd code/andi_2
python3 test_for_submit_file_convert_track1_single.py
```
Remember to run track 2 task before track 1 task, since there are some calibrations code in track 2. Then, we can get the track 1 vip and ensemble results:
```
python3 track1_vip.py
python3 track1_ens.py
```
The final results are in the directory `challenge_results/track_1`

### Training Procedure

First, a training dataset should be generated with `trackdata.py`

To train the model, change the `train_address` and `valid_address` and run the python command below to get a model predicting alpha and K.
```
python3 maintrack_others.py
```

Besides, we have also a two-stage pipeline for predicting alpha, K, state and changepoints.
```
python3 maintrack_pipeline.py
```