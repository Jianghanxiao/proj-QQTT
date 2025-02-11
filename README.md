# proj-QQTT

# ENV setup
```
conda create -y -n qqtt python=3.10
conda activate qqtt
bash ./env_install/env_install.sh
bash ./env_install/download_pretrained_models.sh
```

# Record data
Set the footswitch button
```
footswitch -1 -k space -2 -k space -3 -k space
```

Record the data (Controller need to be in the same location on the object in the whole sequence)
```
python cameras_calibrate.py
python record_data.py
```

# Process the data
```
python data_process/record_data_align.py
python data_process/data_process_pcd.py

# Get the masks from GroundedSAM2 (Go to GroundedSAM2 repo)
python test_real.py 
# Process the semantic mask to deprecate the points with bad depth
python data_process/data_process_mask.py
# Get the track data from CoTracker
python data_process/get_track.py
# Process the track data
python data_process/data_process_track.py
# Do shapre reconstruction for the first frame and do the downsampling of the points
python data_process/data_process_sample.py
```

# Optimize the physical parameters
python real_train.py