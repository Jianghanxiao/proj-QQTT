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
This one use the base environment, the ffmpeg has some version conflict issue. This is okay for now, the released data doesn't need call this env
```
python data_process/record_data_align.py --case_name [] --start [] --end []
```
After this step, the data includes the color (frames and video), depth, calibrate.pkl and metadata.json (Should also be the realeas of our data, about 500 MB)

```
# Get the masks from GroundedSAM2
python test_real.py 

python data_process/data_process_pcd.py
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