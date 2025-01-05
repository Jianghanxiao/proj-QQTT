# proj-QQTT

```
conda create -n qqtt python=3.10
conda activate qqtt
pip install warp-lang
pip install usd-core matplotlib pyglet
pip install open3d
```

The env for inverse physics
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install stannum
pip install termcolor
pip install fvcore
pip install wandb
pip install moviepy imageio
conda install -c conda-forge opencv 
pip install cma
```

The env for cameras:
```
pip install Cython
pip install pyrealsense2
pip install atomics
pip install pynput
```

For record new data
```
# Do camera calibrations to get the extrinsic parameters
python cameras_calibrate.py 
# Record thr raw data
python record_data.py
# Do the video alignment
python record_data_align.py
# Process to get the PCD data for each frame
python data_process_pcd.py
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

Process the data
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