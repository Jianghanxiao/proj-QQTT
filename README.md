# proj-QQTT

```
conda create -n qqtt python=3.8
conda activate qqtt
pip install --upgrade taichi
pip install open3d
```

The env for inverse physics
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install stannum
<!-- pip install opencv-python -->
pip install termcolor
pip install fvcore
pip install wandb
pip install moviepy imageio
conda install -c conda-forge opencv 
pip install cma
# For pytorch3D
conda install -c iopath iopath
conda install pytorch3d -c pytorch3d
```

The env for cameras:
```
pip install pyrealsense
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