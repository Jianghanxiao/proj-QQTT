# proj-QQTT

# ENV setup
```
conda create -n qqtt python=3.10
conda activate qqtt
pip install warp-lang
pip install usd-core matplotlib pyglet
pip install open3d
pip install trimesh
```

The env for inverse physics
```
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install stannum
pip install termcolor
pip install fvcore
pip install wandb
pip install moviepy imageio
conda install -c conda-forge opencv 
pip install cma
conda install pytorch3d -c pytorch3d
```

The env for cameras:
```
pip install Cython
pip install pyrealsense2
pip install atomics
pip install pynput
```

The env for GroundedSAM2
```
pip install git+https://github.com/IDEA-Research/Grounded-SAM-2.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

The env for SDXL
```
pip install diffusers
```

The env for trellis
```
cd data_process
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS
. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

```

# Download the checkpoints for GroundedSAM2
```
mkdir data_process/groundedSAM_checkpoints
cd data_process/groundedSAM_checkpoints
wget ./data_process/groundedSAM_checkpoints/
wget https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

# Download the checkpoints for superglue
```
mkdir data_process/models/weights
cd data_process/models/weights
wget https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superglue_indoor.pth
wget https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superglue_outdoor.pth
wget https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superpoint_v1.pth
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