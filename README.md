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
python process_data.py --base_path [] --case_name [] --category []

# Need to do alignment first, then can do sampling

# Do shapre reconstruction for the first frame and do the downsampling of the points
python data_process/data_process_sample.py
```

# Optimize the physical parameters
python real_train.py