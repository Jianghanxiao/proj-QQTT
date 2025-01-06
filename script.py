import os

# for i in range(6, 30):
#     os.system(f"python data_process/data_process_mask.py --case_name rope_{i}")

for i in range(2, 30):
    os.system(f"python data_process/get_track.py --case_name rope_{i}")

for i in range(2, 30):
    os.system(f"python data_process/data_process_track.py --case_name rope_{i}")

for i in range(2, 30):
    os.system(f"python data_process/data_process_sample.py --case_name rope_{i}")