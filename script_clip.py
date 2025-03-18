import os
import glob
import csv

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/past_data_collect/more_clothes"
output_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/more_clothes"

# Read the csv
with open(f"clip_start_end.csv", "r") as f:
    reader = csv.reader(f)
    data = list(reader)

data = data[1:]
starts = {}
ends = {}
for row in data:
    case_name, start, end = row
    starts[case_name] = int(start)
    ends[case_name] = int(end)

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    print(f"Processing {case_name}")

    os.system(f"python data_process/record_data_align.py --base_path {base_path} --case_name {case_name} --output_path {output_path} --start {starts[case_name]} --end {ends[case_name]}")

    