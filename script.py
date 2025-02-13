import os

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types"

with open("case_category.txt", "r") as f:
    lines = f.readlines()

os.system("rm -f timer.log")

for line in lines:
    case_name, category = line.split()
    print(case_name, category)
    os.system(
        f"python process_data.py --base_path {base_path} --case_name {case_name} --category {category}"
    )
