import csv
import os

base_path = "./data/different_types"
exp_path = "./experiments"

# Read csv
with open("out_domain.csv", "r") as f:
    reader = csv.reader(f)
    data = list(reader)

for row in data:
    from_case, to_case = row
    print(f"Processing {from_case} to {to_case}")
    os.system(
        f"python data_process/outdomain_align.py --base_path {base_path} --from_case {from_case} --to_case {to_case} --exp_path {exp_path}"
    )
