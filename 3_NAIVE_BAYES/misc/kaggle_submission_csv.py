import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--file")

args = parser.parse_args()

in_file = open(args.file)

file_lines = in_file.readlines()
output_csv = {"submission" : [], "private_score" : [], "public_score" : [], "details" : []}

for i,l in enumerate(file_lines):
    curr_str_idx = i%6
    act_str = l.strip()
    if(curr_str_idx == 0):
        output_csv["submission"].append(act_str)
    if(curr_str_idx == 1):
        output_csv["details"].append(act_str)
    if(curr_str_idx == 3):
        output_csv["private_score"].append(float(act_str))
    if(curr_str_idx == 5):
        output_csv["public_score"].append(float(act_str))

pd.DataFrame(output_csv).to_csv("./kaggle_submissions.csv", index=None)