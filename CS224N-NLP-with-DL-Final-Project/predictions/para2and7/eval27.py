import csv 
import pandas as pd
import matplotlib.pyplot as plt


para = {}

# from dct
with open('predictions/para2and7/dct_filter_all-para-dev-output.csv', 'r') as file:
   count = 0
   for line in file:
        if count == 0:
            count += 1
            continue
        line = line.strip()
        line = line.split(',')

        if line[0].rstrip() not in para:
            para[line[0].rstrip()] = {}
        # para_dct[line[0].rstrip()] = float(line[1])
        para[line[0].rstrip()]['dct'] = float(line[1])

file.close()

# from ground truth
with open('data/quora-dev.csv', 'r') as file:
   count = 0
   for line in file:
        if count == 0:
            count += 1
            continue
        line = line.strip()
        line = line.split('\t')
        # print(line)
        # para_best[line[0].rstrip()] = float(line[1])
        if len(line)!=5:
            continue
        para[line[1].rstrip()]['ground_truth'] = float(line[4])

file.close()

# from best performance model
with open('predictions/para2and7/2para-dev-output.csv', 'r') as file:
   count = 0
   for line in file:
        if count == 0:
            count += 1
            continue
        line = line.strip()
        line = line.split(',')
        
        # para_best[line[0].rstrip()] = float(line[1])
        # score_best[int(float(line[1]))]+= 1
        # if len(line)!=2:
        #     continue
        # if line[0].rstrip() not in para:
        #     para[line[0].rstrip()] = {}
        # print(line)
        para[line[0].rstrip()]['best2'] = float(line[1])

file.close()
# print(score_best)
# print(para)

cases = para.keys()
num_cases = len(cases)

dct_tp = 0
dct_tn = 0
dct_fp = 0
dct_fn = 0

best_tp = 0
best_tn = 0
best_fp = 0
best_fn = 0

for case in cases:
    if para[case]['dct'] > 0.5 and para[case]['ground_truth'] == 1:
        dct_tp += 1
    elif para[case]['dct'] <= 0.5 and para[case]['ground_truth'] == 0:
        dct_tn += 1
    elif para[case]['dct'] > 0.5 and para[case]['ground_truth'] == 0:
        dct_fp += 1
    elif para[case]['dct'] <= 0.5 and para[case]['ground_truth'] == 1:
        dct_fn += 1

    if para[case]['best2'] > 0.5 and para[case]['ground_truth'] == 1:
        best_tp += 1
    elif para[case]['best2'] <= 0.5 and para[case]['ground_truth'] == 0:
        best_tn += 1
    elif para[case]['best2'] > 0.5 and para[case]['ground_truth'] == 0:
        best_fp += 1
    elif para[case]['best2'] <= 0.5 and para[case]['ground_truth'] == 1:
        best_fn += 1

print('num_cases:', num_cases)
print('dct_tp:', dct_tp, 'dct_tp%:', dct_tp/num_cases*100, '%')
print('dct_tn:', dct_tn, 'dct_tn%:', dct_tn/num_cases*100, '%')
print('dct_fp:', dct_fp, 'dct_fp%:', dct_fp/num_cases*100, '%')
print('dct_fn:', dct_fn, 'dct_fn%:', dct_fn/num_cases*100, '%')
print('best_tp:', best_tp, 'best_tp%:', best_tp/num_cases*100, '%')
print('best_tn:', best_tn, 'best_tn%:', best_tn/num_cases*100, '%')
print('best_fp:', best_fp, 'best_fp%:', best_fp/num_cases*100, '%')
print('best_fn:', best_fn, 'best_fn%:', best_fn/num_cases*100, '%')



