import glob
import os
import json
import ast
import numpy as np

file_names = glob.glob(r"/home/ian/Desktop/School/zero_shot/zero_shot_atsc/nli_subtask4/results_nli_few_shot_in_domain/*")
print(len(file_names))


line_start = "{'accuracy':"
file_lists = {}

for f_name in file_names:

    print(f_name)
    file = open(f_name, 'r')
    lines = file.readlines()
    metrics=[]
    for line in lines:
        if line_start in line:
            #print(line)
            metrics.append(line)
    dic = ast.literal_eval(metrics[0].strip()[1:-3])
    group_name = f_name[0:-10]
    if group_name not in file_lists:
        file_lists[group_name] = []

    file_lists[group_name].append(dic)
    #print(dic)
    #print(dic['accuracy'])
    #print(os.path.basename(f_name) + metrics[0])
for key in file_lists:

    experiment = file_lists[key]
    accs = []
    f1s = []
    for e in experiment:
        accs.append(e['accuracy'])
        f1s.append(e['f1'])
    accs = np.array(accs)
    f1s = np.array(f1s)

    acc_mean = accs.mean()
    acc_se = accs.std() / np.sqrt(5)

    f1_mean = f1s.mean()
    f1_se = f1s.std() / np.sqrt(5)

    results = {'acc_mean' : acc_mean, 'acc_se': acc_se, 'f1_mean': f1_mean, 'f1_se': f1_se}
    print()
    print(key[-30:])
    print("acc_mean: " + str(results['acc_mean']))
    print("acc_se: " + str(results['acc_se']))
    print("f1_mean: " + str(results['f1_mean']))
    print("f1_se: " + str(results['f1_se']))
    print()
    #print(results)
