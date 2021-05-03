import glob
import os
import json
import ast
import numpy as np

file_names = glob.glob(r"C:\Users\ibirl\Documents\dsProjects\zero_shot_atsc\zero_shot_atsc\nli_experiments\results_nli_few_shot_in_domain_restaurants\*")
print(len(file_names))


line_start = "{'accuracy':"
dic_list = []

for f_name in file_names:

    file = open(f_name, 'r')
    lines = file.readlines()
    metrics=[]
    for line in lines:
        if line_start in line:
            #print(line)
            metrics.append(line)
    dic = ast.literal_eval(metrics[0].strip()[1:-3])
    dic_list.append(dic)
    #print(dic)
    #print(dic['accuracy'])
    #print(os.path.basename(f_name) + metrics[0])

for i in range(15):
    if(i%5==0):
        print()
        print()
        print()
    experiment = dic_list[(i*5):((i+1)*5)]
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

    results = {'acc_mean' : acc_mean, 'acc_se': acc_se, 'f1s_mean': f1_mean, 'f1_se': f1_se}
    print(file_names[i*5])
    print(results)