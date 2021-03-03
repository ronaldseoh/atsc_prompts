from datasets import load_dataset

dataset = load_dataset('load_yelp.py', data_files='yelp_academic_dataset_review.json')["train"]

print(dataset[0])
