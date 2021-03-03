from datasets import load_dataset

dataset_1 = load_dataset('amazon_ucsd_reviews.py','sports')["train"]
dataset_2 = load_dataset('amazon_ucsd_reviews.py','electronics')["train"]

# Sports dataset
print(dataset_1[0])
# Electronics dataset
print(dataset_2[0])