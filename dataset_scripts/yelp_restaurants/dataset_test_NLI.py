import datasets

yelp = datasets.load_dataset(
    './dataset_scripts/yelp_restaurants/yelp_NLI.py',
    data_files={
        'train': 'dataset_files/yelp_restaurants/yelp_academic_dataset_review.json',
        'restaurant_ids': 'dataset_files/yelp_restaurants/restaurantIDs.txt',
        'pos_sentiment': 'dataset_files/opinion_lexicon/positive-words.txt',
        'neg_sentiment': 'dataset_files/opinion_lexicon/negative-words.txt'
    },
    cache_dir='./dataset_cache')

print(yelp["train"][0])
