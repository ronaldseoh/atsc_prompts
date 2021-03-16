import os
import datasets
import json

#This dataset code format has been converted from the HuggingFace SNLI dataset (https://github.com/huggingface/datasets/blob/master/datasets/snli/snli.py)

_CITATION = """\
TODO YELP CITATION
"""

_DESCRIPTION = """\
This is a subset of the Yelp Open Dataset (https://www.yelp.com/dataset). It includes only the review data and it has been filtered to be exclusively restraunt reviews.
Thre is only a train split because this is HuggingFace Dataset is intended to be used as an unlabeled corpus for pretraining.
"""

_DATA_URL = "TODO"


class YelpRestaurants(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text import of SNLI",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "stars": datasets.Value("double")
                }
            ),
            # No default supervised_keys (as we have to pass both premise
            # and hypothesis as input).
            supervised_keys=None,
            homepage="TODO",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        
        urls_to_download = {
            "train": self.config.data_files['train'],
            "restaurant_ids": self.config.data_files['restaurant_ids']
        }

        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        # There is no predefined train/val/test split for this dataset.
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "restaurant_ids_path": downloaded_files["restaurant_ids"],
                    "split": "train"}),
        ]


    def _generate_examples(self, filepath, restaurant_ids_path, split):

        restaurantIDs = set(line.strip() for line in open(restaurant_ids_path))

        with open(filepath, encoding="utf-8") as f:
            for line in f:
                review = json.loads(line)

                if review["business_id"] not in restaurantIDs:
                    continue
                elif len(review["text"]) == 0:
                    continue
                else:
                    yield review["review_id"], {
                        "id": review["review_id"],
                        "text": review["text"],
                        "stars": review["stars"]
                    }
