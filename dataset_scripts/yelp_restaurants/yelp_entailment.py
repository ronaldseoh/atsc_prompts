import os
import datasets
import json
import spacy
from tqdm import tqdm
#This dataset code format has been converted from the HuggingFace SNLI dataset (https://github.com/huggingface/datasets/blob/master/datasets/snli/snli.py)

_CITATION = """\
TODO YELP CITATION
"""

_DESCRIPTION = """\
This is a subset of the Yelp Open Dataset (https://www.yelp.com/dataset). It includes only the review data and it has been filtered to be exclusively restraunt reviews.
Thre is only a train split because this is HuggingFace Dataset is intended to be used as an unlabeled corpus for pretraining.
"""

_DATA_URL = "TODO"

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(nlp.create_pipe('sentencizer'))

def get_entailment_pairs(document):
    sentences = []
    counter = 0
    doc = nlp(document, disable=['parser', 'tagger', 'ner'])
    real_sents = []
    for s in doc.sents:
        s= s.text.strip()
        x = s.replace(' ', '').replace('\n', '')
        if x != '':
            s_sanitized = s.replace('\n', '')
            real_sents.append(s_sanitized)
    pairs = []
    for i,_ in enumerate(real_sents[:-1]):
        pairs.append(real_sents[i]+"\n"+real_sents[i+1])
    return pairs

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
                    "stars": datasets.Value("double"),
                    "judgement": datasets.Value("string")
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

        with open(filepath) as f:
            for line in f:
                review = json.loads(line)

                if review["business_id"] not in restaurantIDs:
                    continue
                else:
                    pairs = get_entailment_pairs(review["text"])
                    for pair in pairs:
                        yield review["review_id"], {
                            "id": review["review_id"],
                            "text": pair,
                            "stars": review["stars"],
                            "judgement": "entailment"
                        }