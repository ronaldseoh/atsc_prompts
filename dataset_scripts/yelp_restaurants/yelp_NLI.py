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

def get_sentences(document):
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

    return real_sents

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
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "label": datasets.Value("string")
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
            "restaurant_ids": self.config.data_files['restaurant_ids'],
            "pos_sentiment": self.config.data_files['pos_sentiment'],
            "neg_sentiment": self.config.data_files['neg_sentiment']
        }

        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        # There is no predefined train/val/test split for this dataset.
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "restaurant_ids_path": downloaded_files["restaurant_ids"],
                    "pos_sentiment_path": downloaded_files['pos_sentiment'],
                    "neg_sentiment_path": downloaded_files['neg_sentiment'],
                    "split": "train"}),
        ]


    def _generate_examples(self, filepath, restaurant_ids_path, pos_sentiment_path, neg_sentiment_path, split):

        restaurantIDs = set(line.strip() for line in open(restaurant_ids_path))

        pos_words = set(line.strip() for line in open(pos_sentiment_path))
        neg_words = set(line.strip() for line in open(neg_sentiment_path))

        with open(filepath, encoding='utf-8') as f:

            prev_sentences = []
            for line in f:
                review = json.loads(line)

                if review["business_id"] not in restaurantIDs:
                    continue
                else:
                    #Extract a list of sentences from text
                    sentences = get_sentences(review["text"])

                    #NLI requires two sentences
                    if len(sentences) < 2:
                        continue
                    
                    for i in range(len(sentences)):
                        sentence = sentences[i]

                        #We only want to look at sentences that have a high polarity word in them
                        if any(word in sentence for word in pos_words) or any(word in sentence for word in neg_words):

                            #The hypothesis is the sentence with a high polarity word
                            hypothesis = sentence

                            #The premise is the rest of the review text
                            premise = " ".join(sentences[0:i]) + " ".join(sentences[i+1:])

                            #Entailment with the current review text
                            yield review["review_id"], {
                                "id": review["review_id"],
                                "premise": premise,
                                "hypothesis": hypothesis,
                                "label": "entailment"
                            }

                            #Neutral with the premise being the first sentence of the prvious review
                            if prev_sentences != []:
                                yield review["review_id"], {
                                    "id": review["review_id"],
                                    "premise": prev_sentences[0],
                                    "hypothesis": hypothesis,
                                    "label": "neutral"
                                }

                    prev_sentences = sentences


                    
                    
