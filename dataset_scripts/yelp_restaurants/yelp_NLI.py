import os
import datasets
import json
import spacy
from tqdm import tqdm
import regex as re
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
    doc = nlp(document, disable=['parser', 'tagger', 'ner'])

    real_sents = []

    for s in doc.sents:
        s= s.text.strip()
        x = s.replace(' ', '').replace('\n', '')
        if x != '':
            s_sanitized = s.replace('\n', '')
            real_sents.append(s_sanitized)

    return real_sents

def get_words(sentence_string):
    no_punc = re.sub("[^\w\s]", "", sentence_string)
    return no_punc.split()

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
                    "review_id": datasets.Value("string"),
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "label": datasets.Value("int64")
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

        with open(pos_sentiment_path, "r", encoding='utf-8') as positive_words_file:
            pos_words = positive_words_file.readlines()
            
        pos_words = pos_words[30:] # The list starts from line 31
        pos_words = [word.rstrip() for word in pos_words]
            
        with open(neg_sentiment_path, "r", encoding="iso-8859-1") as negative_words_file:
            neg_words = negative_words_file.readlines()
            
        neg_words = neg_words[31:] # The list starts from line 32
        neg_words = [word.rstrip() for word in neg_words]

        with open(filepath, encoding='utf-8') as f:

            prev_premise = ""

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

                        # We only want to look at sentences that have a high polarity word in them
                        sentence_words = get_words(sentence)
                        hold_prev_premise = ""
                        if any(word in sentence_words for word in pos_words) or any(word in sentence_words for word in neg_words):

                            # The hypothesis is the sentence with a high polarity word
                            hypothesis = sentence

                            # The premise is the rest of the review text
                            premise = " ".join(sentences[0:i]) + " ".join(sentences[i+1:])

                            # Entailment with the current review text
                            yield review["review_id"] + "_" + str(i), {
                                "review_id": review["review_id"],
                                "premise": premise,
                                "hypothesis": hypothesis,
                                "label": 0
                            }

                            # Neutral with the premise the previous review without the previous hypothesis
                            if prev_premise != "":
                                yield review["review_id"] + "_" + str(i) + "_neu", {
                                    "review_id": review["review_id"],
                                    "premise": prev_premise,
                                    "hypothesis": hypothesis,
                                    "label": 1
                                }
                            hold_prev_premise = " ".join(sentences[0:i]) + " ".join(sentences[i+1:])
                    prev_premise = hold_prev_premise


                    
                    
