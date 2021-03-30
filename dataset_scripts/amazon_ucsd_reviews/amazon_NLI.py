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
This Dataset is an updated version of the Amazon review dataset released in 2014. As in the previous version, this dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs). In addition, this version provides the following features:

More reviews:
The total number of reviews is 233.1 million (142.8 million in 2014).
Newer reviews:
Current data includes reviews in the range May 1996 - Oct 2018.
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

class AmazonUCSDReviews(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="AmazonUCSDReviews",
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
                    "pos_sentiment_path": downloaded_files['pos_sentiment'],
                    "neg_sentiment_path": downloaded_files['neg_sentiment'],
                    "split": "train"}),
        ]


    def _generate_examples(self, filepath, restaurant_ids_path, pos_sentiment_path, neg_sentiment_path, split):

        with open(pos_sentiment_path, "r", encoding='utf-8') as positive_words_file:
            pos_words = positive_words_file.readlines()
            
        pos_words = pos_words[30:] # The list starts from line 31
        pos_words = [word.rstrip() for word in pos_words]
            
        with open(neg_sentiment_path, "r", encoding="iso-8859-1") as negative_words_file:
            neg_words = negative_words_file.readlines()
            
        neg_words = neg_words[31:] # The list starts from line 32
        neg_words = [word.rstrip() for word in neg_words]

        with open(filepath, encoding='utf-8') as f:

            prev_sentences = []

            for line in f:
                review = json.loads(line)

                #Extract a list of sentences from text
                sentences = get_sentences(review["reviewText"])

                #NLI requires two sentences
                if len(sentences) < 2:
                    continue
                
                for i in range(len(sentences)):
                    sentence = sentences[i]

                    # We only want to look at sentences that have a high polarity word in them
                    sentence_words = get_words(sentence)

                    if any(word in sentence_words for word in pos_words) or any(word in sentence_words for word in neg_words):

                        # The hypothesis is the sentence with a high polarity word
                        hypothesis = sentence

                        # The premise is the rest of the review text
                        premise = " ".join(sentences[0:i]) + " ".join(sentences[i+1:])

                        # Entailment with the current review text
                        yield i, {
                            "premise": premise,
                            "hypothesis": hypothesis,
                            "label": 0
                        }

                        # Neutral with the premise being the first sentence of the previous review
                        if prev_sentences != []:
                            yield i, {
                                "premise": prev_sentences[0],
                                "hypothesis": hypothesis,
                                "label": 1
                            }

                prev_sentences = sentences