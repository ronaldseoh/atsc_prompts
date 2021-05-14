# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
# Copyright 2020 Srimoyee Bhattacharyya, Zachary Harrison, Ronald Seoh
# Copyright 2021 Ian Birle, Ronald Seoh, Mrinal Tak
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""

from __future__ import absolute_import, division, print_function

import json
import os
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import datasets


# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
authors={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care. 
"""

@dataclass
class SemEval2014Task4Config(datasets.BuilderConfig):
    """ BuilderConfig for NewDataset"""

    remove_conflicting: bool = True
    use_aspect_categories: bool = False

class SemEval2014Task4Dataset(datasets.GeneratorBasedBuilder):
    """Sentihood dataset main class."""

    VERSION = datasets.Version("0.0.1")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.
    BUILDER_CONFIG_CLASS = SemEval2014Task4Config
    BUILDER_CONFIGS = [
        # Default configuration should have the same 'name' as the dataset
        SemEval2014Task4Config(
            name="SemEval2014Task4Dataset", version=VERSION, remove_conflicting=True, use_aspect_categories=False, description="SemEval 2014 Task 4 (Subtask 2: Aspect term polarity) Dataset - Rows with conflict labels removed."),
        SemEval2014Task4Config(
            name="SemEval2014Task4Dataset - Subtask 4", version=VERSION, remove_conflicting=True, use_aspect_categories=True, description="SemEval 2014 Task 4 (Subtask 4: Aspect category polarity) Dataset - Rows with conflict labels removed."),
    ]

    def _info(self):
        # TODO: Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "aspect": datasets.Value("string"),
                    "sentiment": datasets.features.ClassLabel(names=["positive", "negative", "neutral", "conflict"])
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://huggingface.co/great-new-dataset",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": self.config.data_files['train'],
            "test": self.config.data_files['test'],
        }

        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"], "split": "train"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"], "split": "test"})
        ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. """
        # Based on the codes from
        # https://github.com/deepopinion/domain-adapted-atsc/blob/master/utils.py
        with open(filepath, encoding='utf-8') as semeval_file:
            sentence_elements = ET.parse(semeval_file).getroot().iter('sentence')

        for id_, s in enumerate(sentence_elements):

            sentence_text = s.find('text').text
            aspect_term_sentiment = []
            
            if self.config.use_aspect_categories:
                aspect_iterator = s.iter('aspectCategory')
                term_key = 'category'
            else:
                aspect_iterator = s.iter('aspectTerm')
                term_key = 'term'

            for o in aspect_iterator:
                aspect_term = o.get(term_key)
                sentiment = o.get('polarity')

                # Deal with "conflict" labels
                if sentiment != 'conflict':
                    aspect_term_sentiment.append(
                        {
                            'sentiment': sentiment,
                            'aspect': aspect_term})
                else:
                    # Skip this opinion
                    if self.config.remove_conflicting:
                        pass
                    else:
                        aspect_term_sentiment.append((aspect_term, sentiment))

            if len(aspect_term_sentiment) > 0:
                for ats in aspect_term_sentiment:
                    yield id_, {
                        "text": sentence_text,
                        "aspect": ats["aspect"],
                        "sentiment": ats["sentiment"]
                    }
