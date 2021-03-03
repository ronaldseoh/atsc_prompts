# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

"""Amazon Customer Reviews Dataset --- US REVIEWS DATASET."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os
import datasets


_CITATION = """\
"""

_DESCRIPTION = """\
This Dataset is an updated version of the Amazon review dataset released in 2014. As in the previous version, this dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs). In addition, this version provides the following features:

More reviews:
The total number of reviews is 233.1 million (142.8 million in 2014).
Newer reviews:
Current data includes reviews in the range May 1996 - Oct 2018.
"""


class AmazonUCSDReviewsConfig(datasets.BuilderConfig):
    """BuilderConfig for AmazonUCSDReviews."""

    def __init__(self, **kwargs):
        """Constructs a AmazonUCSDReviewsConfig.
        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(AmazonUCSDReviewsConfig, self).__init__(version=datasets.Version("0.1.0", ""), **kwargs),


class AmazonUCSDReviews(datasets.GeneratorBasedBuilder):
    """AmazonUCSDReviews dataset."""

    BUILDER_CONFIGS = [
        AmazonUCSDReviewsConfig(  # pylint: disable=g-complex-comprehension
            name="AmazonUCSDReviews",
            description=(
                f"A dataset consisting of reviews of Amazon products in US marketplace."
            ),
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    # "asin": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    # "verified": datasets.Value("string"),
                    # "summary": datasets.Value("string"),
                    # "reviewTime": datasets.Value("string"),
                    # "reviewerName": datasets.Value("string"),
                    # "stars": datasets.Value("string"),
                    # "verified": datasets.Value("string"),
                    # "reviewerID": datasets.Value("string")

                }
            ),
            supervised_keys=None,
            homepage="https://jmcauley.ucsd.edu/data/amazon/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        urls_to_download = {
            "train": self.config.data_files['train'],
        }

        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        # There is no predefined train/val/test split for this dataset.
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"], "split": "train"}),
        ]

    def _generate_examples(self, filepath, split):
        row_count = 0
        err_count = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                review = json.loads(line)
                try:
                    yield i, {
                          "text": review["reviewText"],
                          # "asin": review["asin"],
                          # "stars": review["overall"],
                          # "verified": review["verified"],
                          # "reviewTime": review["reviewTime"],
                          # "reviewerName": review["reviewerName"],
                          # "reviewerID": review["reviewerID"],
                          # "summary": review["summary"]
                      }
                except:
                    err_count += 1
                    continue
                row_count += 1