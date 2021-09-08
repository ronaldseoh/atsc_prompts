# atsc_prompts

This repository contains the codes used to produce the results from our paper [*Open Aspect Target Sentiment Classification with Natural Language Prompts*](https://link.iamblogger.net/atscprompts-paper).

Before executing any of the codes in the repo, please set up an [Anaconda environment](https://link.iamblogger.net/2to7k) on your system by running

```bash
conda env create -f environment.yml
```

Although we also listed core dependencies in `requirements.txt` that you could feed into `pip`, we recommend using Anaconda and our [`environment.yml`](https://link.iamblogger.net/y4a1b) as there might be dependencies that are not fully fulfilled by [`requirements.txt`](https://link.iamblogger.net/yxicy). 

If you want to try training your own prompt-based ATSC model, you will probably want to start with the directories with the prefix `prompts_supervised_`. Please refer to the descriptions below for what each directory in this repo is about:

- `pretraining` contains the Jupyter notebook files with the training loop for further pretraining BERT/GPT-2 LMs. As described in our paper, we modify the pretraining objective for BERT, which is implemented as a thing called "data collator" compatible with the Huggingface library (see utils/data_collator_smart_mlm.py.)

- `prompts_supervised_*` directories contain the notebook files for full-shot/few-shot training. I recommend skimming these notebooks first. PyTorch modules for converting LM/NLI outputs to ATSC prediction are defined in [`utils/prompt_output_head.py`](https://link.iamblogger.net/jymqn).

- `prompts_zero_shot_*` directories contain the notebook files for testing zero-shot cases.

- `prompts_subtask4_*` directories contain the notebook files for testing ATSC models on aspect category sentiment classification (ACSC) without additional training.

## Citation

If you are using our code for your paper, please cite our paper using the following BibTeX entry:

```
@inproceedings{seoh2021emnlp,
  title={Open Aspect Target Sentiment Classification with Natural Language Prompts},
  author={Ronald Seoh, Ian Birle, Mrinal Tak, Haw-Shiuan Chang, Brian Pinette, Alfred Hough},
  booktitle={EMNLP},
  year={2021},
}
```

## License

`atsc_prompts` is licensed under the Apache 2.0 license. Please check `LICENSE`.
