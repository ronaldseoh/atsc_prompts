import os
import itertools
import json

import tqdm
import numpy as np
import pandas as pd


# experiment id prefix
experiment_id_prefix = 'bert_prompt_logit_softmax_atsc'

# Random seed
random_seeds = [696, 685, 683, 682, 589]

# path to pretrained MLM model folder or the string "bert-base-uncased"
lm_model_paths = {
    'bert_amazon_electronics': '../progress/lm_further_pretraining_bert_amazon_electronics_bseoh_2021-03-06--18_59_53/results/checkpoint-1180388',
    'bert-base-uncased': 'bert-base-uncased'
}

# Prompts to be added to the end of each review text
# Note: pseudo-labels for each prompt should be given in the order of (positive), (negative), (neutral)
sentiment_prompts = {
    'i_felt': {"prompt": "I felt the {aspect} was [MASK].", "labels": ["good", "bad", "ok"]},
    'i_like': {"prompt": "I [MASK] the {aspect}.", "labels": ["like", "dislike", "ignore"]},
    'made_me_feel': {"prompt": "The {aspect} made me feel [MASK].", "labels": ["good", "bad", "indifferent"]},
    'the_aspect_is': {"prompt": "The {aspect} is [MASK].", "labels": ["good", "bad", "ok"]}
}

run_single_prompt = True
run_multiple_prompts = True

# Training settings
training_domain = 'restaurants' # 'laptops', 'restaurants', 'joint'

# Test settings
testing_batch_size = 32
testing_domain = 'laptops'

if testing_domain != training_domain:
    cross_domain = True
else:
    cross_domain = False
    
experiment_id_prefix_complete = []
    
if run_single_prompt:    
    for config in tqdm.tqdm(itertools.product(lm_model_paths.keys(), sentiment_prompts.keys())):
        
        lm_model_name, prompt_key = config
        
        # We will use the following string ID to identify this particular (training) experiments
        # in directory paths and other settings
        experiment_id_config = experiment_id_prefix + '_'
        experiment_id_config = experiment_id_config + testing_domain + '_'
        
        if cross_domain:
            experiment_id_config = experiment_id_config + 'cross_domain_'

        experiment_id_config = experiment_id_config + lm_model_name + '_'
        experiment_id_config = experiment_id_config + 'single_prompt' + '_'
        experiment_id_config = experiment_id_config + prompt_key + '_'
        
        experiment_id_prefix_complete.append(experiment_id_config)

if run_multiple_prompts:
    for config in tqdm.tqdm(itertools.product(lm_model_paths.keys())):
        
        lm_model_name = config[0]
        
        # We will use the following string ID to identify this particular (training) experiments
        # in directory paths and other settings
        experiment_id_config = experiment_id_prefix + '_'
        experiment_id_config = experiment_id_config + testing_domain + '_'
        
        if cross_domain:
            experiment_id_config = experiment_id_config + 'cross_domain_'

        experiment_id_config = experiment_id_config + lm_model_name + '_'
        experiment_id_config = experiment_id_config + 'multiple_prompts' + '_'
        
        experiment_id_prefix_complete.append(experiment_id_config)
    

for prefix in experiment_id_prefix_complete:
    
    test_metrics_all = []

    for seed in random_seeds:
        
        experiment_id = prefix + str(seed)
        
        # trained_models_prompts
        test_metrics_ = os.path.join('..', 'trained_models_prompts', experiment_id)
        
        # Load test scores file
        test_metrics = json.load(open(os.path.join('..', 'trained_models_prompts', experiment_id, 'test_metrics.json'), 'r'))
        
        test_metrics_all.append(test_metrics)

    test_metrics_all = pd.DataFrame(test_metrics_all)
    
    print(prefix)

    print(test_metrics_all)

    accuracy_mean = test_metrics_all.accuracy.mean()

    accuracy_se = test_metrics_all.accuracy.std() / np.sqrt(len(random_seeds))

    f1_mean = test_metrics_all.f1.mean()
    f1_se = test_metrics_all.f1.std() / np.sqrt(len(random_seeds))

    print("Accuracy mean: ", accuracy_mean)
    print("Accuracy se: ", accuracy_se)

    print("F1 mean: ", f1_mean)
    print("F1 se: ", f1_se)
    
    print()
