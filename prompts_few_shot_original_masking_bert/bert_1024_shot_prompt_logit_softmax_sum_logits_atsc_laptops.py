import os
import itertools

import papermill
import tqdm


# experiment id prefix
experiment_id_prefix = 'bert_prompt_logit_softmax_atsc'

# Random seed
random_seeds = [696, 685, 683, 682, 589]

# path to pretrained MLM model folder or the string "bert-base-uncased"
lm_model_paths = {
    'bert_amazon_om': '../trained_models/lm_further_pretraining_bert_amazon_electronics_original_masking_bseoh_2021-05-08--21_44_30',
    #'bert-base-uncased': 'bert-base-uncased'
}

# Prompts to be added to the end of each review text
# Note: pseudo-labels for each prompt should be given in the order of (positive), (negative), (neutral)
sentiment_prompts = {
    'i_felt': {"prompt": "I felt the {aspect} was [MASK].", "labels": ["good", "bad", "ok"]},
    'i_like': {"prompt": "I [MASK] the {aspect}.", "labels": ["like", "dislike", "ignore"]},
    'made_me_feel': {"prompt": "The {aspect} made me feel [MASK].", "labels": ["good", "bad", "indifferent"]},
    'the_aspect_is': {"prompt": "The {aspect} is [MASK].", "labels": ["good", "bad", "ok"]}
}

run_single_prompt = False
run_multiple_prompts = True

prompts_merge_behavior = 'sum_logits'
prompts_perturb = False

# Training settings
training_domain = 'laptops' # 'laptops', 'restaurants', 'joint'

# Few-shot dataset size
training_dataset_few_shot_size = 1024

experiment_id_prefix_override = 'bert_' + str(training_dataset_few_shot_size) + '_shot_' + 'prompt_logit_softmax_sum_logits_atsc'

# Test settings
testing_batch_size = 32
testing_domain = 'laptops'

if testing_domain != training_domain:
    cross_domain = True
else:
    cross_domain = False

# Results directory path
results_path = 'results_' + experiment_id_prefix_override + '_' + testing_domain
os.makedirs(results_path, exist_ok=True)

# Run single prompt experiments first
if run_single_prompt:
    print("Running single prompts...")
    print()

    for seed, lm_model_name, prompt_key in tqdm.tqdm(itertools.product(random_seeds, lm_model_paths.keys(), sentiment_prompts.keys())):
        
        # We will use the following string ID to identify this particular (training) experiments
        # in directory paths and other settings
        experiment_id = experiment_id_prefix_override + '_'
        experiment_id = experiment_id + testing_domain + '_'
        
        if cross_domain:
            experiment_id = experiment_id + 'cross_domain_'

        experiment_id = experiment_id + lm_model_name + '_'
        experiment_id = experiment_id + 'single_prompt' + '_'
        experiment_id = experiment_id + prompt_key + '_'
        experiment_id = experiment_id + str(seed)
        
        print("Running experiment", experiment_id)
        print()
        
        parameters_to_inject = {
            'experiment_id': experiment_id,
            'random_seed': seed,
            'lm_model_path': lm_model_paths[lm_model_name],
            'training_domain': training_domain,
            'sentiment_prompts': [sentiment_prompts[prompt_key]],
            'training_dataset_few_shot_size': training_dataset_few_shot_size,
            'testing_batch_size': testing_batch_size,
            'testing_domain': testing_domain,
            'prompts_merge_behavior': prompts_merge_behavior,
            'prompts_perturb': prompts_perturb
        }

        papermill.execute_notebook(
            os.path.join('..', 'prompts_supervised_bert', experiment_id_prefix + '.ipynb'),
            os.path.join(results_path, experiment_id + '.ipynb'),
            parameters=parameters_to_inject,
            log_output=True,
            progress_bar=True,
            autosave_cell_every=300  
        )

# Run multiple prompts
if run_multiple_prompts:
    print("Running multiple prompts...")
    print()

    for seed, lm_model_name in tqdm.tqdm(itertools.product(random_seeds, lm_model_paths.keys())):
        
        # We will use the following string ID to identify this particular (training) experiments
        # in directory paths and other settings
        experiment_id = experiment_id_prefix_override + '_'
        experiment_id = experiment_id + testing_domain + '_'
        
        if cross_domain:
            experiment_id = experiment_id + 'cross_domain_'

        experiment_id = experiment_id + lm_model_name + '_'
        experiment_id = experiment_id + 'multiple_prompts' + '_'
        experiment_id = experiment_id + str(seed)
        
        print("Running experiment ", experiment_id)
        print()
        
        parameters_to_inject = {
            'experiment_id': experiment_id,
            'random_seed': seed,
            'lm_model_path': lm_model_paths[lm_model_name],
            'sentiment_prompts': [sentiment_prompts[prompt_key] for prompt_key in sentiment_prompts.keys()],
            'training_domain': training_domain,
            'training_dataset_few_shot_size': training_dataset_few_shot_size,
            'testing_batch_size': testing_batch_size,
            'testing_domain': testing_domain,
            'prompts_merge_behavior': prompts_merge_behavior,
            'prompts_perturb': prompts_perturb
        }

        papermill.execute_notebook(
            os.path.join('..', 'prompts_supervised_bert', experiment_id_prefix + '.ipynb'),
            os.path.join(results_path, experiment_id + '.ipynb'),
            parameters=parameters_to_inject,
            log_output=True,
            progress_bar=True,
            autosave_cell_every=300  
        )