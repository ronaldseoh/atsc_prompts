import os
import itertools

import papermill
import tqdm


# experiment id prefix
experiment_id_prefix = 'gpt2_prompt_logit_softmax_atsc'

# Random seed
random_seeds = [696, 685, 683, 682, 589]

# path to pretrained CLM model folder or the string "gpt2"
lm_model_paths = {
    'gpt2_amazon_electronics': '/mnt/nfs/scratch1/mtak/checkpoint-1180388',
    'gpt2': 'gpt2'
}

# Prompts to be added to the end of each review text
# Note: pseudo-labels for each prompt should be given in the order of (positive), (negative), (neutral)
sentiment_prompts = {
    'i_felt': {"prompt": "I felt the {aspect} was ", "labels": ["good", "bad", "ok"]},
    'made_me_feel': {"prompt": "The {aspect} made me feel ", "labels": ["good", "bad", "indifferent"]},
    'the_aspect_is': {"prompt": "The {aspect} is ", "labels": ["good", "bad", "ok"]}
}

run_single_prompt = True
run_multiple_prompts = False

# Training settings
training_domain = 'restaurants' # 'laptops', 'restaurants', 'joint'

# Test settings
testing_batch_size = 12
testing_domain = 'laptops'

if testing_domain != training_domain:
    cross_domain = True
else:
    cross_domain = False

# Results directory path
results_path = 'results_' + experiment_id_prefix + '_' + testing_domain
os.makedirs(results_path, exist_ok=True)

# Run single prompt experiments first
if run_single_prompt:
    print("Running single prompts...")
    print()

    for seed, lm_model_name, prompt_key in tqdm.tqdm(itertools.product(random_seeds, lm_model_paths.keys(), sentiment_prompts.keys())):
        
        # We will use the following string ID to identify this particular (training) experiments
        # in directory paths and other settings
        experiment_id = experiment_id_prefix + '_'
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
            'testing_batch_size': testing_batch_size,
            'testing_domain': testing_domain
        }

        papermill.execute_notebook(
           experiment_id_prefix + '.ipynb',
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
        experiment_id = experiment_id_prefix + '_'
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
            'testing_batch_size': testing_batch_size,
            'testing_domain': testing_domain
        }

        papermill.execute_notebook(
           experiment_id_prefix + '.ipynb',
           os.path.join(results_path, experiment_id + '.ipynb'),
           parameters=parameters_to_inject,
           log_output=True,
           progress_bar=True,
           autosave_cell_every=300  
        )