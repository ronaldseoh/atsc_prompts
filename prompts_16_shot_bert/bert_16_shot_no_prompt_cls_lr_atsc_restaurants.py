import os
import itertools

import papermill
import tqdm


# experiment id prefix
experiment_id_prefix = 'bert_no_prompt_cls_lr_atsc'

# Random seed
random_seeds = [696, 685, 683, 682, 589]

# path to pretrained MLM model folder or the string "bert-base-uncased"
lm_model_paths = {
    'bert_yelp_restaurants': '../trained_models/lm_further_pretraining_bert_yelp_restaurants_bseoh_2021-03-22--15_03_31',
    'bert-base-uncased': 'bert-base-uncased'
}

# Training settings
training_domain = 'restaurants' # 'laptops', 'restaurants', 'joint'

# Few-shot dataset size
training_dataset_few_shot_size = 16

experiment_id_prefix_override = 'bert_' + str(training_dataset_few_shot_size) + '_shot_' + 'no_prompt_cls_lr_atsc'

# Test settings
testing_batch_size = 32
testing_domain = 'restaurants'

if testing_domain != training_domain:
    cross_domain = True
else:
    cross_domain = False

# Results directory path
results_path = 'results_' + experiment_id_prefix_override + '_' + testing_domain
os.makedirs(results_path, exist_ok=True)

# Run experiments
print("Running experiments...")
print()

for seed, lm_model_name in tqdm.tqdm(itertools.product(random_seeds, lm_model_paths.keys())):
    
    # We will use the following string ID to identify this particular (training) experiments
    # in directory paths and other settings
    experiment_id = experiment_id_prefix_override + '_'
    experiment_id = experiment_id + testing_domain + '_'
    
    if cross_domain:
        experiment_id = experiment_id + 'cross_domain_'

    experiment_id = experiment_id + lm_model_name + '_'
    experiment_id = experiment_id + str(seed)
    
    print("Running experiment", experiment_id)
    print()
    
    parameters_to_inject = {
        'experiment_id': experiment_id,
        'random_seed': seed,
        'lm_model_path': lm_model_paths[lm_model_name],
        'training_domain': training_domain,
        'training_dataset_few_shot_size': training_dataset_few_shot_size,
        'testing_batch_size': testing_batch_size,
        'testing_domain': testing_domain
    }

    papermill.execute_notebook(
       os.path.join('..', 'prompts_supervised_bert', experiment_id_prefix + '.ipynb'),
       os.path.join(results_path, experiment_id + '.ipynb'),
       parameters=parameters_to_inject,
       log_output=True,
       progress_bar=True,
       autosave_cell_every=300  
    )
