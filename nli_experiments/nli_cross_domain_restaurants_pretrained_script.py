import os
import itertools

import papermill
import tqdm


# experiment id prefix
experiment_id_prefix = 'nli_supervised_cross_domain'

# Random seed
random_seeds = [696, 685, 683, 682, 589]

# path to pretrained MLM model folder or the string "bert-base-uncased"
nli_model_paths = {
    'yelp-pretrained': '../progress/yelp_2_epoch/results/checkpoint-875324'
}

# Prompts to be added to the end of each review text
sentiment_prompts = {
    'the_aspect_is': ["The {aspect} is good.", "The {aspect} is bad."],
    'I_like': ["I like the {aspect}.", "I dislike the {aspect}."],
    'made_me_feel': ["The {aspect} made me feel good.", "The {aspect} made me feel bad."],
    'I_felt': ["I felt the {aspect} was good.", "I felt the {aspect} was bad."],
    'multi_prompt': ["The {aspect} is good.", "The {aspect} is bad.", "I like the {aspect}.", "I dislike the {aspect}.",
                    "The {aspect} made me feel good.", "The {aspect} made me feel bad.", "I felt the {aspect} was good.", "I felt the {aspect} was bad."]
}

pos_prompt_indexes = {
    'the_aspect_is': [0],
    'I_like': [0],
    'made_me_feel': [0],
    'I_felt': [0],
    'multi_prompt': [0, 2, 4, 6]
}

neg_prompt_indexes = {
    'the_aspect_is': [1],
    'I_like': [1],
    'made_me_feel': [1],
    'I_felt': [1],
    'multi_prompt': [1, 3, 5, 7]
}

train_batch_sizes = {
    'the_aspect_is': 8,
    'I_like': 8,
    "made_me_feel": 8,
    "I_felt": 8,
    'multi_prompt': 2
}
# Training settings
training_domain = 'laptops' # 'laptops', 'restaurants', 'joint'
testing_domain = 'restaurants'

# Test settings
testing_batch_size = 8


# Results directory path
results_path = 'results_' + experiment_id_prefix + '_' + testing_domain
os.makedirs(results_path, exist_ok=True)


# Run multiple prompts
print("Running experiments...")
print()

for seed, nli_model_name, prompt_key in tqdm.tqdm(itertools.product(random_seeds, nli_model_paths.keys(), sentiment_prompts.keys())):

    # We will use the following string ID to identify this particular (training) experiments
    # in directory paths and other settings
    experiment_id = experiment_id_prefix + '_'
    experiment_id = experiment_id + testing_domain + '_'

    experiment_id = experiment_id + nli_model_name + '_'
    experiment_id = experiment_id + prompt_key + '_'
    experiment_id = experiment_id + str(seed)

    print("Running experiment ", experiment_id)
    print()

    parameters_to_inject = {
        'experiment_id': experiment_id,
        'random_seed': seed,
        'nli_model_path': nli_model_paths[nli_model_name],
        'sentiment_prompts': sentiment_prompts[prompt_key],
        'pos_prompt_indexes': pos_prompt_indexes[prompt_key],
        'neg_prompt_indexes': neg_prompt_indexes[prompt_key],
        'training_domain': training_domain,
        'testing_batch_size': testing_batch_size,
        'testing_domain': testing_domain,
        'training_batch_size': train_batch_sizes[prompt_key]
    }

    papermill.execute_notebook(
       'nli_experiment_template.ipynb',
       os.path.join(results_path, experiment_id + '.ipynb'),
       parameters=parameters_to_inject,
       log_output=True,
       progress_bar=True,
       autosave_cell_every=300
    )
