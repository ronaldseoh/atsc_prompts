#!/usr/bin/env python
# coding: utf-8

# # Initial Setups
# 

# In[1]:


#import os
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '59994'
#os.environ['RANK'] = "0"
#os.environ['LOCAL_RANK'] = "0"
#os.environ['WORLD_SIZE'] = "1"


# ## (Google Colab use only)

# In[2]:


# Use Google Colab
use_colab = True

# Is this notebook running on Colab?
# If so, then google.colab package (github.com/googlecolab/colabtools)
# should be available in this environment

# Previous version used importlib, but we could do the same thing with
# just attempting to import google.colab
try:
    from google.colab import drive
    colab_available = True
except:
    colab_available = False

if use_colab and colab_available:
    drive.mount('/content/drive')

    # cd to the appropriate working directory under my Google Drive
    get_ipython().run_line_magic('cd', "'drive/My Drive/cs696ds_lexalytics/Language Model Finetuning'")
    
    # Install packages specified in requirements
    get_ipython().system('pip install -r requirements.txt')
    
    # List the directory contents
    get_ipython().system('ls')


# ## Experiment Parameters
# 
# **NOTE**: The following `experiment_id` MUST BE CHANGED in order to avoid overwriting the files from other experiments!!!!!!
# 
# **NOTE 2**: The values for the variables in the cell below can be overridden by `papermill` at runtime. Variables in other cells cannot be changed in this manner.

# In[3]:


# We will use the following string ID to identify this particular (training) experiments
# in directory paths and other settings
experiment_id = 'lm_further_pretraining_gpt-2_amazon_electronics_deepspeed_test'

# Random seed
random_seed = 696

# Dataset size related
total_subset_proportion = 1.0 # Do we want to use the entirety of the training set, or some parts of it?
validation_dataset_proportion = 0.1 # Proportion to be reserved for validation (after selecting random subset with total_subset_proportion)

# Training hyperparameters
num_train_epochs = 20 # Number of epochs
per_device_train_batch_size = 16 # training batch size PER COMPUTE DEVICE
per_device_eval_batch_size = 16 # evaluation batch size PER COMPUTE DEVICE
learning_rate = 1e-5
weight_decay = 0.01


# ## Package Imports

# In[4]:


import sys
import os
import random

import numpy as np
import torch
import transformers
import datasets

import utils

# Random seed settings
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Print version information
print("Python version: " + sys.version)
print("NumPy version: " + np.__version__)
print("PyTorch version: " + torch.__version__)
print("Transformers version: " + transformers.__version__)


# ## PyTorch GPU settings

# In[5]:


if torch.cuda.is_available():

    torch_device = torch.device('cuda')

    # Set this to True to make your output immediately reproducible
    # Note: https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = False
    
    # Disable 'benchmark' mode: Set this False if you want to measure running times more fairly
    # Note: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.backends.cudnn.benchmark = True
    
    # Faster Host to GPU copies with page-locked memory
    use_pin_memory = True
    
    # Number of compute devices to be used for training
    training_device_count = torch.cuda.device_count()

    # CUDA libraries version information
    print("CUDA Version: " + str(torch.version.cuda))
    print("cuDNN Version: " + str(torch.backends.cudnn.version()))
    print("CUDA Device Name: " + str(torch.cuda.get_device_name()))
    print("CUDA Capabilities: "+ str(torch.cuda.get_device_capability()))
    print("Number of CUDA devices: "+ str(training_device_count))
    
else:
    torch_device = torch.device('cpu')
    use_pin_memory = False
    
    # Number of compute devices to be used for training
    training_device_count = 1

print()
print("PyTorch device selected:", torch_device)


# # Further pre-training

# ## Load the GPT-2 model

# In[6]:


tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", cache_dir='./gpt2_cache')

# https://github.com/huggingface/transformers/issues/8452
tokenizer.pad_token = tokenizer.eos_token

model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", cache_dir='./gpt2_cache')


# ## Load the Amazon electronics dataset

# In[7]:


amazon = datasets.load_dataset(
    './dataset_scripts/amazon_ucsd_reviews',
    data_files={
        'train': 'dataset_files/amazon_ucsd_reviews/Electronics.json.gz',
    },
    cache_dir='./dataset_cache',
    ignore_verifications=True)


# In[8]:


data_train = amazon['train']
print("Number of training data (original):", len(data_train))


# In[9]:


data_train_selected = data_train.shuffle(seed=random_seed).select(np.arange(0, int(len(data_train) * total_subset_proportion)))
print("Number of training data (subset):", len(data_train_selected))


# In[10]:


# Check out how individual data points look like
print(data_train_selected[0])


# ### Preprocessing: Encode the text with Tokenizer

# In[11]:


train_dataset = data_train_selected.map(
    lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=256),
    remove_columns=data_train_selected.column_names,
    batched=True, num_proc=8,
    load_from_cache_file=True)


# ### Train-validation split

# In[12]:


# Training set size after validation split
new_train_dataset_size = int(len(train_dataset) * (1 - validation_dataset_proportion))
new_valid_dataset_size = len(train_dataset) - new_train_dataset_size

new_train_dataset = train_dataset.select(indices=np.arange(new_train_dataset_size))
new_valid_dataset = train_dataset.select(indices=np.arange(new_train_dataset_size, new_train_dataset_size + new_valid_dataset_size))


# In[13]:


print("Training dataset after split:", len(new_train_dataset))
print("Validation dataset after split:", len(new_valid_dataset))


# ## Pre-train further

# ### Training settings

# In[14]:


# CLM
collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# In[15]:


# How many training steps would we have?
approx_total_training_steps = len(new_train_dataset) // (per_device_train_batch_size * training_device_count) * num_train_epochs

print("There will be approximately %d training steps." % approx_total_training_steps)

# Let's have warmups for the first 1% of steps.
# The BERT paper did 10,000 warmup steps for the 1,000,000 total training steps.
warmup_steps = approx_total_training_steps // 100

print("Warmup steps:", warmup_steps)


# In[16]:


training_args = transformers.TrainingArguments(
    output_dir=os.path.join('.', 'progress', experiment_id, 'results'), # output directory
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,              # total number of training epochs
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    evaluation_strategy='epoch',
    logging_dir=os.path.join('.', 'progress', experiment_id, 'logs'), # directory for storing logs
    logging_first_step=True,
    weight_decay=weight_decay,               # strength of weight decay
    seed=random_seed,
    learning_rate=learning_rate,
    fp16=True,
    fp16_backend='amp',
    fp16_opt_level='O2',
    prediction_loss_only=True,
    load_best_model_at_end=True,
    dataloader_num_workers=training_device_count * 2,
    dataloader_pin_memory=use_pin_memory,
    deepspeed='ds_config.json',
    #local_rank=0,
)


# In[17]:


print(training_args.n_gpu)


# In[18]:


trainer = transformers.Trainer(
    model=model,
    args=training_args,
    data_collator=collator, # do the masking on the go
    train_dataset=new_train_dataset,
    eval_dataset=new_valid_dataset,
)


# ### Training loop

# In[19]:


#get_ipython().run_cell_magic('time', '', 'trainer.train()')
trainer.train()

# ### Save the model to the local directory

# In[ ]:


trainer.save_model(os.path.join('.', 'trained_models', experiment_id))


# In[ ]:


tokenizer.save_pretrained(os.path.join('.', 'trained_models', experiment_id))


# ## LM Evaluation

# In[ ]:


eval_results = trainer.evaluate()


# In[ ]:


print(eval_results)

perplexity = np.exp(eval_results["eval_loss"])

print(perplexity)


# ## Playing with my own input sentences

# In[ ]:


example = f"""The {tokenizer.mask_token} of {tokenizer.mask_token} is awful, but its {tokenizer.mask_token} is fantastic."""

example_encoded = tokenizer.encode(example, add_special_tokens=True, return_tensors="pt").to(torch_device)

# Let's decode this back just to see how they were actually encoded
example_tokens = []

for id in example_encoded[0]:
    example_tokens.append(tokenizer.convert_ids_to_tokens(id.item()))

print(example_tokens)


# In[ ]:


example_prediction = model(example_encoded)

example_prediction_argmax = torch.argmax(example_prediction[0], dim=-1)[0]

print(example_prediction_argmax)

print(tokenizer.decode(example_prediction_argmax))


# In[ ]:





