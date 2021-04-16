import torch


class MultiPromptLogitSentimentClassificationHead(torch.nn.Module):
    def __init__(self, lm, num_class, num_prompts, pseudo_label_words, target_token_id=-1):
        super(MultiPromptLogitSentimentClassificationHead, self).__init__()

        self.num_class = num_class
        self.pseudo_label_words = pseudo_label_words
        self.target_token_id = target_token_id
        self.num_prompts = num_prompts

        self.lm = lm
        
        # Is self.lm BERT or GPT-2?
        if self.lm.config.architectures[0].startswith('Bert'):
            # if self.lm is BERT, then mask_token_id should be specified
            assert self.target_token_id != -1
            self.lm_type = 'bert'
        elif self.lm.config.architectures[0].startswith('GPT2'):
            self.lm_type = 'gpt2'
        else:
            raise Exception('Unsupported language model type.')
            
        print("Detected LM type:", self.lm_type)

    def forward(self, reviews_and_prompts):

        # Figure out where the mask token was placed
        if self.lm_type == 'bert':
            # For BERT, we need to find the token in each input with [MASK]
            target_indexes = torch.nonzero(
                reviews_and_prompts.data["input_ids"] == self.target_token_id)[:, 1]

            lm_outputs = self.lm(**reviews_and_prompts)

            real_batch_size = len(reviews_and_prompts.data["input_ids"]) // self.num_prompts

        elif self.lm_type == 'gpt2':
            lm_outputs = []
            target_indexes = []

            # For GPT-2, we need to find the spot right after the input text
            n = reviews_and_prompts.data["input_ids"].shape[0]
            t = torch.tensor([reviews_and_prompts.data["input_ids"].shape[1]-1])
            target_indexes = torch.cat(n*[t])
            lm_outputs = self.lm(**reviews_and_prompts)
            real_batch_size = len(reviews_and_prompts.data["input_ids"]) // self.num_prompts

        outputs = []
                
        for i in range(real_batch_size):
            scores_batch = []

            for j in range(self.num_prompts):
                # logit output assigned to self.pseudo_label_words
                logits = lm_outputs.logits[i+real_batch_size*j, target_indexes[i+real_batch_size*j], self.pseudo_label_words[j]]       

                scores_batch.append(logits)
                
            # Sum up the scores across rows
            scores_batch = torch.stack(scores_batch, dim=0)
            scores_batch = torch.sum(scores_batch, dim=0)
            
            outputs.append(scores_batch)

        outputs = torch.stack(outputs, dim=0)
            
        return outputs

        
        
class SinglePromptLogitSentimentClassificationHead(MultiPromptLogitSentimentClassificationHead):
    def __init__(self, lm, num_class, pseudo_label_words, target_token_id=-1):
        
        super().__init__(lm, num_class, 1, [pseudo_label_words], target_token_id)


class MultiPromptSentimentClassificationHead(torch.nn.Module):
    def __init__(self, lm, num_class, num_prompts, target_token_id=-1):
        super(MultiPromptSentimentClassificationHead, self).__init__()

        self.num_class = num_class
        self.num_prompts = num_prompts
        self.target_token_id = target_token_id

        self.lm = lm
        
        # Is self.lm BERT or GPT-2?
        if self.lm.config.architectures[0].startswith('Bert'):
            # if self.lm is BERT, then mask_token_id should be specified
            assert self.target_token_id != -1
            self.lm_type = 'bert'
        elif self.lm.config.architectures[0].startswith('GPT2'):
            self.lm_type = 'gpt2'
        else:
            raise Exception('Unsupported language model type.')

        print("Detected LM type:", self.lm_type)

        # Linear layer
        self.linear = torch.nn.Linear(
            self.num_prompts * self.lm.config.hidden_size, self.num_class)

    def forward(self, reviews_and_prompts):

        # Extract hidden states and feed them to self.linear
        outputs = []

        lr_inputs_batch = []

        # Figure out where the mask token was placed
        if self.lm_type == 'bert':
            # For BERT, we need to find the token in each input with [MASK]
            target_indexes = torch.nonzero(
                reviews_and_prompts.data["input_ids"] == self.target_token_id)[:, 1]

            lm_outputs = self.lm(**reviews_and_prompts, output_hidden_states=True)

            real_batch_size = len(reviews_and_prompts.data["input_ids"]) // self.num_prompts

        elif self.lm_type == 'gpt2':
            lm_outputs = []
            target_indexes = []

            # For GPT-2, we need to find the spot right after the input text
            n = reviews_and_prompts.data["input_ids"].shape[0]
            t = torch.tensor([reviews_and_prompts.data["input_ids"].shape[1]-1])
            target_indexes = torch.cat(n*[t])
            lm_outputs = self.lm(**reviews_and_prompts)
            real_batch_size = len(reviews_and_prompts.data["input_ids"]) // self.num_prompts
                
        for i in range(real_batch_size):
            # Create an input to self.linear by
            # concatenating last hidden states for this review
            lr_input = []

            for j in range(self.num_prompts):
                lr_input.append(lm_outputs["hidden_states"][-1][i+real_batch_size*j][target_indexes[i+real_batch_size*j]])
              
            lr_input = torch.cat(lr_input, dim=0)

            lr_inputs_batch.append(lr_input)

        lr_inputs_batch = torch.stack(lr_inputs_batch)

        outputs = self.linear(lr_inputs_batch)

        return outputs


class NoPromptSentimentClassificationHead(torch.nn.Module):
    def __init__(self, lm, num_class):
        super(NoPromptSentimentClassificationHead, self).__init__()

        self.num_class = num_class

        self.lm = lm

        self.linear = torch.nn.Linear(
            self.lm.config.hidden_size, self.num_class)

    def forward(self, reviews_and_prompts):

        lm_outputs = self.lm(**reviews_and_prompts, output_hidden_states=True)

        # Last hidden state for [CLS] token
        last_hidden_state_cls = lm_outputs["hidden_states"][-1][:, 0, :]
        
        outputs = self.linear(last_hidden_state_cls)

        return outputs
