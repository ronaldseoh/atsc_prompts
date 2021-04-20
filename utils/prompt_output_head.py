import torch


class SinglePromptLogitSentimentClassificationHead(torch.nn.Module):
    def __init__(self, lm, num_class, pseudo_label_words, target_token_id=-1):
        super(SinglePromptLogitSentimentClassificationHead, self).__init__()

        self.num_class = num_class
        self.pseudo_label_words = pseudo_label_words
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

    def forward(self, reviews_and_prompts):

        if self.lm_type == 'bert':
            # Figures out where the mask token was placed
            target_indexes = (reviews_and_prompts.data["input_ids"] == self.target_token_id)

            lm_outputs = self.lm(**reviews_and_prompts)

            outputs = lm_outputs.logits[target_indexes]
        
            outputs = outputs[:, self.pseudo_label_words]
            
        elif self.lm_type == 'gpt2':
            
            outputs = []
            
            for example in reviews_and_prompts:
                lm_outputs = self.lm(**example, return_dict=True)
                
                lm_predictions = lm_outputs.logits[0, len(example['input_ids'][0]) - 1, self.pseudo_label_words]
                
                outputs.append(lm_predictions)

            outputs = torch.stack(outputs, dim=0)

        return outputs


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

        # Figures out where the mask token was placed
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
            for example in reviews_and_prompts:
                target_indexes.append(len(example['input_ids'][0]) - 1)

                lm_outputs.append(self.lm(**example, output_hidden_states=True))

            real_batch_size = len(reviews_and_prompts) // self.num_prompts
                
        for i in range(real_batch_size):
            # Create an input to self.linear by
            # concatenating last hidden states for this review
            lr_input = []

            for j in range(self.num_prompts):
                if self.lm_type == 'bert':
                    lr_input.append(lm_outputs["hidden_states"][-1][i+real_batch_size*j][target_indexes[i+real_batch_size*j]])
                elif self.lm_type == 'gpt2':
                    lr_input.append(lm_outputs[i+real_batch_size*j]["hidden_states"][-1][0][target_indexes[i+real_batch_size*j]])
                    
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
    
class NLISentimentClassificationHead(torch.nn.Module):
    def __init__(self, nli_model, num_prompts, pos_prompt_indexes, neg_prompt_indexes):
        super(NLISentimentClassificationHead, self).__init__()
        
        self.num_prompts = num_prompts
        self.nli_model = nli_model
        
        self.pos_prompt_indexes = pos_prompt_indexes
        self.neg_prompt_indexes = neg_prompt_indexes
        
    def forward(self, reviews_and_prompts):
        
        nli_output = self.nli_model(**reviews_and_prompts)["logits"]

        outputs = torch.Tensor().to(self.nli_model.device)

        # Text Attack NLI Labels: 0-> Contradiction, 1-> Entailment, 2-> Neutral
        # Sentiment Polarity Labels: 0-> Positive, 1-> Negative, 2-> Neutral
        for i in range(len(nli_output)//self.num_prompts):
            prompts_batch = nli_output[i*self.num_prompts:(i+1)*self.num_prompts]

            pos_logit = torch.mean(prompts_batch[self.pos_prompt_indexes], dim=0)[1]

            neg_logit = torch.mean(prompts_batch[self.neg_prompt_indexes], dim=0)[1]

            neu_logit = torch.mean(prompts_batch, dim=0)[2]

            pred_logits = torch.stack([pos_logit, neg_logit, neu_logit])
            pred_logits = torch.reshape(pred_logits, (1,-1))

            outputs = torch.cat([outputs, pred_logits])

        return outputs

class NLIMinSentimentClassificationHead(torch.nn.Module):
    def __init__(self, nli_model, num_prompts, pos_prompt_indexes, neg_prompt_indexes):
        super(NLIMinSentimentClassificationHead, self).__init__()
        
        self.num_prompts = num_prompts
        self.nli_model = nli_model
        
        self.pos_prompt_indexes = pos_prompt_indexes
        self.neg_prompt_indexes = neg_prompt_indexes
        
    def forward(self, reviews_and_prompts):
        
        nli_output = self.nli_model(**reviews_and_prompts)["logits"]

        outputs = torch.Tensor().to(self.nli_model.device)

        # Text Attack NLI Labels: 0-> Contradiction, 1-> Entailment, 2-> Neutral
        # Sentiment Polarity Labels: 0-> Positive, 1-> Negative, 2-> Neutral
        for i in range(len(nli_output)//self.num_prompts):
            prompts_batch = nli_output[i*self.num_prompts:(i+1)*self.num_prompts]

            pos_logit = torch.mean(prompts_batch[self.pos_prompt_indexes], dim=0)[1]

            neg_logit = torch.mean(prompts_batch[self.neg_prompt_indexes], dim=0)[1]

            neu_logit = torch.min(prompts_batch, dim=0)[0][2]

            pred_logits = torch.stack([pos_logit, neg_logit, neu_logit])
            pred_logits = torch.reshape(pred_logits, (1,-1))

            outputs = torch.cat([outputs, pred_logits])

        return outputs
