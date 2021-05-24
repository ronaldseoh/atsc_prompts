import torch


class MultiPromptLogitSentimentClassificationHead(torch.nn.Module):
    def __init__(self, lm, num_class, num_prompts, pseudo_label_words, target_token_id=-1,
                 merge_behavior='sum_logits', perturb_prompts=False):

        super(MultiPromptLogitSentimentClassificationHead, self).__init__()

        self.num_class = num_class
        self.pseudo_label_words = pseudo_label_words
        self.target_token_id = target_token_id
        self.num_prompts = num_prompts
        self.merge_behavior = merge_behavior
        self.perturb_prompts = perturb_prompts

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
        
        # Additive perturbation of tokens in the prompt
        if self.perturb_prompts:
            self.perturb_embeddings = torch.nn.Embedding(
                self.lm.config.vocab_size, self.lm.config.hidden_size, padding_idx=self.lm.config.pad_token_id)
                
            # Initialize the perturb embeddings with zeros
            torch.nn.init.zeros_(self.perturb_embeddings.weight)

    def forward(self, reviews_and_prompts):

        # Figure out where the mask token was placed
        if self.lm_type == 'bert':
            # For BERT, we need to find the token in each input with [MASK]
            target_indexes = torch.nonzero(
                reviews_and_prompts.data["input_ids"] == self.target_token_id)[:, 1]

            if self.perturb_prompts:
                word_embeds = self.lm.bert.embeddings.word_embeddings(reviews_and_prompts.input_ids)

                # Use 'token_type_ids' to find where the prompt begins (after the [SEP] token)
                # and filter out positions that don't belong to prompts                
                perturb_input_ids = reviews_and_prompts.input_ids * reviews_and_prompts.token_type_ids

                # Get rid of the last [SEP] token in the end
                perturb_input_ids = perturb_input_ids * (perturb_input_ids != 102)
                
                # Get rid of the [MASK] token as well
                perturb_input_ids = perturb_input_ids * (perturb_input_ids != self.target_token_id)                

                perturb_embeds = self.perturb_embeddings(perturb_input_ids)
                
                lm_outputs = self.lm(
                    inputs_embeds=word_embeds+perturb_embeds,
                    token_type_ids=reviews_and_prompts.token_type_ids,
                    attention_mask=reviews_and_prompts.attention_mask)
            else:
                lm_outputs = self.lm(**reviews_and_prompts)

            real_batch_size = len(reviews_and_prompts.data["input_ids"]) // self.num_prompts

        elif self.lm_type == 'gpt2':
            lm_outputs = []
            target_indexes = []

            # For GPT-2, we need to find the spot right of the last token before <|endoftext|>
            t = (reviews_and_prompts.data["input_ids"] == self.target_token_id).int()
            t = t.cpu() * torch.arange(t.shape[1], 0, -1).cpu()
            target_indexes = torch.argmax(t.cpu(), 1, keepdim=False) -1
            
            lm_outputs = self.lm(**reviews_and_prompts) 
            real_batch_size = len(reviews_and_prompts.data["input_ids"]) // self.num_prompts

        outputs = []
                
        for i in range(real_batch_size):
            scores_batch = []

            for j in range(self.num_prompts):
                if self.merge_behavior == 'sum_logits':
                    # logit output assigned to self.pseudo_label_words
                    logits = lm_outputs.logits[i+real_batch_size*j, target_indexes[i+real_batch_size*j], self.pseudo_label_words[j]]
                    
                    scores_batch.append(logits)

                elif self.merge_behavior == 'sum_probabilities':
                    probabilities = torch.nn.functional.softmax(
                        lm_outputs.logits[i+real_batch_size*j, target_indexes[i+real_batch_size*j]], dim=-1)

                    probs_pseudo_labels = probabilities[self.pseudo_label_words[j]]

                    scores_batch.append(probs_pseudo_labels)
                
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
    def __init__(self, lm, num_class, num_prompts, target_token_id=-1,
                 merge_behavior='concatenate', perturb_prompts=False):

        super(MultiPromptSentimentClassificationHead, self).__init__()

        self.num_class = num_class
        self.num_prompts = num_prompts
        self.target_token_id = target_token_id
        self.merge_behavior = merge_behavior
        self.perturb_prompts = perturb_prompts

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

        # Additive perturbation of tokens in the prompt
        if self.perturb_prompts:
            self.perturb_embeddings = torch.nn.Embedding(
                self.lm.config.vocab_size, self.lm.config.hidden_size, padding_idx=self.lm.config.pad_token_id)
                
            # Initialize the perturb embeddings with zeros
            torch.nn.init.zeros_(self.perturb_embeddings.weight)

        # Linear layer
        if self.merge_behavior == 'concatenate':
            self.linear = torch.nn.Linear(
                self.num_prompts * self.lm.config.hidden_size, self.num_class)
        elif self.merge_behavior == 'sum':
            self.linear = torch.nn.Linear(
                self.lm.config.hidden_size, self.num_class)

    def forward(self, reviews_and_prompts):

        # Extract hidden states and feed them to self.linear
        outputs = []

        lr_inputs_batch = []

        # Figure out where the mask token was placed
        if self.lm_type == 'bert':
            # For BERT, we need to find the token in each input with [MASK]
            target_indexes = torch.nonzero(
                reviews_and_prompts.data["input_ids"] == self.target_token_id)[:, 1]
                
            if self.perturb_prompts:
                word_embeds = self.lm.bert.embeddings.word_embeddings(reviews_and_prompts.input_ids)

                # Use 'token_type_ids' to find where the prompt begins (after the [SEP] token)
                # and filter out positions that don't belong to prompts                
                perturb_input_ids = reviews_and_prompts.input_ids * reviews_and_prompts.token_type_ids

                # Get rid of the last [SEP] token in the end
                perturb_input_ids = perturb_input_ids * (perturb_input_ids != 102)

                # Get rid of the [MASK] token as well
                perturb_input_ids = perturb_input_ids * (perturb_input_ids != self.target_token_id)                

                perturb_embeds = self.perturb_embeddings(perturb_input_ids)
                
                lm_outputs = self.lm(
                    inputs_embeds=word_embeds+perturb_embeds,
                    token_type_ids=reviews_and_prompts.token_type_ids,
                    attention_mask=reviews_and_prompts.attention_mask,
                    output_hidden_states=True)
            else:
                lm_outputs = self.lm(**reviews_and_prompts, output_hidden_states=True)

            real_batch_size = len(reviews_and_prompts.input_ids) // self.num_prompts

        elif self.lm_type == 'gpt2':
            lm_outputs = []
            target_indexes = []

            # For GPT-2, we need to find the spot right of the last token before <|endoftext|>  
            t = (reviews_and_prompts.data["input_ids"] == self.target_token_id).int()
            t = t * torch.arange(t.shape[1], 0, -1).cpu()
            target_indexes = torch.argmax(t, 1, keepdim=False) -1

            lm_outputs = self.lm(**reviews_and_prompts) 
            real_batch_size = len(reviews_and_prompts.data["input_ids"]) // self.num_prompts    
                
        for i in range(real_batch_size):
            # Create an input to self.linear by
            # concatenating last hidden states for this review
            lr_input = []

            for j in range(self.num_prompts):
                lr_input.append(lm_outputs["hidden_states"][-1][i+real_batch_size*j][target_indexes[i+real_batch_size*j]])
              
            if self.merge_behavior == 'concatenate':
                lr_input = torch.cat(lr_input, dim=0)
            elif self.merge_behavior == 'sum':
                # Do not perform stack and sum operation on single prompt
                if self.num_prompts == 1:
                    lr_input = lr_input[0]
                else:
                    lr_input = torch.stack(lr_input, dim=0)
                    lr_input = torch.sum(lr_input, dim=0)

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
