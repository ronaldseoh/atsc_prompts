import torch

class SinglePromptLogitSentimentClassificationHead(torch.nn.Module):
    def __init__(self, mlm, num_class, pseudo_label_words, mask_token_id):
        super(SinglePromptLogitSentimentClassificationHead, self).__init__()

        self.num_class = num_class
        self.pseudo_label_words = pseudo_label_words
        self.mask_token_id = mask_token_id

        self.mlm = mlm

    def forward(self, reviews_and_prompts):

        mlm_outputs = self.mlm(**reviews_and_prompts)

        # Figures out where the mask token was placed
        masked_indexes = (reviews_and_prompts.data["input_ids"] == self.mask_token_id)

        outputs = mlm_outputs.logits[masked_indexes]
        
        outputs = outputs[:, self.pseudo_label_words]

        return outputs


class MultiPromptSentimentClassificationHead(torch.nn.Module):
    def __init__(self, mlm, num_class, num_prompts, mask_token_id):
        super(MultiPromptSentimentClassificationHead, self).__init__()

        self.num_class = num_class
        self.num_prompts = num_prompts
        self.mask_token_id = mask_token_id

        self.mlm = mlm

        self.fc1 = torch.nn.Linear(
            self.num_prompts * self.mlm.config.hidden_size, 300)
        
        self.fc2 = torch.nn.Linear(300, 100)
            
        self.fc3 = torch.nn.Linear(100, self.num_class)

    def forward(self, reviews_and_prompts):

        mlm_outputs = self.mlm(**reviews_and_prompts, output_hidden_states=True)

        # Figures out where the mask token was placed
        masked_indexes = torch.nonzero(
            reviews_and_prompts.data["input_ids"] == self.mask_token_id)[:, 1]

        outputs = []

        lr_inputs_batch = []

        real_batch_size = len(reviews_and_prompts.data["input_ids"]) // self.num_prompts

        for i in range(real_batch_size):
            # Create an input to self.linear by
            # concatenating last hidden states for this review
            lr_input = []

            for j in range(self.num_prompts):
                lr_input.append(mlm_outputs["hidden_states"][-1][i+real_batch_size*j][masked_indexes[i+real_batch_size*j]])

            lr_input = torch.cat(lr_input, dim=0)

            lr_inputs_batch.append(lr_input)

        lr_inputs_batch = torch.stack(lr_inputs_batch)

        outputs = torch.tanh(self.fc1(lr_inputs_batch))
        outputs = torch.tanh(self.fc2(outputs))
        outputs = self.fc3(outputs)

        return outputs
