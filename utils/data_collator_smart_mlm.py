from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
import transformers
import spacy
import tokenizations


def _collate_batch(examples, tokenizer):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


@dataclass
class DataCollatorForSmartMLM:
    """
    Data collator for our "smart" LM objective.
    Based on the codes from
    https://github.com/huggingface/transformers/blob/24184e73c441397edd51e9068e0f49c0418d25ab/src/transformers/data/data_collator.py
    """

    # Tokenizer instance
    tokenizer: transformers.PreTrainedTokenizerBase
    
    # POS tags to mask.
    # Inputs should be a list of strings from
    # https://universaldependencies.org/docs/u/pos/
    pos_tags_to_mask: Optional[List[str]] = field(default_factory=lambda: ['ADJ', 'NOUN', 'PROPN'])

    # Probability of actually considering the detected tokens for masking.
    mlm_probability: float = 1.0
    
    # Probability of replacing the token to mask with `[MASK]'
    prob_replace_with_mask: float = 1.0
    
    # Probability of replacing the token to mask with a random word
    prob_replace_with_random: float = 0.0

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            )
        
        # Load spaCy core model for rule-based POS tagging
        self.spacy_nlp = spacy.load("en_core_web_sm")

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, transformers.BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt")
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        return batch

    def _find_tokens_to_mask(self, encoded_text_batch):
        # Final mask location info to return
        masked_indices_batch = torch.full(encoded_text_batch.shape, False, dtype=torch.bool)
        
        # if self.pos_tags_to_mask was intentionally declared empty,
        # Then we simply do not mask anything
        if len(self.pos_tags_to_mask) == 0:
            return masked_indices_batch
        
        # Otherwise...
        # We need to decode back to the original sentence, in order to process
        # with spaCy
        decoded_text_batch = self.tokenizer.batch_decode(encoded_text_batch, skip_special_tokens=True)
        
        # Also get a list of tokens, to find the alignment between BERT tokens and spaCy tokens
        decoded_text_tokens_batch = [
            self.tokenizer.convert_ids_to_tokens(encoded_text, skip_special_tokens=True) for encoded_text in encoded_text_batch]
        
        # Just in case there was only one token, still make decoded_text_tokens a list
        if isinstance(decoded_text_tokens_batch, str):
            decoded_text_tokens_batch = [decoded_text_tokens_batch]
        
        for i, (decoded_text, decoded_text_tokens) in enumerate(zip(decoded_text_batch, decoded_text_tokens_batch)):
            # Process decoded_text through spaCy
            tokens_from_spacy = []
            tokens_with_target_pos = []
            
            for j, token in enumerate(self.spacy_nlp(decoded_text)):
                tokens_from_spacy.append(token.text)
                
                if token.pos_ in self.pos_tags_to_mask:
                    tokens_with_target_pos.append(j)
            
            # Find the alignment
            spacy_to_bert, _ = tokenizations.get_alignments(tokens_from_spacy, decoded_text_tokens)
            
            # We only need to look at alignments with relevant POS
            alignments_to_mask = []

            for token_index in tokens_with_target_pos:
                alignments_to_mask.append(spacy_to_bert[token_index])
            
            # Store the masking information for the current text in the batch
            # to masked_indices_batch
            for alignment_index in alignments_to_mask:
                for bert_index in alignment_index:
                    # Actually mask the token, `mlm_probability` percent of the time
                    random_mask_decision = torch.bernoulli(torch.Tensor([1]), self.mlm_probability).bool()
                    
                    # item() is used to get a Python boolean
                    if random_mask_decision.item():
                        # bert_index + 1 because There is a [CLS] token in 0-th position,
                        # which we removed when we did the decoding and compute the alignments
                        # with skip_special_tokens=True
                        masked_indices_batch[i][bert_index + 1] = True

        return masked_indices_batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # The ones in masked_indices will get the masking treatment
        masked_indices = self._find_tokens_to_mask(inputs)
        
        # Make sure that the positions with those special tokens do not get masked out
        masked_indices.masked_fill_(special_tokens_mask, value=False)

        # No loss calculation on non-masked tokens
        labels[~masked_indices] = -100

        # `prob_replace_with_mask` percent of the time,
        # we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.prob_replace_with_mask)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        remaining_probs = 1 - self.prob_replace_with_mask

        if remaining_probs > 0:
            # `prob_replace_with_random` percent of the time,
            # we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, self.prob_replace_with_random / remaining_probs)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

        # The rest of the time, we keep the masked input tokens unchanged
        return inputs, labels
