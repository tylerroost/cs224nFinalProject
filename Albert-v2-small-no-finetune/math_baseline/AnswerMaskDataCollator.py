import torch
import transformers
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class AnswerMaskDataCollator(transformers.DataCollator):
    tokenizer: transformers.PreTrainedTokenizer
    mlm_probability: float = 0.15

    def collate_batch(self, batch) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        # print(batch)
        input_ids = torch.stack([torch.tensor(example.input_ids, dtype = torch.long) for example in batch])
        attention_mask = torch.stack([torch.tensor(example.attention_mask, dtype = torch.long) for example in batch])
        token_type_ids = torch.stack([torch.tensor(example.token_type_ids, dtype = torch.long) for example in batch])

        input_ids, labels = self.mask_answers(input_ids, token_type_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'masked_lm_labels': labels
        }

    def mask_answers(self, input_ids, token_type_ids):
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        # we never want to mask the questions because the model may not be able to infer the values from the answers
        # TODO see if it can infer the values from the answers
        # question_mask = token_type_ids.eq(0)
        # probability_matrix.masked_fill_(question_mask, value = 0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        answer_mask = token_type_ids.eq(1)

        # 50% of the time, we replace answer tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & answer_mask
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # The rest of the time (50% of the time) we keep the masked answer tokens unchanged
        return input_ids, labels
