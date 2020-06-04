import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import transformers

class AlbertForMathQuestions(nn.Module):
    def __init__(self, albert_for_math_config):
        super(AlbertForMathQuestions, self)
        self.albert = transformers.AlbertModel(albert_for_math_config)
        self.dense  = nn.Linear(albert_for_math_config.hidden_size, albert_for_math_config.embedding_size)
        self.output_projection = nn.Linear(albert_for_math_config.embedding_size, albert_for_math_config.vocab_size)
        self.layer_norm = nn.LayerNorm(albert_for_math_config.embedding_size)
        self.activation = transformers.ACT2FN[config.hidden_act]
    def forward(self,
                input_ids = None,
                attention_masks = None,
                token_type_ids = None,
                position_ids = None,
                head_mask = None,
                input_embeds = None,
                masked_lm_labels = None,
    ):
        outputs = self.albert(input_ids,
                              input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds,
                             )
        hidden_layer_outputs = outputs
        outputs = self.dense(outputs[0])
        outputs = self.activation(outputs)
        outputs = self.layer_norm(outputs)
        outputs = self.output_projection(outputs)
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs
