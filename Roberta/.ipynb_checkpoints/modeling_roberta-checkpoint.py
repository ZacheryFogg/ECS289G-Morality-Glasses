import math 
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from typing import List, Optional, Tuple, Union
from transformers import RobertaForMaskedLM

class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # self.register_buffer(
        #     "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        # )
    
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids):

        def create_position_ids_from_input_ids(input_ids, padding_idx):
        
            # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
            mask = input_ids.ne(padding_idx).int()
            incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
            return incremental_indices.long() + padding_idx
    

        position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
            
        embeddings = self.word_embeddings(input_ids)
                
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings += position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        
        return embeddings

class RobertaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
       
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward( self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
    
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs =  context_layer

        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSdpaSelfAttention with Bert->Roberta
class RobertaSdpaSelfAttention(RobertaSelfAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward( self, hidden_states, attention_mask = None):
        
        bsz, tgt_len, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        current_states = hidden_states
        attention_mask = attention_mask

        key_layer = self.transpose_for_scores(self.key(current_states))
        value_layer = self.transpose_for_scores(self.value(current_states))

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)

        outputs = attn_output

        return outputs

class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class RobertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = RobertaSdpaSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(self, hidden_states,attention_mask = None):
        
        self_outputs = self.self( hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        
        return attention_output

class RobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
        
class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.attention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward( self, hidden_states, attention_mask = None):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention( hidden_states, attention_mask)
        attention_output = self_attention_outputs

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        return layer_output


class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward( self, hidden_states, attention_mask = None):
        
        for i, layer_module in enumerate(self.layer):
          
            layer_outputs = layer_module(
                hidden_states,
                attention_mask
            )

            hidden_states = layer_outputs

        return hidden_states


class RobertaModel(nn.Module):


    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

    def forward(self, input_ids, attention_mask = None):
   
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        
        device = input_ids.device
    
        embedding_output = self.embeddings(input_ids=input_ids)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        use_sdpa_attention_masks = True

        extended_attention_mask = attention_mask
        # # Expand the attention mask
        # if use_sdpa_attention_masks and attention_mask.dim() == 2:
        #     # Expand the attention mask for SDPA.
        #     # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
        #     extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
        #         attention_mask, embedding_output.dtype, tgt_len=seq_length
        #     )
        # else:
        #     # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        #     # ourselves in which case we just need to make it broadcastable to all heads.
        #     extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        encoder_outputs = self.encoder( embedding_output, attention_mask=extended_attention_mask)
        
        return encoder_outputs

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.gelu = nn.GELU()
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, features):
        x = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class RobertaMaskedLM(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.config = config

        # weight tying between input embedding and prediction head "de-embedding"
        self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight 


    def forward( self, input_ids, attention_mask = None, labels = None):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

       
        output = prediction_scores
        return output
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
    
    @classmethod
    def from_pretrained(cls, model_type = "FacebookAI/roberta-base"):
        """ Loading pretrained Roberta weights from hugging face """
        # print("loading weights for %s" % model_type)

        # Random init of model
        config = RobertaConfig()
        model = RobertaMaskedLM(config)
        
        sd = model.state_dict()
        sd_keys = sd.keys()

        # Init a Roberta from hugging face 
        model_hf = RobertaForMaskedLM.from_pretrained("FacebookAI/roberta-base")
        sd_hf = model_hf.state_dict()
        sd_hf_keys = [k for k in sd_hf.keys() if not k.endswith('lm_head.bias')]
        # Copy over weights. State Dicts are currently in same order, so I can just blind copy 
        for keys in zip(sd_keys, sd_hf_keys):
            # print(sd[keys[0]].shape)
            # print(sd_hf[keys[1]].shape)
            
            assert(sd[keys[0]].shape == sd_hf[keys[1]].shape)
            assert(keys[0] == keys[1])
            
            with torch.no_grad():
                sd[keys[0]].copy_(sd_hf[keys[1]])

        return model
    

class RobertaClassificationAndLM(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.config = config

        self.classification_head = RobertaClassificationHead(config)

        # weight tying between input embedding and prediction head "de-embedding"
        self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight 

    def forward( self, input_ids, attention_mask = None, labels = None):

        outputs = self.roberta( input_ids, attention_mask=attention_mask)
        token_predictions = self.lm_head(outputs)
        classification_scores = self.classification_head(outputs)

        # masked_lm_loss = None
        # if labels is not None:
        #     # move labels to correct device to enable model parallelism
        #     labels = labels.to(prediction_scores.device)
        #     loss_fct = CrossEntropyLoss()
        #     masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

       
        return token_predictions, classification_scores, outputs
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
    
    @classmethod
    def from_pretrained(cls, model_type = "FacebookAI/roberta-base"):
        """ Loading pretrained Roberta weights from hugging face """
        # print("loading weights for %s" % model_type)

        # Random init of model
        config = RobertaConfig()
        model = RobertaClassificationAndLM(config)
        
        sd = model.state_dict()
        sd_keys = sd.keys()

        # Init a Roberta from hugging face 
        model_hf = RobertaForMaskedLM.from_pretrained("FacebookAI/roberta-base")
        sd_hf = model_hf.state_dict()
        sd_hf_keys = [k for k in sd_hf.keys() if not k.endswith('lm_head.bias')]
        # Copy over weights. State Dicts are currently in same order, so I can just blind copy 
        for keys in zip(sd_keys, sd_hf_keys):
            # print(sd[keys[0]].shape)
            # print(sd_hf[keys[1]].shape)
            
            assert(sd[keys[0]].shape == sd_hf[keys[1]].shape)
            assert(keys[0] == keys[1])
            
            with torch.no_grad():
                sd[keys[0]].copy_(sd_hf[keys[1]])

        return model

@dataclass
class RobertaConfig:
    vocab_size = 50265
    hidden_size = 768 
    num_hidden_layers = 12
    num_attention_heads = 12
    intermediate_size = 3072
    max_position_embeddings = 514
    layer_norm_eps = 1e-12
    num_labels = 1
    
    type_vocab_size = 1
    pad_token_id = 1
    bos_token_id = 0
    eos_token_id = 2