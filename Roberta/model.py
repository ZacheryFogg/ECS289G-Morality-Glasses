import math 
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from transformers import RobertaForMaskedLM



#####################################
#      Building Block Classes       #
#####################################
class RobertaEmbeddings(nn.Module):
   
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
      
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids):

        def create_position_ids_from_input_ids(input_ids, padding_idx):
        
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

class RobertaSdpaSelfAttention(RobertaSelfAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward( self, hidden_states, attention_mask = None):
        
        bsz, tgt_len, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

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

class LoraSdpaSelfAttention(RobertaSdpaSelfAttention):
    def __init__(self, config):
        super().__init__(config)

        self.lora_q_A = nn.Parameter(torch.zeros(self.all_head_size, config.rank))
        self.lora_q_B = nn.Parameter(torch.randn(config.rank, self.all_head_size))

        self.lora_v_A = nn.Parameter(torch.zeros(self.all_head_size, config.rank))
        self.lora_v_B = nn.Parameter(torch.randn(config.rank, self.all_head_size))

    def forward(self, hidden_states, attention_mask = None):

        bsz, tgt_len, _ = hidden_states.size()

        # LoRA Query
        lora_q_weights = torch.matmul(self.lora_q_B, self.lora_q_A)
        query_layer = self.query(hidden_states) + F.linear(hidden_states, lora_q_weights)
        query_layer = self.transpose_for_scores(query_layer)

        # LoRA Value
        lora_v_weights = torch.matmul(self.lora_v_B, self.lora_v_A)
        value_layer = self.value(hidden_states) + F.linear(hidden_states, lora_v_weights)
        value_layer = self.transpose_for_scores(value_layer)

        key_layer = self.transpose_for_scores(self.key(hidden_states))

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

class RobertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.attn_type == 'spda':
            self.self = RobertaSdpaSelfAttention(config)
        elif config.attn_type == 'lora_spda':
            self.self = LoraSdpaSelfAttention(config)
        else: 
            self.self = RobertaSelfAttention(config)

        self.output = RobertaSelfOutput(config)

    def forward(self, hidden_states,attention_mask = None):
        
        self_outputs = self.self( hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        
        return attention_output


class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


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
     
        embedding_output = self.embeddings(input_ids=input_ids)

        encoder_outputs = self.encoder( embedding_output, attention_mask=attention_mask)
        
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


#####################################
#    Different RoBERTa Classes      #
#####################################




#############
# Standard  #
#############

class RobertaClassificationAndLM(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.config = config

        self.classification_head = RobertaClassificationHead(config)

        # weight tying between input embedding and prediction head "de-embedding"
        self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight 

    def forward( self, input_ids, attention_mask = None, labels = None, run_lm_head = False, run_classification_head = True):

        outputs = self.roberta( input_ids, attention_mask=attention_mask)

        token_predictions = None 
        if run_lm_head:
            token_predictions = self.lm_head(outputs)

        classification_scores = None 
        if run_classification_head:
            classification_scores = self.classification_head(outputs)

        return token_predictions, classification_scores, outputs
    
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
      
            assert(sd[keys[0]].shape == sd_hf[keys[1]].shape)
            assert(keys[0] == keys[1])
            
            with torch.no_grad():
                sd[keys[0]].copy_(sd_hf[keys[1]])

        return model
    

class LoraRobertaClassificationAndLM(RobertaClassificationAndLM):
    def __init__(self, config):
        super(config).__init__()
    
    def freeze