import math 
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as n
from transformers import RobertaForMaskedLM



#####################################
#      Building Block Classes       #
#####################################
class RobertaEmbeddings(nn.Module):
   
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

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

        self.lora_q_B = nn.Parameter(torch.zeros(self.all_head_size, config.rank))
        self.lora_q_A = nn.Parameter(torch.randn(config.rank, self.all_head_size))

        self.lora_v_B = nn.Parameter(torch.zeros(self.all_head_size, config.rank))
        self.lora_v_A = nn.Parameter(torch.randn(config.rank, self.all_head_size))

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

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, attention_mask = None):
        
        self_outputs = self.self( hidden_states, attention_mask)

        attention_output = self.dense(self_outputs)
        
        return attention_output
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_projection = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_projection = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        x = self.up_projection(x)
        x = F.gelu(x)
        x = self.down_projection(x)
        return x
    
class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.attention = RobertaAttention(config)
        self.LayerNorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.ffn = FeedForwardNetwork(config)
        self.LayerNorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward( self, hidden_states, attention_mask = None):

        attention_outputs = self.attention( hidden_states, attention_mask)
        hidden_states = self.LayerNorm1(attention_outputs + hidden_states)

        ffn_outputs = self.ffn(hidden_states)
        layer_output = self.LayerNorm2(ffn_outputs + hidden_states)
        
        return layer_output

# class RobertaLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
        
#         self.attention = RobertaAttention(config)
#         self.intermediate = RobertaIntermediate(config)
#         self.output = RobertaOutput(config)

#     def forward( self, hidden_states, attention_mask = None):
#         # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
#         self_attention_outputs = self.attention( hidden_states, attention_mask)
#         attention_output = self_attention_outputs

#         intermediate_output = self.intermediate(attention_output)
#         layer_output = self.output(intermediate_output, attention_output)
        
#         return layer_output
    

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
        self.out_proj = nn.Linear(config.hidden_size, config.num_class_labels)

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

class RobertaClassificationAndLM2(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.config = config

        self.classification_head = RobertaClassificationHead(config)

        # weight tying between input embedding and prediction head "de-embedding"
        self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight 

    def forward( self, input_ids, attention_mask, run_lm_head = False, run_classification_head = True):

        outputs = self.roberta( input_ids, attention_mask=attention_mask)

        token_predictions = None 
        if run_lm_head:
            token_predictions = self.lm_head(outputs)

        classification_scores = None 
        if run_classification_head:
            classification_scores = self.classification_head(outputs)

        return token_predictions, classification_scores, outputs
    
    @classmethod
    def from_pretrained(cls, config):
        """ Loading pretrained Roberta weights from hugging face """

        # Random init of model
        model = RobertaClassificationAndLM2(config)
        
        sd = model.state_dict()

        # Init a Roberta from hugging face 
        model_hf = RobertaForMaskedLM.from_pretrained("FacebookAI/roberta-base")

        sd_hf = model_hf.state_dict()
        sd_hf_keys = [k for k in sd_hf.keys() if not k.endswith('lm_head.bias')]
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith('roberta.embeddings.token_type_embeddings.weight')]

        # Copy over weights from pre-trained models 
        key_map = {
            'attention.self.query.weight' : 'attention.self.query.weight',
            'attention.self.query.bias' : 'attention.self.query.bias',
            'attention.self.key.weight' : 'attention.self.key.weight',
            'attention.self.key.bias' : 'attention.self.key.bias',
            'attention.self.value.weight' : 'attention.self.value.weight',
            'attention.self.value.bias' : 'attention.self.value.bias',
            'attention.output.dense.weight' : 'attention.dense.weight',
            'attention.output.dense.bias' : 'attention.dense.bias',
            'attention.output.LayerNorm.weight' : 'LayerNorm1.weight',
            'attention.output.LayerNorm.bias' : 'LayerNorm1.bias',
            'intermediate.dense.weight' : 'ffn.up_projection.weight',
            'intermediate.dense.bias' : 'ffn.up_projection.bias',
            'output.dense.weight' : 'ffn.down_projection.weight',
            'output.dense.bias' : 'ffn.down_projection.bias',
            'output.LayerNorm.weight' : 'LayerNorm2.weight',
            'output.LayerNorm.bias' : 'LayerNorm2.bias',
        }
        for keys in zip(sd.keys(),sd_hf_keys):
            assert(sd[keys[0]].shape == sd_hf[keys[1]].shape)
            
            with torch.no_grad():
                sd[keys[0]].copy_(sd_hf[keys[1]])
        # for key in sd_hf_keys:
            
        #     correct_key = None

        #     name = key.split('.')

        #     if name[2] == 'layer':
        #         l_num = name[3]
        #         prefix_name = f'roberta.encoder.layer.{l_num}.'
        #         suffix_name = key.split(l_num + '.')[1]
        #         correct_key = prefix_name + key_map[suffix_name]
        #     else: 
        #         correct_key = key


        #     assert(sd[correct_key].shape == sd_hf[key].shape)
            
        #     with torch.no_grad():
        #         sd[correct_key].copy_(sd_hf[key])

        return model

# class AdapterRobertaClassificationAndLM(RobertaClassificationAndLM):
#     def __init__(self, config):
#         super(config).__init__()

#     def freeze_base_weights(self):
#         pass

#     @classmethod
#     def load_from_pretrained(cls, config):
        
#         model = AdapterRobertaClassificationAndLM(config)

#         sd = model.state_dict()

#         model_hf = RobertaForMaskedLM.from_pretrained("FacebookAI/roberta-base")
#         sd_hf = model_hf.state_dict()

#         sd_hf_keys = [k for k in sd_hf.keys() if not k.endswith('lm_head.bias')]

#         for key in sd_hf_keys:

#             assert(sd[key].shape == sd_hf[key].shape)

#             with torch.no_grad():
#                 sd[key].copy_(sd_hf[key])

#         return model
