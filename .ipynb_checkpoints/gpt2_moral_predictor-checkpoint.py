import math 
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken



class MLP(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,config.n_embd * 4)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class CasualSelfAttention(nn.Module):

    def __init__(self,config):
        super().__init__()

        # Ensure that when we split Q,K,V into the multiple heads, the head_size makes sense
        assert config.n_embd % config.n_head == 0

        # Projection matrices for QKV, formatted at one matrix 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection which projects the concatenated head outputs back to emb_dim 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # Batch size, sequence length, embedding dim
        B, T, C = x.size()

        qkv = self.c_attn(x)

        q, k ,v = qkv.split(self.n_embd, dim = 2) # Split on the channel dimension into the 3 equal sized q, k, v

        # Split each q, k, v - which is currently of size embedding dim - into num_head smaller matrices of size head_size
        # This way of calculating QKV is more efficient than doing calculation for num_head different heads 
        # Head size is inferred from embedding_dim / num_heads ... e.g. 768 / 6 heads = 128 head size 
        # Switch seq_len and num_head dimension
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num head, T, head size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num head, T, head size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num head, T, head size)
        
        # Calculate attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) = (B, nh, T, hs)

        y = y.transpose(1,2).contiguous().view(B,T,C) # Concat head outputs

        # Project 
        y = self.c_proj(y)

        return y

class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence len
    vocab_size: int = 50257 # num tokensL 50k BPE + 256 bytes tokens + 1 <|endoftext|>
    n_layer: int = 12 # num layers
    n_head: int = 12 # num heads
    n_embd: int = 768 # embedding dim


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Dictionary that holds all submodules that represent the transformer 
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Token embedding matrix 
            wpe = nn.Embedding(config.block_size, config.n_embd), # Positional Encoding embedding matrix
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # The n blocks of the transformer 
            ln_f = nn.LayerNorm(config.n_embd), # Final layer norm
        ))

        self.lm_head = nn.Linear(config.n_embd, 1, bias = False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False) # Prediction head ##TODO Replace with single nuuron prediction head


        

    def forward(self, idx):
        # idx is shape (B, T) and represent B batches or T indexes corresponding to tokens in embedding
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence too long"

        # Retrieve positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # pos embeddings of shape (T, n_emb)
        tok_emb = self.transformer.wte(idx) # tok embeddings of shape (B, T, n_embd)

        x = tok_emb + pos_emb

        # forward input through transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward through final layer norm and classifer
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x) # (B, T, vocab_size)

        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('lm_head.weight')] # will not be replacing lm_head

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                print(sd_hf[k])
        # Create weight matrix for prediction head

        return model



model = GPT.from_pretrained('gpt2')
print("Model weights loaded")
model.eval()

tokenizer = tiktoken.get_encoding('gpt2')
 
prompt = "Hello, I'm a loser model, "

tokens = tokenizer.encode(prompt)
tokens = torch.tensor(tokens, dtype = torch.long)
x = tokens.unsqueeze(0)

# torch.manual_seed(42)

with torch.no_grad():

    logits = model(x)
    logits = logits[:, -1, :]

    print(logits)
    probs = F.sigmoid(logits)

print(probs)
