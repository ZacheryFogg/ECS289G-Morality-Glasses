####################################
#            Imports               #
####################################

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from transformers import RobertaTokenizer, RobertaForMaskedLM
from typing import List
from torch.utils.data import DataLoader, Dataset, Subset
from model import RobertaClassificationAndLM
from data import EthicsDataset, MoralStoriesDataset
from datasets import load_dataset
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from pynvml import *
from matplotlib.pyplot import figure
import time

from helper import create_attention_mask, calculate_accuracy_loss, train_model, print_gpu_utilization, get_gpu_mem_usage


if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    
print(f"using device: {device}")

torch.set_float32_matmul_precision('high')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")


####################################
#            Datasets              #
####################################
train_dataset_moral = torch.load('./Datasets/train_dataset_moral.pt')
val_dataset_moral = torch.load('./Datasets/val_dataset_moral.pt')
test_dataset_moral = torch.load('./Datasets/test_dataset_moral.pt')

batch_size_moral = 96
train_moral_loader_moral = DataLoader(train_dataset_moral, batch_size = batch_size_moral, shuffle = False)
val_moral_loader_moral = DataLoader(val_dataset_moral, batch_size = batch_size_moral, shuffle = False)
test_moral_loader_moral = DataLoader(test_dataset_moral, batch_size = batch_size_moral, shuffle = False)


####################################
#      Creation of Models          #
####################################

# Creation of base model
@dataclass
class RobertaConfig:
    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 514
    layer_norm_eps: float = 1e-12
    num_class_labels: int = 1
    pad_token_id: int = 1

    # Special Configs 
    rank: int = None
    attn_type: str = 'spda'
    use_bottleneck: bool = False
    bottleneck_size: int = None
    prefix_size: int = None
    use_prefix: bool = False

base_model = RobertaClassificationAndLM.from_pretrained(RobertaConfig())
for name, param in base_model.named_parameters():
    if "lm_head" in name:
        param.requires_grad = False 

# # Creation LoRA model 
lora_model_rank_1 = RobertaClassificationAndLM.from_pretrained(RobertaConfig(attn_type = 'lora_spda', rank = 1))
lora_model_rank_2 = RobertaClassificationAndLM.from_pretrained(RobertaConfig(attn_type = 'lora_spda', rank = 2))

# # Freeze non lora params 
for name, param in lora_model_rank_1.named_parameters():
    if "lora" not in name and "classification" not in name:
        param.requires_grad = False  
for name, param in lora_model_rank_2.named_parameters():
    if "lora" not in name and "classification" not in name:
        param.requires_grad = False


# # Creation of Adapter model 
adapter_model_8 = RobertaClassificationAndLM.from_pretrained(RobertaConfig(use_bottleneck = True, bottleneck_size = 8))
adapter_model_32 = RobertaClassificationAndLM.from_pretrained(RobertaConfig(use_bottleneck = True, bottleneck_size = 32))

for name, param in adapter_model_8.named_parameters():
    if "bottleneck" not in name and "classification" not in name and 'LayerNorm2' not in name:
        param.requires_grad = False
for name, param in adapter_model_32.named_parameters():
    if "bottleneck" not in name and "classification" not in name and 'LayerNorm2' not in name:
        param.requires_grad = False

# Creation of Prefix Model 
prefix_model_64 = RobertaClassificationAndLM.from_pretrained(RobertaConfig(use_prefix = True, prefix_size = 64))
prefix_model_50 = RobertaClassificationAndLM.from_pretrained(RobertaConfig(use_prefix = True, prefix_size = 50))

# Freeze non prefix weights 
for name, param in prefix_model_64.named_parameters():
    if "prefix" not in name and 'classification' not in name: 
        param.requires_grad = False

for name, param in prefix_model_50.named_parameters():
    if "prefix" not in name and 'classification' not in name: 
        param.requires_grad = False

models ={
    'prefix_model_64' : {
        'model' : prefix_model_64,
        'prefix_size' : 64,
        'lr' : 1e-3
    },
    'prefix_model_50' : {
        'model' : prefix_model_50,
        'prefix_size' : 50,
        'lr' : 1e-3
    },
    'base_model' : {
        'model' : base_model,
        'prefix_size' : 0,
        'lr' : 1e-4
    },
    'lora_model_rank_1' : {
        'model' : lora_model_rank_1,
        'prefix_size' : 0,
        'lr' : 1e-4
    },
    'lora_model_rank_2' : {
        'model' : lora_model_rank_2,
        'prefix_size' : 0,
        'lr' : 1e-4
    },
    'adapter_model_8' : {
        'model' : adapter_model_8,
        'prefix_size' : 0,
        'lr' : 1e-4
    },
    'adapter_model_32' : {
        'model' : adapter_model_32,
        'prefix_size' : 0,
        'lr' : 1e-4
    },
}


####################################
#      Train for 180s              #
####################################

training_results ={}
epochs = 100 # Models will not actually get to train for this long
max_training_time = 180 # Max training time per model is 180s, this is the training compute cap

for key in models.keys():
    print(f'\n{key}\n')

    model = models[key]['model'].to(device)
    results = train_model(model, 
                          epochs, 
                          train_moral_loader_moral, 
                          val_moral_loader_moral, 
                          f'180/{key}',
                          device, 
                          max_training_time= max_training_time,
                          prefix_size = models[key]['prefix_size'], 
                          lr = models[key]['lr'])
    training_results[key] = results

with open('results/capped_180_training.json', 'w') as f:
    json.dump(training_results , f)



####################################
#      Train for 600s              #
####################################

training_results ={}
epochs = 100 
max_training_time = 600 # Max training time per model is 180s, this is the training compute cap

for key in models.keys():
    print(f'\n{key}\n')
    print_gpu_utilization()

    model = models[key]['model'].to(device)
    results = train_model(model, 
                          epochs, 
                          train_moral_loader_moral, 
                          val_moral_loader_moral, 
                          f'600/{key}',
                          device, 
                          max_training_time= max_training_time,
                          prefix_size = models[key]['prefix_size'], 
                          lr = models[key]['lr'])
    training_results[key] = results
    torch.cuda.empty_cache() 
    print_gpu_utilization()
with open('results/capped_600_training.json', 'w') as f:
    json.dump(training_results , f)
