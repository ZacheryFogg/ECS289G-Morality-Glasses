####################################
#            Imports               #
####################################

from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from transformers import RobertaTokenizer, RobertaForMaskedLM
from typing import List
from torch.utils.data import DataLoader, Dataset, Subset
from model import RobertaClassificationAndLM
from data import EthicsDataset, MoralStoriesDataset, WikiTextDataset, morality_probing_examples_easy, morality_probing_examples_hard
from datasets import load_dataset
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from pynvml import *
from matplotlib.pyplot import figure
import time
from helper import create_attention_mask, calculate_accuracy_loss, train_model, print_gpu_utilization, get_gpu_mem_usage, calculate_wikitext_loss
import argparse





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("max_training_time")

    model_name = parser.parse_args().model_name
    max_training_time = int(parser.parse_args().max_training_time)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    torch.manual_seed(1337)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-large")

    
    ####################################
    #            Datasets              #
    ####################################

    train_dataset_moral = torch.load('./Datasets/moral_stories/train_dataset_moral.pt', weights_only=False)
    val_dataset_moral = torch.load('./Datasets/moral_stories/val_dataset_moral.pt', weights_only=False)
    test_dataset_moral = torch.load('./Datasets/moral_stories/test_dataset_moral.pt',weights_only=False)
    val_datatset_moral_unmasked = torch.load('./Datasets/moral_stories/val_dataset_moral_unmasked.pt', weights_only = False)

    batch_size = 32

    train_loader_moral = DataLoader(train_dataset_moral, batch_size = batch_size, shuffle = False)
    val_loader_moral = DataLoader(val_dataset_moral, batch_size = batch_size, shuffle = False)
    test_loader_moral = DataLoader(test_dataset_moral, batch_size = batch_size, shuffle = False)
    val_loader_moral_unmasked = DataLoader(val_datatset_moral_unmasked, batch_size=batch_size, shuffle=False)

    test_dataset_wikitext = torch.load('./Datasets/wikitext/wikitext_dataset_test.pt', weights_only=False)
    test_loader_wikitext = DataLoader(test_dataset_wikitext, batch_size = batch_size, shuffle = False)

    morality_probing_dataset = morality_probing_examples_easy + morality_probing_examples_hard


    ####################################
    #      Creation of Models          #
    ####################################
    
    @dataclass 
    class RobertaLargeConfig:
        mod_layers: list = field(default_factory=lambda: list(range(24)))
        vocab_size: int = 50265
        hidden_size: int = 1024
        num_hidden_layers: int = 24
        num_attention_heads: int = 16
        intermediate_size: int = 4096
        max_position_embeddings: int = 514
        layer_norm_eps: float = 1e-05
        num_class_labels: int = 1
        pad_token_id: int = 1

        # Special Configs 
        rank: int = None
        attn_type: str = 'spda'
        use_bottleneck: bool = False
        bottleneck_size: int = None
        prefix_size: int = None
        use_prefix: bool = False

    @dataclass 
    class RobertaBaseConfig:
        mod_layers: list = field(default_factory=lambda: list(range(12)))
        vocab_size: int = 50265
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        max_position_embeddings: int = 514
        layer_norm_eps: float = 1e-05
        num_class_labels: int = 1
        pad_token_id: int = 1

        # Special Configs 
        rank: int = None
        attn_type: str = 'spda'
        use_bottleneck: bool = False
        bottleneck_size: int = None
        prefix_size: int = None
        use_prefix: bool = False

    model = None 
    prefix_size = 0
    size = 'base'


    # if model_name == 'lora':
    #     model = RobertaClassificationAndLM.from_ethics_pretrained(RobertaLargeConfig(attn_type = 'lora_spda', rank = 4, mod_layers= list(range(18,24))), size = size)
    #     lr = 1e-3
    # elif model_name == 'base_model':
    #     model =RobertaClassificationAndLM.from_ethics_pretrained(RobertaLargeConfig(), size = size)
    #     lr = 1e-5
    # elif model_name == 'adapter':
    #     model = RobertaClassificationAndLM.from_ethics_pretrained(RobertaLargeConfig(use_bottleneck = True, bottleneck_size = 8, mod_layers= list(range(18,24))), size = size)
    #     lr = 1e-3
    # elif model_name == 'prefix':
    #     model = RobertaClassificationAndLM.from_ethics_pretrained(RobertaLargeConfig(use_prefix = True, prefix_size = 64), size = size)
    #     prefix_size = 64
    #     lr = 1e-3

    if model_name == 'lora':
        model = RobertaClassificationAndLM.from_ethics_pretrained(RobertaBaseConfig(attn_type = 'lora_spda', rank = 4, mod_layers= list(range(6,12))), size = size)
        lr = 1e-3
    elif model_name == 'base_model':
        model =RobertaClassificationAndLM.from_ethics_pretrained(RobertaBaseConfig(), size = size)
        lr = 1e-4
    elif model_name == 'adapter':
        model = RobertaClassificationAndLM.from_ethics_pretrained(RobertaBaseConfig(use_bottleneck = True, bottleneck_size = 8, mod_layers= list(range(6,12))), size = size)
        lr = 1e-3
    elif model_name == 'prefix':
        model = RobertaClassificationAndLM.from_ethics_pretrained(RobertaBaseConfig(use_prefix = True, prefix_size = 64), size = size)
        prefix_size = 64
        lr = 1e-3

    if model_name == 'lora':
        for name, param in model.named_parameters():
            if "lora" not in name: 
                param.requires_grad = False
                
    if model_name == 'adapter':
        for name, param in model.named_parameters():
            if "bottleneck" not in name and 'LayerNorm2' not in name:
                param.requires_grad = False
    
    if model_name == 'prefix':
        for name, param in model.named_parameters():
            if "prefix" not in name: 
                param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if "lm_head" in name or 'classification' in name: 
    #         param.requires_grad =  True
    for name, param in model.named_parameters():
        if 'classification' in name or 'lm_head' in name: 
            param.requires_grad =  True

    ####################################
    #    Train Models and Save Results #
    ####################################

    epochs = 10

    print(model_name)
    s = time.time()
    model.to(device)

    results = train_model(model, 
                          epochs, 
                          train_loader_moral, 
                          val_loader_moral, 
                          val_loader_moral_unmasked, 
                          test_loader_wikitext,
                          morality_probing_dataset,
                          f'{size}/{model_name}_{max_training_time}',
                          device,
                          tokenizer,
                          max_training_time= max_training_time,
                          prefix_size = prefix_size,
                          lr = lr)

    print(f'Total time {time.time() - s}.')


    with open(f'results/{size}/{model_name}_{max_training_time}.json', 'w') as f:
        json.dump(results, f)
