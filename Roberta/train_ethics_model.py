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

if __name__ == "__main__":

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

    train_dataset_ethics = torch.load('./Datasets/ethics/train_dataset_ethics.pt', weights_only=False)
    val_dataset_ethics = torch.load('./Datasets/ethics/val_dataset_ethics.pt', weights_only=False)
    test_dataset_ethics = torch.load('./Datasets/ethics/test_dataset_ethics.pt',weights_only=False)
    val_datatset_ethics_unmasked = torch.load('./Datasets/ethics/val_dataset_ethics_unmasked.pt', weights_only = False)

    batch_size = 32

    train_loader_ethics = DataLoader(train_dataset_ethics, batch_size = batch_size, shuffle = False)
    val_loader_ethics = DataLoader(val_dataset_ethics, batch_size = batch_size, shuffle = False)
    test_loader_ethics = DataLoader(test_dataset_ethics, batch_size = batch_size, shuffle = False)
    val_loader_ethics_unmasked = DataLoader(val_datatset_ethics_unmasked, batch_size=batch_size, shuffle=False)

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

    prefix_size = 0
    lr = 1e-4

    model = RobertaClassificationAndLM.from_pretrained(RobertaBaseConfig(), size = 'base')


    # Try freezing nothing, and see what happens 
    
    # Try freezing all but classificatio head, see what happens : bad, nothing of use really
    # for name, param in model.named_parameters():
    #     if 'classification' not in name and 'lm_head' not in name: 
    #         param.requires_grad = False

    # Try freezing only LM head, see what happens
    # for name, param in model.named_parameters():
    #     if 'lm_head' in name:
    #         param.requires_grad = False

    for name, param in model.named_parameters():
        if 'classification' not in name: 
            param.requires_grad = False

    ####################################
    #    Train Models and Save Results #
    ####################################

    epochs = 1
    max_training_time = -1

    s = time.time()
    model.to(device)

    results = train_model(model, 
                          epochs, 
                          train_loader_ethics, 
                          val_loader_ethics, 
                          val_loader_ethics_unmasked, 
                          test_loader_wikitext,
                          morality_probing_dataset,
                          'base/base_ethics_model_cls_only',
                          device,
                          tokenizer,
                          max_training_time= max_training_time,
                          prefix_size = prefix_size,
                          lr = lr)
    print(f'Total time {time.time() - s}')

    with open(f'results/base/base_ethics_model_cls_only.json', 'w') as f:
        json.dump(results, f)
