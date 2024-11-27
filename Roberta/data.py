import random
import torch
import torch.nn as nn
from transformers import RobertaTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EthicsDataset(Dataset):
    def __init__(self, split, max_seq_len = 128):
        super().__init__()

        self.tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
        
        # Fetch Ethics data
        self.commonsense = load_dataset("hendrycks/ethics", "commonsense")
        self.deontology = load_dataset("hendrycks/ethics", "deontology")
        self.justice = load_dataset("hendrycks/ethics", "justice") 

        # Properties
        self.invert_labels = False

        self.max_seq_len = max_seq_len

        self.masked_seqs = []
        self.masked_labels = []
        self.cls_labels = []
        
        self.create_dataset(split)

    def __len__(self):
        return len(self.masked_seqs)
        
    def pad(self, seq, max_len, padding_token = 1):
        while len(seq) < max_len:
            seq.append(padding_token)
        return seq

    def retrieve_raw_data(self, dataset, split, keys):
        masked_seqs = []
        cls_labels = []
        
        for row in dataset[split]: 
            x = ""
            for key in keys: 
                x += row[key] + " " 
            x = x.strip()
            masked_seqs.append(x)
            cls_labels.append(int(not(row['label'])) if self.invert_labels else row['label'])

        return masked_seqs, cls_labels

    def tokenize_and_mask_sequence(self, sequence): 
        '''
        Replace 15% of tokens
        - 80% will be replaced with <mask> 
        - 10% will be replaced with random token
        - 10% will be unchanged
        
        I may omit random token masking for now and introduce later in training to see if it helps 
        '''
        
        tokens = self.tokenizer.encode(sequence)[1:-1]
        
        label = [] # O if token not replaced, token_id is token is replace with <mask>
        
        output_sequence = [] # sequence of tokens with some tokens masked out
        
        for token in tokens:
            prob = random.random()
        
            # Replace word
            if prob < 0.50:
                prob/= 0.50
        
                # 80% chance token will be masked out
                if prob < 0.75: 
                    output_sequence.append(token)
        
                # 10% chance token will be replaced with random tokens
                elif prob < 0.95:
                    # output_sequence.append(random.randrange(len(self.tokenizer.get_vocab())))
                    output_sequence.append(self.tokenizer.get_vocab()['<mask>'])
        
                # 10% chance for no replacement
                else:
                    # output_sequence.append(random.randrange(len(self.tokenizer.get_vocab())))
                    output_sequence.append(token)
                label.append(token)
                
            else:
                output_sequence.append(token)
                label.append(0)

        # Replace the <s> and </s> tokens 
        output_sequence = [self.tokenizer.get_vocab()['<s>']] + output_sequence + [self.tokenizer.get_vocab()['</s>']]
        label = [0] + label + [0]
        return output_sequence, label

    def create_dataset(self, split):

        ##########################
        #### Collect raw data ####
        ##########################
        
        raw_seqs = []
        raw_cls = []

        # Commonsense
        data_x, data_y = self.retrieve_raw_data(self.commonsense, split = split, keys = ['input'])
        raw_seqs = raw_seqs + data_x
        raw_cls = raw_cls + data_y

        # Justice
        data_x, data_y = self.retrieve_raw_data(self.justice, split = split, keys = ['scenario'])
        raw_seqs = raw_seqs + data_x
        raw_cls = raw_cls + data_y

        # Deontology
        data_x, data_y = self.retrieve_raw_data(self.deontology, split = split, keys = ['scenario', 'excuse'])
        raw_seqs = raw_seqs + data_x
        raw_cls = raw_cls + data_y

        ##########################
        ####    Mask  Data    ####
        ##########################

        for data in zip(raw_seqs, raw_cls):
            seq = data[0]
            cls = data[1]
            
            s, l = self.tokenize_and_mask_sequence(seq)

            s = s[0: self.max_seq_len]
            l = l[0: self.max_seq_len]

            s = self.pad(s, self.max_seq_len)
            l = self.pad(l, self.max_seq_len, padding_token = 0)

            # Convert to tensor
            s = torch.tensor(s)
            l = torch.tensor(l)
            cls = torch.tensor(cls)
            
            self.masked_seqs.append(s)
            self.masked_labels.append(l)
            self.cls_labels.append(cls)
        
    def __getitem__(self, idx):
        output = {
            "x" : self.masked_seqs[idx],
            "y_lm" : self.masked_labels[idx],
            "y_cls"  : self.cls_labels[idx]
        }

        return output
    
class MoralStoriesDataset(Dataset):
    def __init__(self, split, max_seq_len = 128, mask_data = True, moral_prefix = True):
        super().__init__()

        self.tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
        
        # Fetch Ethics data
        self.moral_stories = load_dataset("demelin/moral_stories", "cls-action+context+consequence-lexical_bias")

        # Properties
        self.invert_labels = False
        self.moral_prefix = moral_prefix
        self.mask_data = mask_data

        self.max_seq_len = max_seq_len

        self.moral_token = self.tokenizer.encode(' moral')[1:-1][0]
        self.immoral_token = self.tokenizer.encode(' immoral')[1:-1][0]
        self.cls_token = self.tokenizer.encode('<s>')[1:-1][0]
        self.eos_token = self.tokenizer.encode('</s>')[1:-1][0]
        self.moral_prefix = self.tokenizer.encode("This is<mask>: ")[1:-1]

        self.masked_seqs = []
        self.masked_labels = []
        self.cls_labels = []
        
        self.create_dataset(split)

    def __len__(self):
        return len(self.masked_seqs)
        
    def pad(self, seq, max_len, padding_token = 1):
        while len(seq) < max_len:
            seq.append(padding_token)
        return seq

    def retrieve_raw_data(self, dataset, split, keys):
        masked_seqs = []
        cls_labels = []
        
        for row in dataset[split]: 
            x = ""
            for key in keys: 
                x += row[key] + " " 
            x = x.strip()
            masked_seqs.append(x)
            cls_labels.append(int(not(row['label'])) if self.invert_labels else row['label'])

        return masked_seqs, cls_labels

    def tokenize_and_mask_sequence(self, sequence): 
        '''
        Replace 15% of tokens
        - 80% will be replaced with <mask> 
        - 10% will be replaced with random token
        - 10% will be unchanged
        
        I may omit random token masking for now and introduce later in training to see if it helps 
        '''
        
        tokens = self.tokenizer.encode(sequence)[1:-1]
        
        label = [] # O if token not replaced, token_id is token is replace with <mask>
        
        output_sequence = [] # sequence of tokens with some tokens masked out
        
        for token in tokens:
            prob = random.random()
        
            # Replace word
            if prob < 0.50 and self.mask_data:
                prob/= 0.50
        
                # 80% chance token will be masked out
                if prob < 0.75: 
                    output_sequence.append(token)
        
                # 10% chance token will be replaced with random tokens
                elif prob < 0.95:
                    # output_sequence.append(random.randrange(len(self.tokenizer.get_vocab())))
                    output_sequence.append(self.tokenizer.get_vocab()['<mask>'])
        
                # 10% chance for no replacement
                else:
                    # output_sequence.append(random.randrange(len(self.tokenizer.get_vocab())))
                    output_sequence.append(token)
                label.append(token)
                
            else:
                output_sequence.append(token)
                label.append(0)

        # Replace the <s> and </s> tokens 
        # output_sequence = [self.tokenizer.get_vocab()['<s>']] + output_sequence + [self.tokenizer.get_vocab()['</s>']]
        # label = [0] + label + [0]
        return output_sequence, label

    def add_moral_prefix(self, s, l, cls):
        correct_pred = self.moral_token if cls == 1 else self.immoral_token
        prefix_l = [0, 0, correct_pred, 0, 0]

        s = self.moral_prefix + s
        l = prefix_l + l

        return s, l
        
    def create_dataset(self, split):

        ##########################
        #### Collect raw data ####
        ##########################
        
        raw_seqs = []
        raw_cls = []

        # Collect Raw Data
          
        for data in self.moral_stories[split]: 
            if(data['moral_action'] == 'not specified'):
                x = f"{data['situation']} {data['intention']} {data['immoral_action']} {data['immoral_consequence']}"  
            else:
                x = f"{data['situation']} {data['intention']} {data['moral_action']} {data['moral_consequence']}" 
            raw_seqs.append(x)
            raw_cls.append(int(not(data['label'])) if self.invert_labels else data['label'])

        ##########################
        ####    Mask  Data    ####
        ##########################

        for data in zip(raw_seqs, raw_cls):
            seq = data[0]
            cls = int(data[1])
            
            s, l = self.tokenize_and_mask_sequence(seq)

            if self.moral_prefix:
                s, l = self.add_moral_prefix(s, l, cls)

            s = s[0: self.max_seq_len - 2]
            l = l[0: self.max_seq_len - 2]

            s = [self.cls_token] + s + [self.eos_token]
            l = [0] + l + [0]

            s = self.pad(s, self.max_seq_len)
            l = self.pad(l, self.max_seq_len, padding_token = 0)

            # Convert to tensor
            s = torch.tensor(s)
            l = torch.tensor(l)
            cls = torch.tensor(cls)
            
            self.masked_seqs.append(s)
            self.masked_labels.append(l)
            self.cls_labels.append(cls)
        
    def __getitem__(self, idx):
        output = {
            "x" : self.masked_seqs[idx],
            "y_lm" : self.masked_labels[idx],
            "y_cls"  : self.cls_labels[idx]
        }

        return output


