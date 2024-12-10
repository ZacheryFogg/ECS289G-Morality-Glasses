import torch
from torch.nn import functional as F
import time
from pynvml import *
from tqdm.auto import tqdm
import numpy as np
import gc
from data import morality_classification_examples


padding_idx = 1
cls_idx = 0
vocab_size = 50265

moral_pref = "This is moral: "
immoral_pref = "This is immmoral: "
neutral_pref  = "This is neutral: "


def calculate_morality_classification(model, device, tokenizer, prefix_size):

    cls_correct = 0
    for data in morality_classification_examples:
        x = data['x']
        x = tokenizer.encode(x)
        x = torch.tensor(x).unsqueeze(0).to(device)
    
        y = torch.tensor(data['y'])
        y = y.to(device).float()

        attn_mask = create_attention_mask(x, device, dtype = torch.bfloat16, prefix_size = prefix_size)
        attn_mask = attn_mask.to(torch.float32)

        with torch.autocast(device_type = device, dtype = torch.bfloat16) and torch.no_grad():
            _, cls_pred , _ = model(x, attention_mask = attn_mask, run_lm_head = False)

            # Calculate CLS Pred Loss
            cls_pred_unsqz = cls_pred.squeeze()

            cls_preds = (F.sigmoid(cls_pred) > .5).squeeze()
            
            cls_correct += (cls_preds == y).sum().item()

    return cls_correct / len(morality_classification_examples) * 100


def get_probs(model, x, device, tokenizer, prefix_size):
    x = tokenizer.encode(x)
    x = torch.tensor(x).unsqueeze(0).to(device)
    
    attn_mask = create_attention_mask(x, device, dtype = torch.bfloat16, prefix_size= prefix_size)
    attn_mask = attn_mask.to(torch.float32)

    with torch.no_grad() and torch.autocast(device_type = device, dtype = torch.bfloat16):
        logits, _, _ = model(x, attention_mask = attn_mask, run_lm_head = True)

    mask_token_index = (x == tokenizer.mask_token_id)[0].nonzero(as_tuple = True)[0]

    probs = F.softmax(logits[0, mask_token_index], dim=1)

    return probs 

def collect_ratios(model, data, device, tokenizer, prefix_size = 0):
    seq = data["Seq"]
    moral_token = tokenizer.encode(data["Moral"])[1:-1]
    immoral_token = tokenizer.encode(data["Immoral"])[1:-1]

    if len(moral_token) != 1 or len(immoral_token) != 1: 
        print(f'Could not encoder targets to single token: {data}' )
        return None, None, None
    
    moral_token = moral_token[0]
    immoral_token = immoral_token[0]

    moral_probs = get_probs(model, moral_pref + seq, device, tokenizer, prefix_size= prefix_size).view(-1)
    immoral_probs = get_probs(model, immoral_pref + seq, device, tokenizer, prefix_size= prefix_size).view(-1)
    neutral_probs = get_probs(model, neutral_pref + seq, device, tokenizer, prefix_size= prefix_size).view(-1)

    # How much more likely model thinks that moral token is compared to the immoral token

    # Ratio for model that was hopefully conditioned to choose a produce moral text
    moral_ratio = (moral_probs[moral_token] / moral_probs[immoral_token]).item() 
    
    # Ratio for model that was hopefully conditioned to choose a produce immoral text
    immoral_ratio = (immoral_probs[moral_token] / immoral_probs[immoral_token]).item()

    # Ratio for model that was not conditioned to produce moral or immoral text
    neutral_ratio = (neutral_probs[moral_token] / neutral_probs[immoral_token]).item()

    return moral_ratio, neutral_ratio, immoral_ratio

def get_top_k_preds(model, x, device, tokenizer, prefix_size = 0, k = 5):
    probs = get_probs(model, x, device, prefix_size)
    
    topk = torch.topk(probs, k)

    topk = [(tokenizer.decode(topk.indices.squeeze()[i].item()), round(topk.values.squeeze()[i].item() * 100, 2)) for i in range(topk.indices.shape[1])]
    
    return topk


def moral_prediction_accuracy(model, dataset, device, tokenizer, prefix_size = 0):

    correct = 0

    for data in dataset:
        moral_ratio, neutral_ratio, immoral_ratio = collect_ratios(model, data, device, tokenizer, prefix_size)

        '''
        If model has learned to output moral/immmoral text, then it moral_ratio should be higher then neutral since the model should     
        prefer the moral word more than it did before and prefer the immoral word less than is did before 

        Immoral_ratio should be less than neutral ratio since the model should prefer the moral word less than it did before and 
        prefer the immoral word more than it did before

        This intuition is expressed in this equality 
        '''

        if moral_ratio > neutral_ratio and neutral_ratio > immoral_ratio:
            correct +=1 

    return round((correct  / len(dataset)) * 100, 2)


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def get_gpu_mem_usage():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used//1024**2



def create_attention_mask(x, device, padding_idx = 1, dtype = torch.float, prefix_size = 0):

    if prefix_size != 0:
        prefix_dummy_data = torch.zeros(x.shape[0], prefix_size).to(device)
        x = torch.cat((prefix_dummy_data, x), dim = 1).to(dtype)

    mask = (x != padding_idx)

    bsz, slen = mask.size()
    
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, slen, slen).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def calculate_accuracy_loss(model, dataset, device, prefix_size = 0):

    cls_correct = 0
    moral_token_correct = 0
    moral_token_index = 3
    moral_token = 7654
    immoral_token = 33231
    total = 0

    
    with torch.no_grad():
        for data in dataset:
            x, y_lm, y_cls = data['x'], data['y_lm'], data['y_cls']

            y_lm = F.one_hot(y_lm, num_classes = vocab_size).float()
            y_lm[:,:,0] =  y_lm[:,:,0] * 0 # Set target of all 0 tokens to 0 vector so no loss contribution
            y_lm = y_lm.to(device)

            x = x.to(device)
            y_cls = y_cls.to(device).float()

            y_moral = y_cls.clone()
            for i in range(y_moral.size()[0]):
                if y_moral[i] == 1:
                    y_moral[i] = moral_token
                else: 
                    y_moral[i] = immoral_token
    
            attn_mask = create_attention_mask(x, device, dtype = torch.bfloat16, prefix_size = prefix_size)
            attn_mask = attn_mask.to(torch.float32)

            with torch.autocast(device_type = device, dtype = torch.bfloat16) and torch.no_grad():
                token_preds_logits, cls_pred , _ = model(x, attention_mask = attn_mask, run_lm_head = True)
            
                # Calculate LM Loss 
                token_preds_logits = token_preds_logits.view(-1, token_preds_logits.size(-1)) # Flatten logits to (B * T, Vocab_Size)
                y_lm = y_lm.view(-1, y_lm.size(-1)) # Flatten targets to (B * T, Vocab_Size)
                lm_loss = F.cross_entropy(token_preds_logits, y_lm)

                # Calculate CLS Pred Loss
                cls_pred_unsqz = cls_pred.squeeze()
                cls_loss = F.binary_cross_entropy_with_logits(cls_pred_unsqz, y_cls)

            cls_preds = (F.sigmoid(cls_pred) > .5).squeeze()
            
            cls_correct += (cls_preds == y_cls).sum().item()

            # Calculate if model correctly predicted moral and immoral
            token_preds_logits = token_preds_logits.view(x.shape[0], x.shape[1], vocab_size)
            moral_preds_logits = token_preds_logits[:,moral_token_index,:] # Retrieve just the token preds corresponsing to the moral <mask> tokens
            moral_preds = moral_preds_logits.argmax(dim = -1) # Retrieve the models predictions for the <mask> tokens

            moral_token_correct += (moral_preds == y_moral).sum().item()
            
            total += y_cls.size(0)

    return (cls_correct / total) * 100, (moral_token_correct / total) * 100, lm_loss.item(), cls_loss.item()

def calculate_loss(model, data, device, prefix_size = 0):
     
    x, y_lm, y_cls = data['x'], data['y_lm'], data['y_cls']

    # One hot encode LM targets 
    y_lm = F.one_hot(y_lm, num_classes = 50265).float()
    y_lm[:,:,0] =  y_lm[:,:,0] * 0 # Set target of all 0 tokens to 0 vector so no loss contribution

    # Move data to device
    y_lm = y_lm.to(device)
    x = x.to(device)
    y_cls = y_cls.to(device).float()

    # Attention Mask
    attn_mask = create_attention_mask(x, device, dtype = torch.bfloat16, prefix_size = prefix_size)
    attn_mask = attn_mask.to(torch.float32)

    with torch.autocast(device_type = device, dtype = torch.bfloat16):

        token_preds_logits, cls_pred , _ = model(x, attention_mask = attn_mask, run_lm_head = True)

        # Calculate LM Loss 
        token_preds_logits = token_preds_logits.view(-1, token_preds_logits.size(-1)) # Flatten logits to (B * T, Vocab_Size)
        y_lm = y_lm.view(-1, y_lm.size(-1)) # Flatten targets to (B * T, Vocab_Size)

        lm_loss = F.cross_entropy(token_preds_logits, y_lm)

        # Calculate CLS Pred Loss
        cls_pred = cls_pred.squeeze()
        cls_loss = F.binary_cross_entropy_with_logits(cls_pred, y_cls)
        lm_loss + cls_loss

    return lm_loss, cls_loss

def print_token_from_logits(logits, tokenizer):

    for i in range(logits.size()[0]):
        probs = F.softmax(logits[i])
        pred_idx = probs.argmax(-1)
        print(tokenizer.decode(pred_idx))


def calculate_wikitext_loss(model, wikitext_loader, device, prefix_size = 0):
    running_loss = 0

    for data in wikitext_loader:
        x, y = data['x'], data['y']


        y = F.one_hot(y, num_classes = 50265).float()
        y[:,:,0] =  y[:,:,0] * 0 # Set target of all 0 tokens to 0 vector so no loss contribution
        
        y = y.to(device)
        x = x.to(device)

        attn_mask = create_attention_mask(x, device, dtype = torch.bfloat16, prefix_size = prefix_size)
        attn_mask = attn_mask.to(torch.float32)

        with torch.autocast(device_type = device, dtype = torch.bfloat16) and torch.no_grad():
            token_preds_logits, _ , _ = model(x, attention_mask = attn_mask, run_lm_head = True, run_classification_head = False)

        token_preds_logits = token_preds_logits.view(-1, token_preds_logits.size(-1)) # Flatten logits to (B * T, Vocab_Size)
        y = y.view(-1, y.size(-1)) # Flatten targets to (B * T, Vocab_Size)


        loss = F.cross_entropy(token_preds_logits, y)
        running_loss += loss.item()

    return running_loss / len(wikitext_loader)

def train_model(model, 
                num_epochs, 
                train_loader, 
                val_loader, 
                val_loader_unmasked,
                test_loader_wikitext,
                morality_probing_dataset,
                save_key, 
                device, 
                tokenizer,
                max_training_time = -1, 
                prefix_size = 0, 
                lr = 1e-4, 
                save_best_model = True):

    model.to(device)
    
    min_val_loss = np.inf

    optimizer = torch.optim.AdamW(model.parameters(), lr)

    train_losses_lm = []
    train_losses_cls = []

    val_losses_lm = []
    val_losses_cls = []
    val_cls_accs = []
    val_moral_token_accs = []

    val_moral_token_accs_unmasked = []
    val_cls_accs_unmasked = []
    val_losses_cls_unmasked = []


    training_mem_usage = []
    gpu_utilization = []

    val_wikitext_losses = []
    val_moral_completion_accs = []

    val_morality_classifcation_accs = []
    
    
    # Calculate accuracy and loss for validation sets before any training
    # _, _, lm_loss_t, cls_loss_t = calculate_accuracy_loss(model, train_loader, device, prefix_size)
    cls_acc_v, moral_tokens_acc_v, lm_loss_v, cls_loss_v = calculate_accuracy_loss(model, val_loader, device, prefix_size)
    
    # Track metrics 
    # train_losses_lm.append(lm_loss_t), train_losses_cls.append(cls_loss_t)
    val_losses_lm.append(lm_loss_v), val_losses_cls.append(cls_loss_v), val_cls_accs.append(cls_acc_v), val_moral_token_accs.append(moral_tokens_acc_v)

    cls_acc_v_unmasked, moral_tokens_accs_v_unmasked, _, cls_loss_v_unmasked = calculate_accuracy_loss(model, val_loader_unmasked, device, prefix_size=prefix_size)
    val_losses_cls_unmasked.append(cls_loss_v_unmasked), val_cls_accs_unmasked.append(cls_acc_v_unmasked), val_moral_token_accs_unmasked.append(moral_tokens_accs_v_unmasked)

    # Validate model on WikiText and MoralTokenPrediction tasks 
    val_wikitext_losses.append(calculate_wikitext_loss(model, test_loader_wikitext, device, prefix_size=prefix_size))
    val_moral_completion_accs.append(moral_prediction_accuracy(model, morality_probing_dataset, device, tokenizer, prefix_size= prefix_size))
    
    val_morality_classifcation_accs.append(calculate_morality_classification(model, device, tokenizer, prefix_size))

    elapsed_training_time = 0 

    batch_num = 0
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')
        # Train model + Collect Metrics 

        for data in tqdm(train_loader):
            batch_start_time = time.time()
            
            batch_num +=1 
            
            optimizer.zero_grad()

            lm_loss, cls_loss = calculate_loss(model, data, device, prefix_size)

            loss = lm_loss + cls_loss 

            loss.backward()
            optimizer.step()

            train_losses_lm.append(lm_loss.item()), train_losses_cls.append(cls_loss.item())

            # Track GPU memory usage
            training_mem_usage.append(get_gpu_mem_usage())
            
            # Track GPU Utilization 
            gpu_util = torch.cuda.utilization(torch.device('cuda'))
            gpu_utilization.append(gpu_util)

            # Stop early if training time exceeded
            batch_elapsed_time = time.time() - batch_start_time
            elapsed_training_time += batch_elapsed_time

            if max_training_time > 0:
                if elapsed_training_time > max_training_time:
                    break

        # Validate model + Collect Metrics
        cls_acc_v, moral_tokens_acc_v, lm_loss_v, cls_loss_v = calculate_accuracy_loss(model, val_loader, device, prefix_size)   
        val_losses_lm.append(lm_loss_v), val_losses_cls.append(cls_loss_v), val_cls_accs.append(cls_acc_v), val_moral_token_accs.append(moral_tokens_acc_v)

        cls_acc_v_unmasked, moral_tokens_accs_v_unmasked, _, cls_loss_v_unmasked = calculate_accuracy_loss(model, val_loader_unmasked, device, prefix_size=prefix_size)
        val_losses_cls_unmasked.append(cls_loss_v_unmasked), val_cls_accs_unmasked.append(cls_acc_v_unmasked), val_moral_token_accs_unmasked.append(moral_tokens_accs_v_unmasked)

        # Validate model on WikiText and MoralTokenPrediction tasks 
        val_wikitext_losses.append(calculate_wikitext_loss(model, test_loader_wikitext, device, prefix_size=prefix_size))
        val_moral_completion_accs.append(moral_prediction_accuracy(model, morality_probing_dataset, device, tokenizer, prefix_size= prefix_size))
        val_morality_classifcation_accs.append(calculate_morality_classification(model, device, tokenizer, prefix_size))
        # Report Validation Metrics
        print(f'Val | CLS Acc: {cls_acc_v:.4} | Moral Acc: {round(moral_tokens_acc_v, 3)} | LM Loss {round(lm_loss_v, 5)} | CLS Loss {round(cls_loss_v, 5)}')
        
        # Save Best Model
        val_loss = lm_loss_v + cls_loss_v

        if val_loss < min_val_loss:
            min_val_loss = val_loss

            torch.save(model.state_dict(), f'./trained_models/{save_key}')
        
        # Stop early if training time exceeded
        if max_training_time > 0:
            if elapsed_training_time > max_training_time:
                print(f'Training time limit {max_training_time}s exceeded at {batch_num * train_loader.batch_size}/{len(train_loader) * train_loader.batch_size * (epoch + 1)} examples')
                break
            
    results_dict = {
        'train_losses_lm' : train_losses_lm,
        'train_losses_cls' : train_losses_cls,
        'val_losses_lm' : val_losses_lm,
        'val_losses_cls' : val_losses_cls,
        'val_cls_accs' : val_cls_accs,
        'val_moral_token_accs' : val_moral_token_accs,
        'training_mem_usage' : training_mem_usage,
        'gpu_utilization' : gpu_utilization,
        'val_wikitext_losses' : val_wikitext_losses,
        'val_moral_completion_accs' : val_moral_completion_accs,
        'val_moral_token_accs_unmasked' : val_moral_token_accs_unmasked,
        'val_cls_accs_unmasked' : val_cls_accs_unmasked,
        'val_losses_cls_unmasked' : val_losses_cls_unmasked,
        'val_morality_classifcation_accs' : val_morality_classifcation_accs
    }

    return results_dict
