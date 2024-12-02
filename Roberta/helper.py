import torch
from torch.nn import functional as F
import time
from pynvml import *
from tqdm.auto import tqdm
import numpy as np
import gc


padding_idx = 1
cls_idx = 0
vocab_size = 50265\


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

def train_model(model, num_epochs, train_loader, val_loader, save_key, device, max_training_time = -1, prefix_size = 0, lr = 1e-4, save_best_model = True):

    model.to(device)
    
    min_val_loss = np.inf

    optimizer = torch.optim.AdamW(model.parameters(), lr)

    train_losses_lm = []
    train_losses_cls = []

    val_losses_lm = []
    val_losses_cls = []
    val_cls_accs = []
    val_moral_token_accs = []

    training_mem_usage = []
    gpu_utilization = []
    
    
    # Calculate accuracy and loss for training and validation sets before any training
    # _, _, lm_loss_t, cls_loss_t = calculate_accuracy_loss(model, train_loader, device, prefix_size)
    # cls_acc_v, moral_tokens_acc_v, lm_loss_v, cls_loss_v = calculate_accuracy_loss(model, val_loader, device, prefix_size)
    
    # # Track metrics 
    # train_losses_lm.append(lm_loss_t), train_losses_cls.append(cls_loss_t)
    # val_losses_lm.append(lm_loss_v), val_losses_cls.append(cls_loss_v), val_cls_accs.append(cls_acc_v), val_moral_token_accs.append(moral_tokens_acc_v)

    start_time = time.time()
    batch_num = 0
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')
        # Train model + Collect Metrics 
        for data in tqdm(train_loader):
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
            if max_training_time > 0:
                elapsed_time = time.time() - start_time
                if elapsed_time > max_training_time:
                    break
            # Validate model + Collect Metrics
        cls_acc_v, moral_tokens_acc_v, lm_loss_v, cls_loss_v = calculate_accuracy_loss(model, val_loader, device, prefix_size)   
        val_losses_lm.append(lm_loss_v), val_losses_cls.append(cls_loss_v), val_cls_accs.append(cls_acc_v), val_moral_token_accs.append(moral_tokens_acc_v)

        # Report Validation Metrics
        print(f'Val | CLS Acc: {cls_acc_v:.4} | Moral Acc: {round(moral_tokens_acc_v, 3)} | LM Loss {round(lm_loss_v, 5)} | CLS Loss {round(cls_loss_v, 5)}')
        
        # Save Best Model
        val_loss = lm_loss_v + cls_loss_v

        if val_loss < min_val_loss:
            min_val_loss = val_loss

            torch.save(model.state_dict(), f'./trained_models/{save_key}')
        
        # Stop early if training time exceeded
        if max_training_time > 0:
            elapsed_time = time.time() - start_time
            if elapsed_time > max_training_time:
                print(f'Training time limit exceeded at {batch_num}/{len(train_loader) * (epoch + 1)} batches')
                break
            
    results_dict = {
        'train_losses_lm' : train_losses_lm,
        'train_losses_cls' : train_losses_cls,
        'val_losses_lm' : val_losses_lm,
        'val_losses_cls' : val_losses_cls,
        'val_cls_accs' : val_cls_accs,
        'val_moral_token_accs' : val_moral_token_accs,
        'training_mem_usage' : training_mem_usage,
        'gpu_utilization' : gpu_utilization

    }

    # Cleaing up resources 
    del model
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()

    return results_dict

def print_token_from_logits(logits, tokenizer):

    for i in range(logits.size()[0]):
        probs = F.softmax(logits[i])
        pred_idx = probs.argmax(-1)
        print(tokenizer.decode(pred_idx))
