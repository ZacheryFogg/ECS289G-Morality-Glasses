import torch
from torch.nn import functional as F

padding_idx = 1
cls_idx = 0
vocab_size = 50265

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

def calculate_loss(model, data, prefix_size = 0, is_val = False):
     
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
        if is_val: 
            with torch.no_grad():
                token_preds_logits, cls_pred , _ = model(x, attention_mask = attn_mask, run_lm_head = True)
        else: 
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
