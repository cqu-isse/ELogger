# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
        
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj_1 = nn.Linear(config.hidden_size, 2)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj_2 = nn.Linear(config.hidden_size, 2)
        self.dense3 = nn.Linear(config.hidden_size, int(config.hidden_size/2))
        self.out_proj_3 = nn.Linear(int(config.hidden_size/2), 2)
        
    def forward(self, x):
        # x = x.reshape(-1,x.size(-1)*2)
        x = x.reshape(-1,x.size(-1))
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x1 = self.out_proj_1(x)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x2 = self.out_proj_2(x)
        x = self.dense3(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x3 = self.out_proj_3(x)
        return x1, x2, x3    
    
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
    
        
    def forward(self, input_ids=None, labels=None):
        # Reshape input_ids
        input_ids = input_ids.view(-1, self.args.block_size)
        
        # Encode input_ids using the RoBERTa encoder
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        
        # Get the representation of the CLS token
        cls_representation = outputs[:, 0, :]
        
        # Pass the CLS representation through the classifier
        logits1, logits2, logits3=self.classifier(cls_representation)
        prob1 =F.softmax(logits1, dim=-1)
        prob2 =F.softmax(logits2, dim=-1)
        prob3 =F.softmax(logits3, dim=-1)

        if labels is not None:
            # Compute the loss using CrossEntropyLoss
            loss_fct = FocalLoss(gamma=2,alpha=0.39)
            loss1 = loss_fct(logits1, labels)
            loss2 = loss_fct(logits2, labels)
            loss3 = loss_fct(logits3, labels)
        
            loss = 0.05*loss1 + 0.35*loss2 + 0.6*loss3
            prob = 0.05*prob1 + 0.35*prob2 + 0.6*prob3
            return loss, prob
        
        else:
            # Return logits if no labels are provided
            prob = prob3
            return prob
        