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
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj_1 = nn.Linear(config.hidden_size, 2)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj_2 = nn.Linear(config.hidden_size, 2)
        self.dense3 = nn.Linear(config.hidden_size, int(config.hidden_size/2))
        self.out_proj_3 = nn.Linear(int(config.hidden_size/2), 2)
        


    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = x.reshape(-1,x.size(-1)*2)
        x = x.reshape(-1,x.size(-1))
        x = self.dropout(x)
        x = self.dense1(x)
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
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
    
        
    def forward(self, inputs_ids,position_idx,attn_mask,labels=None): 
        # bs,l=inputs_ids_1.size()
        # inputs_ids=torch.cat((inputs_ids_1.unsqueeze(1),inputs_ids_2.unsqueeze(1)),1).view(bs*2,l)
        # position_idx=torch.cat((position_idx_1.unsqueeze(1),position_idx_2.unsqueeze(1)),1).view(bs*2,l)
        # attn_mask=torch.cat((attn_mask_1.unsqueeze(1),attn_mask_2.unsqueeze(1)),1).view(bs*2,l,l)

        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx,token_type_ids=position_idx.eq(-1).long())[0]
        
        logits1, logits2, logits3 =self.classifier(outputs)
        # shape: [batch_size, num_classes]
        prob1 =F.softmax(logits1, dim=-1)
        prob2 =F.softmax(logits2, dim=-1)
        prob3 =F.softmax(logits3, dim=-1)
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss_fct = FocalLoss(gamma=2,alpha=0.39)
            loss1 = loss_fct(logits1, labels)
            loss2 = loss_fct(logits2, labels)
            loss3 = loss_fct(logits3, labels)
        
            loss = 0.05*loss1 + 0.35*loss2 + 0.6*loss3
            prob = 0.05*prob1 + 0.35*prob2 + + 0.6*prob3
            
            return loss, prob
        else:
            prob = prob3
            return prob
      
      
