# -*- coding: UTF-8 -*-
'''=================================================
@Project -> ContactCry    ：FWorks -> model
@IDE    ：PyCharm
@Author : wenwuZeng
@Date   ：2023/8/7 16:30
=================================================='''

import torch.nn as nn
import torch


class_num=33
pred=torch.rand(2,10,33)
#target = torch.empty((2, 100), dtype=torch.long).random_(class_num)


class BERT_MLM_Loss(nn.CrossEntropyLoss):
    #pred shape:(batch*seq_len*33)
    #masked_position_lab:[([1,2,3],[])]
    def forward(self, pred, masked_position_lab,device):
        #self.reduction = 'sum'
        self.reduction = 'none'
        target=torch.zeros(pred.shape[0],pred.shape[1]).to(device).long()
        weight = torch.zeros_like(target).to(device)
        for index, one in enumerate(masked_position_lab):
            position,one_true_lab=torch.tensor(one[0]).to(device),torch.tensor(one[1]).to(device)
            weight[index][position]=1
            target[index][position]=one_true_lab
        unweighted_loss = super().forward(pred.permute(0,2,1), target)
        weighted_loss=unweighted_loss*weight
        return weighted_loss.sum()/weight.sum()


if __name__ == '__main__':
    loss=BERT_MLM_Loss()
    masked_position_lab=[([0,1,2,5],[15,20,30,31]),([8,9,7],[24,28,4])]
    res_loss=loss(pred,masked_position_lab,'cpu')
    print(res_loss)