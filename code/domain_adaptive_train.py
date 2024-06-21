
from torch.cuda.amp import autocast as autocast, GradScaler
import torch
import torch.nn as nn
import numpy as np
import os,sys
import torch.optim as optim
import esm
import mask_loss
import fea_process as fea

fasta_path=sys.argv[1]
device=sys.argv[2]
batch_size=int(sys.argv[3])
device_list=list(map(int,sys.argv[4].split(',')))
epochs=int(sys.argv[5])

need_token_len=512


esm_model, alphabet=esm.pretrained.esm2_t33_650M_UR50D()
para_list=[".{0}.".format(i) for i in range(29)]
for name, param in esm_model.named_parameters():
    if name=="embed_tokens.weight":
        param.requires_grad = False
        continue
    for one in para_list:
        if one in name:
            param.requires_grad=False
            break

print("requires_grad layers:")
for one in filter(lambda p: p.requires_grad, esm_model.parameters()):
    print(one)

batch_converter = alphabet.get_batch_converter()
esm_model=nn.DataParallel(esm_model,device_ids=device_list).to(device)
esm_model.train()
loss=mask_loss.BERT_MLM_Loss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, esm_model.parameters()), lr=0.0001,betas=(0.9, 0.98), eps=1e-08, weight_decay=1e-2)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,15],gamma = 0.9)

dicts=fea.fasta2dict(fasta_path)
#data_loader
token_lists=[]
masked_list=[]

for one_pro,one_seq in zip(dicts.keys(),dicts.values()):
    token=fea.token_represent(one_pro,one_seq,need_token_len,alphabet.tok_to_idx)
    token.random_mask(token.token_list_represent, 0.15)
    token_lists+=token.list_token_to_str()
    masked_list+=token.masked_position_lab

loader=fea.data_loader(token_lists,masked_list,batch_size)

print("data load over, len of all seqs: {0}".format(loader.len))

if __name__=='__main__':
    epoch_loss = []
    scaler = GradScaler()
    for epoch in range(epochs):
        one_epoch_loss = 0
        batch_num=0
        all_masked_sum=0
        all_right_value=0
        for data, masked_pl in loader:
            batch_num+=batch_size
            if batch_num%100000==0:print(batch_num)
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            optimizer.zero_grad()
            with autocast():
                results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            # batch*len*33
                pred = results["logits"]
                pred=pred[:,1:-1,:]
                mean_loss = loss(pred, masked_pl, device)
            scaler.scale(mean_loss).backward() 

            scaler.step(optimizer)
            scaler.update()
            one_epoch_loss += mean_loss.item()
            pred_value=torch.argmax(pred,dim=2)
            index=0
            for one_position,one_value in masked_pl:
                all_masked_sum+=len(one_position)
                right=torch.sum(pred_value[index][one_position]==torch.tensor(one_value).to(device))
                all_right_value+=right
                index+=1
        scheduler.step()
        epoch_loss.append(one_epoch_loss)
        perplexity=np.exp(one_epoch_loss/all_masked_sum)
        print("epoch:[{0}] -->Loss:{1:.2f}, ACC:{2:.4f}, perplexity:{3:.4f}".format(epoch, one_epoch_loss,all_right_value/all_masked_sum,perplexity))

    torch.save(esm_model.state_dict(),'./ESM_DBP_PLM.model')