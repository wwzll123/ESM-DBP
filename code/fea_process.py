# -*- coding: UTF-8 -*-
'''=================================================
@Project -> ESM-DBP    ：FWorks -> fea_process
@IDE    ：PyCharm
@Author : wenwuZeng
@Date   ：2023/8/4 10:16
=================================================='''

import random
import esm
import numpy as np


class token_represent():
    def __init__(self,pro_name,true_seq,need_token_len,convert_dict):
        self.token_list_represent=self.seq_to_eq_len_list(pro_name,true_seq,need_token_len)
        self.convert_dict=convert_dict
        self.masked_position_lab=[]

    def list_token_to_str(self):
        str_lists=[]
        for one in self.token_list_represent:
            str_lists.append((one[0],"".join(one[1])))
        return str_lists

    def seq_to_eq_len(self,pro_name,seq,need_token_len):
        seq_len=len(seq)
        if seq_len<=need_token_len:
            need_pad_len=need_token_len-seq_len
            return [(pro_name,need_pad_len*'<pad>'+seq,need_pad_len)]
        else:
            lists=[]
            i=0
            for index in range(0,seq_len,need_token_len):
                if index+need_token_len<=seq_len:
                    lists.append((pro_name+'_{0}'.format(i),seq[index:index+need_token_len],0))
                    i+=1
                else:
                    last_seq=seq[index:]
                    need_pad_len=need_token_len-len(last_seq)
                    lists.append((pro_name+'_{0}'.format(i),need_pad_len*'<pad>'+seq[index:],need_pad_len))
            return lists

    def seq_to_eq_len_list(self,pro_name,seq,need_token_len):
        seq_len=len(seq)
        if seq_len<=need_token_len:
            need_pad_len=need_token_len-seq_len
            return [(pro_name,need_pad_len*['<pad>']+list(seq),need_pad_len)]
        else:
            lists=[]
            i=0
            for index in range(0,seq_len,need_token_len):
                if index+need_token_len<=seq_len:
                    lists.append((pro_name+'_{0}'.format(i),list(seq[index:index+need_token_len]),0))
                    i+=1
                else:
                    last_seq=seq[index:]
                    need_pad_len=need_token_len-len(last_seq)
                    lists.append((pro_name+'_{0}'.format(i),need_pad_len*['<pad>']+list(seq[index:]),need_pad_len))
            return lists

    def random_mask(self,lists, rate):
        for one in lists:
            pro, seq_list, pad_len = one[0], one[1], one[2]
            true_seq = seq_list[pad_len:]
            mask_len = int(len(true_seq) * rate)
            if mask_len==0:mask_len=1
            res = random.sample(range(pad_len, len(seq_list), 1), k=mask_len)
            lab_list=[]

            for one_position in res:
                lab_list.append(self.convert_dict[seq_list[one_position]])
                if random.random() < 0.8:
                    seq_list[one_position] = '<mask>'
                elif random.random()<0.5:
                    seq_list[one_position]=random.choice(true_seq)

            self.masked_position_lab.append((res,lab_list))
        return lists


class data_loader():

    def __init__(self,featureList,masked_position_lab,batchSize,drop_last=False,shuffle=False):
        """
        :param featureList: [(pro,seq),()...]
        :param masked_position_lab: [([masked_position],[masked_lab]),(),...]
        """
        self.point=0
        self.batchSize=batchSize

        self.drop_last=drop_last
        if shuffle:
            self.list,self.lab=self.shuffle(featureList,masked_position_lab)
        else:
            self.list, self.lab = featureList, masked_position_lab
        self.len = len(self.list)


    def __iter__(self):
        return self

    def __next__(self):

        if self.point+self.batchSize<=self.len:#如果够一个batch的走法
            target_list=self.list[self.point:self.point+self.batchSize]
            target_lab=self.lab[self.point:self.point+self.batchSize]

        elif self.point<self.len and not self.drop_last:
            target_list = self.list[self.point:]
            target_lab=self.lab[self.point:]
        else:
            self.point=0
            raise StopIteration
        self.point+=self.batchSize
        return target_list,target_lab


    def shuffle(self,feature,lab):
        """
        :param feature: 一张列表，每个元素是一个L*featureSize的张量
        :param lab: 一张列表，每个元素是一个batch大小的张量
        :return: 随机打乱的特征和lab，是相对应上的
        """
        lists = []
        for index, one_feature in enumerate(feature):
            lists.append((one_feature, lab[index]))
        np.random.shuffle(lists)

        target_feature = []
        target_lab = []
        for i, one_feature in enumerate(lists):
            target_lab.append(one_feature[1])
            target_feature.append(one_feature[0])
        return target_feature, target_lab


def read_fasta(fasta_path):
    fi=open(fasta_path,'r')
    dicts={}
    while True:
        one_line=fi.readline()
        if not one_line:break
        one_index=one_line.index("|")
        t = one_line[one_index + 1:].index("|")
        pro_name=one_line[one_index+1:one_index+t+1]
        seq=fi.readline().replace("\n","")
        dicts[pro_name]=seq
    fi.close()
    return dicts


def fasta2dict(path):
    dicts={}
    fi=open(path,'r')
    while True:
        one_line=fi.readline()
        if not one_line:break
        if one_line[0]=='>':
            index=one_line[4:].find('|')
            pro_name=one_line[4:][0:index]
            dicts[pro_name]=''
        else:
            dicts[pro_name]+=one_line.replace('\n','')
    return dicts


if __name__ == '__main__':
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    pro="A6NCS4"
    seq="MLLSPVTSTPFSVKDILRLERERSCPAASPHPRVRKSPENFQYLRMDAEPRGSEVHNAGGGGGDRKLDGSEPPGGPCEAVLEMDAERMGEPQPGLNAASPLGGGTRVPERGVGNSGDSVRGGRSEQPKARQRRKPRVLFSQAQVLALERRFKQQRYLSAPEREHLASALQLTSTQVKIWFQNRRYKCKRQRQDKSLELAGHPLTPRRVAVPVLVRDGKPCLGPGPGAPAFPSPYSAAVSPYSCYGGYSGAPYGAGYGTCYAGAPSGPAPHTPLASAGFGHGGQNATPQGHLAATLQGVRAW"
    lens=58
    token=token_represent(pro,seq,lens,alphabet.tok_to_idx)
    token.random_mask(token.token_list_represent,0.15)
    loader=data_loader(token.list_token_to_str(),token.masked_position_lab,2)
    for input,masked_pl in loader:
        print(input)
        print(masked_pl)

