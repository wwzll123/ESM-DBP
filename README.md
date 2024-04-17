# ESM-DBP
The data and a user-friendly standalone program of ESM-DBP

# Pre-requisite:
- Python3, numpy(1.20 or higher), pandas, pytorch(1.5 or higher)
- esm(https://github.com/facebookresearch/esm)
- Linux system (suggested CentOS 7)
  
# Installation:
- 1.Download the source code in this repository.
- 2.Download the models of ESM-DBP at https://huggingface.co/zengwenwu/ESM-DBP, and make sure they locate in the same folder.

 # Running
- Enter the following command lines on Linux System.
 ```
 $ python prediciton.py esm_model_dict_dir fasta_path result_dir device
```
- [esm_model_dict_dir]: The path of folder that contains all ESM-DBP models.
- [fasta_path]: The path of query protein sequences in fasta format.
- [result_dir]: The path of folder of prediction results.
- [device]: cuda or cpu
  
# Note
- The feature representation of ESM-DBP will be generated named protein_name.fea.
- The prediction results in .csv format will be generated in $result_dir.
- The pretraining data set UniDBP40 can be download at https://huggingface.co/zengwenwu/UniDBP40.
- If you have any question, please email to wwz_cs@126.com freely.
- All the best to you!

# Reference
[1] Wenwu Zeng, Yutao Dou, Liangrui Pan, Liwen Xu, Shaoliang Peng. Interpretable improving prediction performance of general protein language model by domain-adaptive pretraining on DNA-binding protein. Submitted.
 

