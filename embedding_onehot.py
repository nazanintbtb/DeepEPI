import itertools
import numpy as np
import os

def seqtonum(sequence):
  onehot=[]
  for i in range(len(sequence)):
    if(sequence[i]=="A"):
      onehot.append(1.)
    if(sequence[i]=="C"):
      onehot.append(2.)
    if(sequence[i]=="G"):
      onehot.append(3.)
    if(sequence[i]=="T"):
      onehot.append(4.)
    if(sequence[i]=="N"):
      onehot.append(0.)
  return onehot;

names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK','ALL']
name=names[5]
train_dir=f"./data/{name}/"
test_dir=f"./data/{name}/"
Data_dir=f"./data/{name}/"
print ('Experiment on %s dataset' % name)

print ('Loading seq data...')
enhancers_tra=open(train_dir+'%s_enhancer.fasta'%name,'r').read().splitlines()[1::2]
promoters_tra=open(train_dir+'%s_promoter.fasta'%name,'r').read().splitlines()[1::2]
y_tra=np.loadtxt(train_dir+'%s_label.txt'%name)
enhancers_tes=open(test_dir+'%s_enhancer_test.fasta'%name,'r').read().splitlines()[1::2]
promoters_tes=open(test_dir+'%s_promoter_test.fasta'%name,'r').read().splitlines()[1::2]
y_tes=np.loadtxt(test_dir+'%s_label_test.txt'%name)

X_en_tra=[]
X_pr_tra=[]
X_en_tra = np.empty((len(enhancers_tra), 3000))
X_pr_tra = np.empty((len(promoters_tra), 2000))
for i in range(len(enhancers_tra)):
    onehot_en=seqtonum(enhancers_tra[i])
    onehot_pr=seqtonum(promoters_tra[i])
    X_en_tra[i] = onehot_en
    X_pr_tra[i] = onehot_pr

X_en_tes = np.empty((len(enhancers_tes), 3000))
X_pr_tes = np.empty((len(promoters_tes),2000))
for i in range(len(promoters_tes)):
    onehot_en_test = seqtonum(enhancers_tes[i])
    onehot_pr_test = seqtonum(promoters_tes[i])
        
    X_en_tes[i] = onehot_en_test
    X_pr_tes[i] = onehot_pr_test

print("______________&&&&&&&&&&&&&&&&&&&&_________________")
print(np.array(X_en_tra).shape)
print(np.array(X_pr_tra).shape)

np.savez(Data_dir+'%s_train.npz'%name,X_en_tra=X_en_tra, X_pr_tra= X_pr_tra, y_tra=y_tra)
np.savez(Data_dir+'%s_test.npz'%name,X_en_tes=X_en_tes, X_pr_tes= X_pr_tes , y_tes=y_tes)

