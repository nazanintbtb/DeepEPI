# DeepEPI
DeepEPI is a deep learning model designed to advance the study of Enhancer-Promoter Interactions (EPIs) using genomic sequences. By incorporating Convolutional Neural Networks (CNN) and Transformer blocks, our model aims to unravel the complex interplay between enhancers and promoters. DeepEPI employs innovative methods such as embedding layers for one-hot coding and multihead attention mechanisms to enhance the detection and interaction analysis of Transcription Factors (TFs).
# DATASET
**Data_Augmentation.R**

**A tool of data augmentation provided by Mao et al. (2017). The details of the tool can be seen in https://github.com/wgmao/EPIANN.
We used this tool to create and amplify the positive samples in the training set to 20 times to achieve class balance.**
# USAGE
### Need package
python3.9.18,  tensorflow==2.10.1, scikit-learn==1.4.0, numpy==1.26.4, gensim==4.3.2, cuda and cuDNN if you have GPU
###  Preprocess instructions
**If you want to use embedding_onehot encoding, run python (embedding_onehot.py) else if you want to use DNA2Vec encoding, run (python DNA2Vec.py)**
### Train instruction
pyhton train.py [tool name] [cel line name]

tool name can be = DNA2Vec_DeepEPI, embedding_onehot_DeepEPI

cel line name can be = GM12878, HUVEC, HeLa-S3, IMR90, K562, NHEK, ALL

for example run= python train.py embedding_onehot_DeepEPI NHEK

note: make sure save dataset in correct direction
### Test
python test.py [cel line name]

cel line name can be = GM12878, HUVEC, HeLa-S3, IMR90, K562, NHEK, ALL
# DOWNLOAD TRAINED WEIGHTS
Download  best_DNA2Vec_DeepEPI from [here](https://drive.google.com/file/d/1XZRxnyQT0w75ilElmITlEzUSU2jUML99/view?usp=drive_link)

Download  best_embedding_onehot_DeepEPI from [here](https://drive.google.com/file/d/18GLDWqdNA4jXP3cseOhpKQDLeu_abVmM/view?usp=drive_link)

# TF EXTRACTION & TF INTERACTIONS
See the motifs extraction and thier interaction code in [colab](https://colab.research.google.com/drive/1_tL7PddKWJFgNBfTh5Lp33dIUxZOX_8J?usp=sharing)

After feed the motifs to TOMTOM tool and check their interaction by bioGRID .

Download the ready extracted TFs and their interactions [here from supplementary_file4](https://drive.google.com/file/d/14R9kmUfYq3MZL7_Y-E2xpmSwTCDWwVYJ/view?usp=drive_link)

See the code of TFs heatmap from [colab](https://colab.research.google.com/drive/1SWpUxXdYzBljfho7Han-PXtnhwt54__4?usp=sharing)
See the code of TF interactions from [colab](https://colab.research.google.com/drive/17c-Gw1z8hBCa6LqmMhcJUfvhhG6RI56h?usp=sharing) 
# SUPPLEMENTARY FILES
[files](https://drive.google.com/file/d/1EU-QfkpD5BB2-yzjSOTywwNWTS7kzZTG/view?usp=drive_link)

# CONTACT INFO
Somayyeh Koohi

Department of Computer Engineering

Sharif University of Technology

e-mail: koohi@sharfi.edu

WWW: http://sharif.ir/~koohi/
