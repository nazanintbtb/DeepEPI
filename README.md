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
**If you want to use embedding_onehot encoding, run (<span style="color:red">python embedding_onehot.py</span>) else if you want to use DNA2Vec encoding, run python DNA2Vec.py**


# CONTACT INFO
Somayyeh Koohi

Department of Computer Engineering

Sharif University of Technology

e-mail: koohi@sharfi.edu

WWW: http://sharif.ir/~koohi/
