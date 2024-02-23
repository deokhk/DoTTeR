#!/bin/bash

conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -y pytorch-scatter -c pyg
conda install -y -c conda-forge faiss-gpu
conda install -y -c conda-forge accelerate

pip install transformers==4.25.1
pip install sentence-transformers==2.2.2
pip install scipy
pip install scikit-learn
pip install bottle
pip install nltk
pip install pexpect
pip install prettytable
pip install fuzzywuzzy
pip install dateparser
pip install pathos
pip install pandas 
pip install tensorboard
pip install pyserini
pip install wget 
pip install datasets
pip install seqeval
pip install bitarray
pip install ujson
pip install rank_bm25
pip install wandb
pip install evaluate

echo "Done"