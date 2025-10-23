# DILI_AI
AI agent that predicts hepatotoxicity using ChEMBL and other data. The model leverages molecular fingerprints, topological features, and physicochemical descriptors, with scaffold-based splitting to ensure generalization to unseen chemical structures.

To test DILI_AI you should first download files from https://disk.360.yandex.ru/d/soBWfDHb3r-Fkg and put them into folder ```/models```. Then setup ```python 3.10.0``` and ```requirements.txt``` including UniCore: https://github.com/dptech-corp/Uni-Core 
You can look at the example of using our model in the file ```example.ipynb```. 

References:
https://doi.org/10.21203/rs.3.rs-4926613/v1
