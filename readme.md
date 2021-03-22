# Movie Reviews Analysis via Bidirectional Gated Recurrent Neural Network and Self-Attention Transformer

The repository will walk you through the process of re-implementing our project. Please follow the following illustration step by step. We also provide the pre-trained model link in the bottom.

### Dataset

Dataset is available under the following link:
<http://ai.stanford.edu/~amaas/data/sentiment/>

Please download it.

Then unpack the downloaded *tar.gz* file using:

`tar -xzf acllmdb.tar.gz`

Rearrange the data to the following structure:

    aclImdb
      ├── test
      │     ├── positive
      │     ├── negative
      ├── train
            ├── positive
            └── negative

Here are two choice for you to preprocess the data. We recommend to adopt the second one.

1. preprocess the raw_data from scratch by:

	`python data.py`
	
2. just simply download the file from the [link](https://drive.google.com/drive/folders/19RgUVCWkrbRkaewUChLvyP9wrdT4Qa6w?usp=sharing), which contains the data preprocessing results, and replace with the original file.

Finally, download the pretrained word embedding from [link](https://drive.google.com/drive/folders/1qWNm8fUCudcW99abrpAiw3Zi9eVBOExw?usp=sharing), adn replace the glove file.
### Set up the environment
We recommend to run this code in the anaconda virtual environment
1. Install anaconda from this [website](https://www.anaconda.com/)
1. Create a virtual environment in anaconda and activate it

	`conda create -n text_analysis python=3.6`
	
	`conda activate text_analysis`

2. Install pytorch and torchvision framework, we here give the example of installing pytorch and torchvision with cudatoolkit

    `conda install pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch`


5. Install requirements.

	`pip install matplotlib`

    `pip install pandas==0.25.1`
    
    `pip install spacy==2.1.8`
    
    `pip install scattertext==0.0.2.52`
    
    `pip install numpy`
    
    `pip install tqdm`
    
    `pip install w3lib`
    
    `pip install wordcloud==1.5.0`

    `pip install seaborn==0.9.0`
    
    `pip install textblob==0.15.3`
    
    `pip install ipython==7.8.0`
    
    `pip install scikit_learn==0.21.3`
    
    `pip install tensorboardX`
    
    `pip install ipdb`
    
    `pip install ipywidgets`   
    
### Train the model anc check its performance
To train the BiGRU-Gwo model and check the performace, please use the following script:

    `python train_birguw.py` 

To train the BiGRU-Gw model and check the performace, please use the following script:

    `python train_birguwp.py` 

To train the BiGRU-G model and check the performance, please use the following script:

    `python train_birgug.py` 

To train the  transformer model and check the performance, please use the following script:

    `python train_transformer.py` 

### Model Performance

Model  | Dataset | Test accuracy | Validation accuracy | Training accuracy | Link
------------- | :---: | :---: |:---: | :---: | :---: 
BiGRU-Gwo  | IMDB| 0.908 |0.878 | 0.879 | [ckpt](https://drive.google.com/drive/folders/1gKIc95mNRUx2491x7pNX8p9gGSDwIcbu?usp=sharing)
BiGRU-Gw | IMDB | 0.911 | 0.882 | 0.882 | [ckpt](https://drive.google.com/drive/folders/1gKIc95mNRUx2491x7pNX8p9gGSDwIcbu?usp=sharing)
BiGRU-G | IMDB | 0.844 | 0.846 | 0.861 | [ckpt](https://drive.google.com/drive/folders/1gKIc95mNRUx2491x7pNX8p9gGSDwIcbu?usp=sharing)
Transformer | IMDB | 0.930 | 0.884 | 0.85 | [ckpt](https://drive.google.com/drive/folders/1gKIc95mNRUx2491x7pNX8p9gGSDwIcbu?usp=sharing)

For the convenience, you can get the trained model from the [ckpt link](https://drive.google.com/drive/folders/1gKIc95mNRUx2491x7pNX8p9gGSDwIcbu?usp=sharing)
