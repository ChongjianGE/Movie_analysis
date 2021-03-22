import os
import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
#import seaborn as sns
import timeit
import scattertext as st
import collections
from IPython.display import HTML, IFrame
from textblob import TextBlob
from w3lib.html import remove_tags
from wordcloud import WordCloud
from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
import ipdb


def load_data(path, file_list, dataset, encoding='utf8'):
    for file in file_list:
        with open(os.path.join(path, file), 'r', encoding=encoding) as text:
            dataset.append(text.read())

def process_raw(old_path, new_path):
    train_pos, train_neg, test_pos, test_neg = [], [], [], []
    sets_dict = {'train/positive/': train_pos, 'train/negative/': train_neg,
            'test/positive/': test_pos, 'test/negative/': test_neg}
    for dataset in sets_dict:
            file_list = [f for f in os.listdir(os.path.join(old_path, dataset)) if f.endswith('.txt')]
            load_data(os.path.join(old_path, dataset), file_list, sets_dict[dataset])
    dataset = pd.concat([pd.DataFrame({'review': train_pos, 'label':1}),
                     pd.DataFrame({'review': test_pos, 'label':1}),
                     pd.DataFrame({'review': train_neg, 'label':0}),
                     pd.DataFrame({'review': test_neg, 'label':0})],
                     axis=0, ignore_index=True)
    print(dataset.head())
    duplicate_indices = dataset.loc[dataset.duplicated(keep='first')].index
    print('Number of duplicates in the dataset: {}'.format(dataset.loc[duplicate_indices, 'review'].count()))
    dataset.drop_duplicates(keep='first', inplace=True)
    print('Dataset shape after removing duplicates: {}'.format(dataset.shape))
    dataset.to_csv(os.path.join(new_path, 'dataset_raw/dataset_raw.csv'), index=False)

def polarity(text):
    return TextBlob(text).sentiment.polarity

def subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def pos(df, batch_size, n_threads, required_tags):
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat', '...'])
    # Create empty dictionary
    review_dict = collections.defaultdict(dict)
    for i, doc in enumerate(nlp.pipe(df, batch_size=batch_size, n_threads=n_threads)):
        for token in doc:
            pos = token.pos_
            if pos in required_tags:
                review_dict[i].setdefault(pos, 0)
                review_dict[i][pos] = review_dict[i][pos] + 1
    # Transpose data frame to shape (index, tags)
    return pd.DataFrame(review_dict).transpose()

def extract_features(df, batch_size, n_threads, required_tags):
    df['polarity'] = df.review.apply(polarity).astype('float16')
    df['subjectivity'] = df.review.apply(subjectivity).astype('float16')
    df['word_count'] = df.review.apply(lambda text: len(text.split())).astype('int16')
    df['UPPERCASE'] = df.review.apply(lambda text: len([word for word in text.split() \
                                                        if word.isupper()])) / df.word_count
    df.UPPERCASE = df.UPPERCASE.astype('float16')
    df['DIGITS'] = df.review.apply(lambda text: len([word for word in text.split() \
                                                     if word.isdigit()])) / df.word_count
    df.DIGITS = df.DIGITS.astype('float16')
    pos_data = pos(df.review, batch_size=batch_size, n_threads=n_threads, required_tags=required_tags)
    pos_data = pos_data.div(df.word_count, axis=0).astype('float16')
    return pd.concat([df, pos_data], axis=1)

def split_extract_save(df, name, path, part_size, batch_size, n_threads, required_tags):
    if name not in os.listdir(path):
        dataset_parts = []
        N = int(len(df) / part_size)
        data_frames = [df.iloc[i * part_size:(i + 1) * part_size].copy() for i in range(N + 1)]
        for frame in tqdm_notebook(data_frames):
            dataset_part = extract_features(frame, batch_size=batch_size, n_threads=n_threads,
                                            required_tags=required_tags)
            dataset_parts.append(dataset_part)


        dataset_feat = pd.concat(dataset_parts, axis=0, sort=False)
        dataset_feat.fillna(0, inplace=True)
        dataset_feat.label = dataset_feat.label.astype('int16')
        dataset_feat.to_csv(path + name, index=False)
    else:
        print('File {} already exists in given directory.'.format(name))

def clean_data():
    dataset_feat = pd.read_csv('aclImdb/dataset_feat/dataset_feat.csv')
    dataset_feat = dataset_feat.drop(index=dataset_feat.review[dataset_feat.review == '0'].index)
    dataset_feat.to_csv('aclImdb/dataset_feat/dataset_feat1.csv', index=False)


def token_filter(token):
    return not (token.is_punct | token.is_space | token.is_stop | token.is_digit | token.like_num)


def text_preprocessing(df, batch_size, n_threads):
    df = df.apply(remove_tags)
    df = df.str.lower()
    processed_docs = []
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat', '...'])
    for doc in list(nlp.pipe(df, batch_size=batch_size, n_threads=n_threads)):
        text = [token for token in doc if token_filter(token)]
        text = [token.lemma_ for token in text if token.lemma_ != '-PRON-']
        processed_docs.append(' '.join(text))
    return pd.Series(processed_docs, name='clean_review', index=df.index)


def split_norm_save(df, name, path, part_size, batch_size, n_threads):
    if name not in os.listdir(path):
        dataset_parts = []
        N = int(len(df) / part_size)
        data_frames = [df.iloc[i * part_size:(i + 1) * part_size, 0].copy() for i in range(N + 1)]
        for frame in tqdm_notebook(data_frames):
            dataset_part = text_preprocessing(frame, batch_size=batch_size, n_threads=n_threads)
            dataset_parts.append(dataset_part)


        concat_clean = pd.concat(dataset_parts, axis=0, sort=False)
        dataset_clean = pd.concat([df, concat_clean], axis=1)
        dataset_clean.to_csv(path + name, index=False)
    else:
        print('File {} already exists in given directory.'.format(name))


def train_val_test_split(df, val_size, test_size, random_state=0):
    assert (val_size + test_size) < 1, 'Validation size and test size sum is greater or equal 1'
    assert val_size >= 0 and test_size >= 0, 'Negative size is not accepted'
    train, val, test = np.split(df.sample(frac=1, random_state=random_state),
                                [int((1 - (val_size + test_size)) * len(df)), int((1 - test_size) * len(df))])
    print('Training set shape: {}'.format(train.shape))
    print('Validation set shape: {}'.format(val.shape))
    print('Test set shape: {}'.format(test.shape))

    train.to_csv('aclImdb/dataset_feat_clean/train_feat_clean.csv', index=False)
    val.to_csv('aclImdb/dataset_feat_clean/val_feat_clean.csv', index=False)
    test.to_csv('aclImdb/dataset_feat_clean/test_feat_clean.csv', index=False)

def main():
    old_path = 'C:/Users/chongjiange/Desktop/sentiment-analysis-pytorch-master/aclImdb'
    new_path = 'C:/Users/chongjiange/Desktop/imdb_analysis/aclImdb'
    old_path = 'aclImdb/'
    new_path = 'aclImdb/'
    process_raw(old_path, new_path)
    dataset = pd.read_csv(os.path.join(new_path, 'dataset_raw/dataset_raw.csv'))

    required_tags = ['PROPN', 'PUNCT', 'NOUN', 'ADJ', 'VERB']
    batch_size = 512
    n_threads = 2
    part_size = 5000
    path = os.path.join(os.getcwd(), 'aclImdb/dataset_feat/')
    name = 'dataset_feat.csv'
    split_extract_save(dataset, name, path, part_size, batch_size, n_threads, required_tags)

    clean_data()

    col_types = {'review': str, 'label': np.int16, 'polarity': np.float16, 'subjectivity': np.float16,
                 'word_count': np.int16, 'UPPERCASE': np.float16, 'DIGITS': np.float16, 'PROPN': np.float16,
                 'VERB': np.float16, 'NOUN': np.float16, 'PUNCT': np.float16, 'ADJ': np.float16}
    dataset_feat = pd.read_csv('aclImdb/dataset_feat/dataset_feat1.csv', dtype=col_types)

    batch_size = 512
    n_threads = 2
    part_size = 5000
    path = os.path.join(os.getcwd(), 'aclImdb/dataset_feat_clean/')
    name = 'dataset_feat_clean.csv'
    split_norm_save(dataset_feat, name, path, part_size, batch_size, n_threads)

    dataset_feat_clean = pd.read_csv('aclImdb/dataset_feat_clean/dataset_feat_clean.csv')
    train_val_test_split(dataset_feat_clean, val_size=0.20, test_size=0.10)

if __name__ == '__main__':
    main()
