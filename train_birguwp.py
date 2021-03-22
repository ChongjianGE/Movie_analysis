from batch_iterator import BatchIterator
from early_stopping import EarlyStopping
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from tqdm import tqdm_notebook
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from model.bigruwp import BiGRU
from sklearn.preprocessing import StandardScaler
import ipdb

def data_processing():
    train_dataset = pd.read_csv('aclImdb/dataset_feat_clean/train_feat_clean.csv',
                                usecols=['clean_review', 'polarity', 'subjectivity', 'word_count', 'UPPERCASE',
                                         'DIGITS',
                                         'PROPN', 'VERB', 'NOUN', 'PUNCT', 'ADJ', 'label'])

    train_dataset = train_dataset[['clean_review', 'polarity', 'subjectivity', 'word_count', 'UPPERCASE', 'DIGITS',
                                   'PROPN', 'VERB', 'NOUN', 'PUNCT', 'ADJ', 'label']]
    train_scaler = StandardScaler()
    train_dataset.iloc[:, 3:11] = train_scaler.fit_transform(train_dataset.iloc[:, 3:11])
    val_dataset = pd.read_csv('aclImdb/dataset_feat_clean/val_feat_clean.csv',
                              usecols=['clean_review', 'polarity', 'subjectivity', 'word_count', 'UPPERCASE', 'DIGITS',
                                       'PROPN', 'VERB', 'NOUN', 'PUNCT', 'ADJ', 'label'])
    val_dataset = val_dataset[['clean_review', 'polarity', 'subjectivity', 'word_count', 'UPPERCASE', 'DIGITS',
                               'PROPN', 'VERB', 'NOUN', 'PUNCT', 'ADJ', 'label']]
    val_scaler = StandardScaler()
    val_dataset.iloc[:, 3:11] = val_scaler.fit_transform(val_dataset.iloc[:, 3:11])
    train_iterator = BatchIterator(train_dataset, batch_size=64, vocab_created=False, vocab=None, target_col=None,
                                   word2index=None, sos_token='<SOS>', eos_token='<EOS>', unk_token='<UNK>',
                                   pad_token='<PAD>', min_word_count=3, max_vocab_size=None, max_seq_len=0.9,
                                   use_pretrained_vectors=False, glove_path='glove/', glove_name='glove.6B.100d.txt',
                                   weights_file_name='glove/weights.npy')
    val_iterator = BatchIterator(val_dataset, batch_size=64, vocab_created=False, vocab=None, target_col=None,
                                 word2index=train_iterator.word2index, sos_token='<SOS>', eos_token='<EOS>',
                                 unk_token='<UNK>', pad_token='<PAD>', min_word_count=3, max_vocab_size=None,
                                 max_seq_len=0.9, use_pretrained_vectors=False, glove_path='glove/',
                                 glove_name='glove.6B.100d.txt', weights_file_name='glove/weights.npy')
    test_dataset = pd.read_csv('dataset/datasets_feat_clean/test_feat_clean.csv',
                               usecols=['clean_review', 'polarity', 'subjectivity', 'word_count', 'UPPERCASE', 'DIGITS',
                                        'PROPN', 'VERB', 'NOUN', 'PUNCT', 'ADJ', 'label'])
    test_dataset = test_dataset[['clean_review', 'polarity', 'subjectivity', 'word_count', 'UPPERCASE', 'DIGITS',
                                 'PROPN', 'VERB', 'NOUN', 'PUNCT', 'ADJ', 'label']]
    test_scaler = StandardScaler()
    test_dataset.iloc[:, 3:11] = test_scaler.fit_transform(test_dataset.iloc[:, 3:11])
    test_iterator = BatchIterator(test_dataset, batch_size=256, vocab_created=False, vocab=None, target_col=None,
                                  word2index=train_iterator.word2index, sos_token='<SOS>', eos_token='<EOS>',
                                  unk_token='<UNK>', pad_token='<PAD>', min_word_count=3, max_vocab_size=None,
                                  max_seq_len=0.9, use_pretrained_vectors=False, glove_path='glove/',
                                  glove_name='glove.6B.100d.txt', weights_file_name='glove/weights.npy')
    return train_iterator, val_iterator, test_iterator

def train_model(train_iterator, val_iterator, test_iterator):
    hidden_size = 8
    vocab_size = len(train_iterator.word2index)
    n_extra_feat = 10
    embedding_dim = 200
    output_size = 2
    n_layers = 1
    dropout = 0.5
    learning_rate = 0.001
    epochs = 20
    spatial_dropout = True

    # Check whether system supports CUDA
    CUDA = torch.cuda.is_available()

    model = BiGRU(hidden_size, vocab_size, n_extra_feat, embedding_dim, output_size, n_layers, dropout,
                  spatial_dropout, bidirectional=True)

    # Move the model to GPU if possible
    if CUDA:
        model.cuda()

    model.add_loss_fn(nn.NLLLoss())

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.add_optimizer(optimizer)

    device = torch.device('cuda' if CUDA else 'cpu')

    model.add_device(device)

    # Instantiate the EarlyStopping
    early_stop = EarlyStopping(wait_epochs=2)

    train_losses_list, train_avg_loss_list, train_accuracy_list = [], [], []
    eval_avg_loss_list, eval_accuracy_list, conf_matrix_list = [], [], []

    for epoch in range(epochs):

        print('\nStart epoch [{}/{}]'.format(epoch + 1, epochs))

        train_losses, train_avg_loss, train_accuracy = model.train_model(train_iterator)

        train_losses_list.append(train_losses)
        train_avg_loss_list.append(train_avg_loss)
        train_accuracy_list.append(train_accuracy)

        _, eval_avg_loss, eval_accuracy, conf_matrix = model.evaluate_model(val_iterator)

        eval_avg_loss_list.append(eval_avg_loss)
        eval_accuracy_list.append(eval_accuracy)
        conf_matrix_list.append(conf_matrix)

        print(
            '\nEpoch [{}/{}]: Train accuracy: {:.3f}. Train loss: {:.4f}. Evaluation accuracy: {:.3f}. Evaluation loss: {:.4f}' \
            .format(epoch + 1, epochs, train_accuracy, train_avg_loss, eval_accuracy, eval_avg_loss))

        if early_stop.stop(eval_avg_loss, model, delta=0.003):
            break

    _, test_avg_loss, test_accuracy, test_conf_matrix = model.evaluate_model(test_iterator)
    print('Test accuracy: {:.3f}. Test error: {:.3f}'.format(test_accuracy, test_avg_loss))


def main():
    train_iterator, val_iterator, test_iterator = data_processing()
    train_model(train_iterator, val_iterator, test_iterator)
    print(len(train_iterator.word2index))

if __name__ == '__main__':
    main()
    
    # #debug
    # train_iterator, val_iterator = data_processing()
    # for batches in train_iterator:
    #     # Unpack the dictionary of batches
    #     input_seq, target, x_lengths = batches['input_seq'], batches['target'], batches['x_lengths']
    #     print('input_seq shape: ', input_seq.size())
    #     print('target shape: ', target.size())
    #     print('x_lengths shape: ', x_lengths.size())
    #     break
    #
    # for batches in val_iterator:
    #     # Unpack the dictionary of batches
    #     input_seq, target, x_lengths = batches['input_seq'], batches['target'], batches['x_lengths']
    #     print('input_seq shape: ', input_seq.size())
    #     print('target shape: ', target.size())
    #     print('x_lengths shape: ', x_lengths.size())
    #     break