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
from model.transformer import MultiHeadAttention,PositionalEncoding,LabelSmoothingLoss,TransformerBlock,Transformer
import ipdb

def data_processing():
    train_dataset = pd.read_csv('aclImdb/dataset_feat_clean/train_feat_clean.csv',
                                usecols=['clean_review', 'label'])
    train_dataset = train_dataset[['clean_review', 'label']]
    val_dataset = pd.read_csv('aclImdb/dataset_feat_clean/val_feat_clean.csv',
                              usecols=['clean_review', 'label'])
    val_dataset = val_dataset[['clean_review', 'label']]
    batch_size = 32
    train_iterator = BatchIterator(train_dataset, batch_size=batch_size, vocab_created=False, vocab=None,
                                   target_col=None,
                                   word2index=None, sos_token='<SOS>', eos_token='<EOS>', unk_token='<UNK>',
                                   pad_token='<PAD>', min_word_count=3, max_vocab_size=None, max_seq_len=0.9,
                                   use_pretrained_vectors=False, glove_path='glove/', glove_name='glove.6B.100d.txt',
                                   weights_file_name='glove/weights.npy')
    val_iterator = BatchIterator(val_dataset, batch_size=batch_size, vocab_created=False, vocab=None, target_col=None,
                                 word2index=train_iterator.word2index, sos_token='<SOS>', eos_token='<EOS>',
                                 unk_token='<UNK>', pad_token='<PAD>', min_word_count=3, max_vocab_size=None,
                                 max_seq_len=0.9, use_pretrained_vectors=False, glove_path='glove/',
                                 glove_name='glove.6B.100d.txt', weights_file_name='glove/weights.npy')
    test_dataset = pd.read_csv('aclImdb/dataset_feat_clean/test_feat_clean.csv',
                              usecols=['clean_review', 'label'])
    test_dataset = test_dataset[['clean_review', 'label']]
    test_iterator = BatchIterator(test_dataset, batch_size=batch_size, vocab_created=False, vocab=None, target_col=None,
                                 word2index=train_iterator.word2index, sos_token='<SOS>', eos_token='<EOS>',
                                 unk_token='<UNK>', pad_token='<PAD>', min_word_count=3, max_vocab_size=None,
                                 max_seq_len=0.9, use_pretrained_vectors=False, glove_path='glove/',
                                 glove_name='glove.6B.100d.txt', weights_file_name='glove/weights.npy')
    return train_iterator, val_iterator, test_iterator

def train_model(train_iterator, val_iterator, test_iterator):
    batch_size = 32
    vocab_size = len(train_iterator.word2index)
    dmodel = 64
    output_size = 2
    padding_idx = train_iterator.word2index['<PAD>']
    n_layers = 4
    ffnn_hidden_size = dmodel * 2
    heads = 8
    pooling = 'max'
    dropout = 0.5
    label_smoothing = 0.1
    learning_rate = 0.001
    epochs = 30
    CUDA = torch.cuda.is_available()
    max_len=0
    for batches in train_iterator:
        x_lengths = batches['x_lengths']
        if max(x_lengths) > max_len:
            max_len = int(max(x_lengths))
    model = Transformer(vocab_size, dmodel, output_size, max_len, padding_idx, n_layers, \
                        ffnn_hidden_size, heads, pooling, dropout)
    if CUDA:
        model.cuda()

    if label_smoothing:
        loss_fn = LabelSmoothingLoss(output_size, label_smoothing)
    else:
        loss_fn = nn.NLLLoss()
    model.add_loss_fn(loss_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.add_optimizer(optimizer)

    device = torch.device('cuda' if CUDA else 'cpu')

    model.add_device(device)
    params = {'batch_size': batch_size,
              'dmodel': dmodel,
              'n_layers': n_layers,
              'ffnn_hidden_size': ffnn_hidden_size,
              'heads': heads,
              'pooling': pooling,
              'dropout': dropout,
              'label_smoothing': label_smoothing,
              'learning_rate': learning_rate}

    train_writer = SummaryWriter('runs/transformer_train')

    val_writer = SummaryWriter('runs/transformer_val')

    early_stop = EarlyStopping(wait_epochs=3)

    train_losses_list, train_avg_loss_list, train_accuracy_list = [], [], []
    eval_avg_loss_list, eval_accuracy_list, conf_matrix_list = [], [], []
    for epoch in range(epochs):

        try:
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

            train_writer.add_scalar('Training loss', train_avg_loss, epoch)
            val_writer.add_scalar('Validation loss', eval_avg_loss, epoch)

            if early_stop.stop(eval_avg_loss, model, delta=0.003):
                break

        finally:
            train_writer.close()
            val_writer.close()

    _, test_avg_loss, test_accuracy, test_conf_matrix = model.evaluate_model(test_iterator)
    print('Test accuracy: {:.3f}. Test error: {:.3f}'.format(test_accuracy, test_avg_loss))



def main():
    train_iterator, val_iterator, test_iterator = data_processing()
    train_model(train_iterator, val_iterator, test_iterator)
    print(len(train_iterator.word2index))

if __name__ == '__main__':
    main()