import numpy as np
import torch
from vocabulary import Vocab


class BatchIterator:
    def __init__(self, dataset, batch_size=None, vocab_created=False, vocab=None, target_col=None, word2index=None,
             sos_token='<SOS>', eos_token='<EOS>', unk_token='<UNK>', pad_token='<PAD>', min_word_count=5,
             max_vocab_size=None, max_seq_len=0.8, use_pretrained_vectors=False, glove_path='Glove/',
             glove_name='glove.6B.100d.txt', weights_file_name='Glove/weights.npy'):

        if not vocab_created:
            self.vocab = Vocab(dataset, target_col=target_col, word2index=word2index, sos_token=sos_token, eos_token=eos_token,
                               unk_token=unk_token, pad_token=pad_token, min_word_count=min_word_count,
                               max_vocab_size=max_vocab_size, max_seq_len=max_seq_len,
                               use_pretrained_vectors=use_pretrained_vectors, glove_path=glove_path,
                               glove_name=glove_name, weights_file_name=weights_file_name)

            self.dataset = self.vocab.dataset

        else:
            self.dataset = dataset
            self.vocab = vocab

        self.target_col = target_col

        self.word2index = self.vocab.word2index

        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = len(self.dataset)

        self.x_lengths = np.array(self.vocab.x_lengths)

        if self.target_col:
            self.y_lengths = np.array(self.vocab.y_lengths)

        self.pad_token = self.vocab.word2index[pad_token]

        self.sort_and_batch()


    def sort_and_batch(self):
        if not self.target_col:
            sorted_indices = np.argsort(self.x_lengths)
        else:
            sorted_indices = np.lexsort((self.y_lengths, self.x_lengths))

        self.sorted_dataset = self.dataset[sorted_indices[::-1]]
        self.sorted_x_lengths = np.flip(self.x_lengths[sorted_indices])

        if self.target_col:
            self.sorted_target = self.sorted_dataset[:, self.target_col]
            self.sorted_y_lengths = np.flip(self.x_lengths[sorted_indices])
        else:
            self.sorted_target = self.sorted_dataset[:, -1]

        self.input_batches = [[] for _ in range(self.sorted_dataset.shape[1]-1)]

        self.target_batches, self.x_len_batches = [], []

        self.y_len_batches = [] if self.target_col else None

        for i in range(self.sorted_dataset.shape[1]-1):
            if i == 0:
                self.create_batches(self.sorted_dataset[:, i], self.input_batches[i], pad_token=self.pad_token)
            else:
                self.create_batches(self.sorted_dataset[:, i], self.input_batches[i])

        if self.target_col:
            self.create_batches(self.sorted_target, self.target_batches, pad_token=self.pad_token)
            self.create_batches(self.sorted_y_lengths, self.y_len_batches)
        else:
            self.create_batches(self.sorted_target, self.target_batches)

        self.create_batches(self.sorted_x_lengths, self.x_len_batches)

        self.indices = np.arange(len(self.input_batches[0]))
        np.random.shuffle(self.indices)

        for j in range(self.sorted_dataset.shape[1]-1):
            self.input_batches[j] = [self.input_batches[j][i] for i in self.indices]

        self.target_batches = [self.target_batches[i] for i in self.indices]
        self.x_len_batches = [self.x_len_batches[i] for i in self.indices]

        if self.target_col:
            self.y_len_batches = [self.y_len_batches[i] for i in self.indices]

        print('Batches created')


    def create_batches(self, sorted_dataset, batches, pad_token=-1):
        n_batches = int(len(sorted_dataset)/self.batch_size)

        list_of_batches = np.array([sorted_dataset[i*self.batch_size:(i+1)*self.batch_size].copy()\
                                    for i in range(n_batches+1)])

        for batch in list_of_batches:
            tensor_batch = []
            tensor_type = None
            for seq in batch:
                if isinstance(seq, np.ndarray):
                    tensor = torch.LongTensor(seq)
                    tensor_type = 'int'
                elif isinstance(seq, np.integer):
                    tensor = torch.LongTensor([seq])
                    tensor_type = 'int'
                elif isinstance(seq, np.float):
                    tensor = torch.FloatTensor([seq])
                    tensor_type = 'float'
                elif isinstance(seq, int):
                    tensor = torch.LongTensor([seq])
                    tensor_type = 'int'
                elif isinstance(seq, float):
                    tensor = torch.FloatTensor([seq])
                    tensor_type = 'float'
                else:
                    raise TypeError('Cannot convert to Tensor. Data type not recognized')

                tensor_batch.append(tensor)
            if pad_token != -1:
                # Pad required sequences
                pad_batch = torch.nn.utils.rnn.pad_sequence(tensor_batch, batch_first=True)
                batches.append(pad_batch)
            else:
                if tensor_type == 'int':
                    batches.append(torch.LongTensor(tensor_batch))
                else:
                    batches.append(torch.FloatTensor(tensor_batch))


    def __iter__(self):
        to_yield = {}

        for i in range(len(self.input_batches[0])):
            feat_list = []
            for j in range(1, len(self.input_batches)):
                feat = self.input_batches[j][i].type(torch.FloatTensor).unsqueeze(1)
                feat_list.append(feat)

            if feat_list:
                input_feat = torch.cat(feat_list, dim=1)
                to_yield['input_feat'] = input_feat

            to_yield['input_seq'] = self.input_batches[0][i]

            to_yield['target'] = self.target_batches[i]
            to_yield['x_lengths'] = self.x_len_batches[i]

            if self.target_col:
                to_yield['y_length'] = self.y_len_batches[i]


            yield to_yield


    def __len__(self):
        return len(self.input_batches[0])

