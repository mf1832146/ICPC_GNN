import random
import sys
from timeit import default_timer as timer
import random
import keras
import numpy as np

# do NOT import keras in this header area, it will break predict.py
# instead, import keras as needed in each function

start = 0
end = 0


def init_tf(gpu):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def index2word(tok):
    i2w = {}
    for word, index in tok.w2i.items():
        i2w[index] = word
    return i2w


def seq2sent(seq, tokenizer):
    sent = []
    check = index2word(tokenizer)
    for i in seq:
        sent.append(check[i])

    return ' '.join(sent)


class BatchGen(keras.utils.Sequence):
    def __init__(self, config, data_name, code_data, ast_data, nl_data, edges, vocab):
        self.code_data = code_data
        self.ast_data = ast_data
        self.nl_data = nl_data
        self.edges = edges
        self.data_name = data_name
        self.ids = range(len(self.code_data))
        self.vocab = vocab

        self.batch_size = config['batch_size']
        self.max_ast_len = config['maxastnodes']
        self.max_nl_len = config['comvocabsize']
        self.max_code_len = config['tdatvocabsize']

        self.nl_vocab_size = len(self.vocab.nl2index)

        random.random(self.ids)

    def __len__(self):
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, idx):
        start_id = idx * self.batch_size
        end_id = self.batch_size * (idx + 1)
        batch_ids = self.ids[start_id: end_id]
        return self.make_batch(batch_ids)

    def on_epoch_end(self):
        random.shuffle(self.ids)

    def make_batch(self, batch_ids):
        import keras.utils

        batch_data = dict()

        batch_code_seq = []

        batch_ast_seq = []
        batch_edges = []
        batch_nl_seq = []
        batch_target = []

        for _id in batch_ids:
            code_seq = self.code_data[_id]
            nl_seq = self.nl_data[_id]
            ast_seq = self.ast_data[_id]

            code_seq = code_seq[:self.max_code_len]
            ast_seq = ast_seq[:self.max_ast_len]
            nl_seq = ['<SOS>'] + nl_seq + ['<EOS>']

            ast_seq = ast_seq + ['<PAD>' for i in range(self.max_ast_len - len(ast_seq))]

            code_seq_ids = [self.vocab.code2index[x] if x in self.vocab.code2index else self.vocab.code2index['<UNK>'] for x in code_seq]
            ast_seq_ids = [self.vocab.ast2index[x] if x in self.vocab.ast2index else self.vocab.ast2index['<UNK>'] for x in ast_seq]
            nl_seq_ids = [self.vocab.nl2index[x] if x in self.vocab.nl2index else self.vocab.nl2index['<UNK>'] for x in nl_seq]

            edge = np.zeros((self.max_ast_len, self.max_ast_len), dtype='int32')

            if self.data_name == 'test':
                batch_data[_id] = [code_seq_ids, nl_seq_ids, ast_seq_ids, edge]
            else:
                for i in range(1, len(nl_seq_ids)):
                    batch_code_seq.append(code_seq_ids)
                    batch_ast_seq.append(ast_seq_ids)
                    batch_edges.append(edge)

                    input_nl = nl_seq_ids[: i]
                    target = nl_seq_ids[i]

                    target = keras.utils.to_categorical(target, num_classes=self.nl_vocab_size)

                    input_nl = input_nl + [self.vocab.PAD for i in range(len(nl_seq_ids) - 1 - i)]

                    batch_nl_seq.append(input_nl)
                    batch_target.append(np.asarray(target))

        batch_code_seq = np.asarray(batch_code_seq)
        batch_ast_seq = np.asarray(batch_ast_seq)
        batch_edges = np.asarray(batch_edges)
        batch_nl_seq = np.asarray(batch_nl_seq)

        batch_target = np.asarray(batch_target)

        if self.data_name == 'test':
            return batch_data
        else:
            return [[batch_code_seq, batch_nl_seq, batch_ast_seq, batch_edges],
                    batch_target]
