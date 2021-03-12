from collections import Counter
import pickle


class Vocab(object):
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    def __init__(self, max_vocab_size=-1, data_dir=None):
        self.ast2index = None
        self.index2ast = None

        self.nl2index = None
        self.index2nl = None

        self.code2index = None
        self.index2code = None

        self.max_vocab_size = max_vocab_size
        self.data_dir = data_dir

    def build_vocab(self):
        print('building code vocab....')
        code_tokens = []
        with open(self.data_dir + '/train/code.token', 'r') as f:
            for line in f.readlines():
                code_tokens.append(line.split())
        with open(self.data_dir + '/dev/code.token', 'r') as f:
            for line in f.readlines():
                code_tokens.append(line.split())

        code2index = self.generate_dict(code_tokens, ['<PAD>', '<UNK>'])

        print('building nl vocab....')
        nl_tokens = []
        with open(self.data_dir + '/train/nl.original', 'r') as f:
            for line in f.readlines():
                nl_tokens.append(line.split())
        with open(self.data_dir + '/dev/nl.original', 'r') as f:
            for line in f.readlines():
                nl_tokens.append(line.split())

        nl2index = self.generate_dict(nl_tokens, ['<PAD>', '<UNK>', '<SOS>', '<EOS>'])

        print('building ast vocab....')
        ast_tokens = []
        with open(self.data_dir + '/train/sbt.seq', 'r') as f:
            for line in f.readlines():
                ast_tokens.append(eval(line))
        with open(self.data_dir + '/dev/sbt.seq', 'r') as f:
            for line in f.readlines():
                ast_tokens.append(eval(line))

        ast2index = self.generate_dict(ast_tokens, ['<PAD>', '<UNK>'])

        print('saving vocabs....')
        pickle.dump(ast2index, open(self.data_dir + "/ast_w2i.pkl", "wb"))
        pickle.dump(nl2index, open(self.data_dir + "/nl_w2i.pkl", "wb"))
        pickle.dump(code2index, open(self.data_dir + "/code_w2i.pkl", "wb"))

    def generate_dict(self, tokens, special_tokens=[]):
        word_counter = Counter([x for c in tokens for x in c])
        if self.max_vocab_size < 0:
            words = [x[0] for x in word_counter.most_common()]
        else:
            words = [x[0] for x in word_counter.most_common(self.max_vocab_size - len(special_tokens))]
        w2i = {w: i for i, w in enumerate(special_tokens + words)}
        print('vocab size :', len(w2i))
        return w2i

    def load_vocab(self):
        self.ast2index = read_pickle(self.data_dir + "/ast_w2i.pkl")
        self.nl2index = read_pickle(self.data_dir + "/nl_w2i.pkl")
        self.code2index = read_pickle(self.data_dir + "/code_w2i.pkl")

        self.index2ast = {v: k for k, v in self.ast2index.items()}
        self.index2nl = {v: k for k, v in self.nl2index.items()}
        self.index2code = {v: k for k, v in self.code2index.items()}
        print('loaded ...')


def read_pickle(path):
    return pickle.load(open(path, "rb"))
