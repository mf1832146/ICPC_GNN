import argparse
import os
import pickle
import random
import sys

import numpy as np
import tensorflow as tf

from utils.myutils import BatchGen, init_tf, seq2sent, load_data
import keras
import keras.backend as K
from utils.model import create_model
from timeit import default_timer as timer
from models.custom.graphlayer import GCNLayer
from vocab import Vocab


def gen_pred(model, data, comstok, comlen, batchsize):
    # right now, only greedy search is supported...
    tdats, coms, wsmlnodes, wedge_1 = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wedge_1 = np.array(wedge_1)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, wsmlnodes, wedge_1],
                                batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', type=str, default=None)
    parser.add_argument('--modeltype', dest='modeltype', type=str, default=None)
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--data', dest='dataprep', type=str, default='../data')
    parser.add_argument('--outdir', dest='outdir', type=str, default='modelout/')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=30)
    parser.add_argument('--outfile', dest='outfile', type=str, default=None)

    args = parser.parse_args()

    modelfile = args.model
    outdir = args.outdir
    gpu = args.gpu
    batchsize = args.batchsize
    modeltype = args.modeltype
    outfile = args.outfile

    config = dict()

    # User set parameters#
    config['maxastnodes'] = 100
    config['asthops'] = 2

    if modeltype == None:
        modeltype = modelfile.split('_')[0].split('/')[-1]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    data_dir = '../data_set/py'

    # load vocab
    vocab = Vocab(data_dir=data_dir)
    vocab.load_vocab()

    # load data
    test_code_data, test_ast_data, test_edges, test_nl = load_data(data_dir, 'test')
    test_ids = list(range(len(test_code_data)))

    # code vocab size
    config['tdatvocabsize'] = len(vocab.code2index)
    # comment vocab size
    config['comvocabsize'] = len(vocab.nl2index)
    # ast vocab size
    config['smlvocabsize'] = len(vocab.ast2index)

    # set sequence lengths
    # set sequence length for our input
    # code seq len
    config['tdatlen'] = 100
    # ast seq len
    config['maxastnodes'] = 100
    # comment seq len
    config['comlen'] = 30
    config['batch_size'] = batchsize

    config, _ = create_model(modeltype, config)
    print("MODEL LOADED")
    model = keras.models.load_model(modelfile, custom_objects={"tf": tf, "keras": keras, 'GCNLayer': GCNLayer})

    config['batch_maker'] = 'graph_multi_1'

    print(model.summary())

    # set up prediction string and output file
    comstart = np.zeros(config['comlen'])
    stk = vocab.nl2index['<SOS>']
    comstart[0] = stk
    outfn = outdir + "/predictions/predict-{}.txt".format(modeltype)
    outf = open(outfn, 'w')
    print("writing to file: " + outfn)
    batch_sets = [test_ids[i:i + batchsize] for i in range(0, len(test_ids), batchsize)]

    input_nl_data = dict()

    for c, fid_set in enumerate(batch_sets):
        st = timer()
        for fid in fid_set:
            input_nl_data[fid] = comstart  # np.asarray([stk])

        bg = BatchGen(config, 'test', test_code_data, test_ast_data, input_nl_data, test_edges, vocab)
        batch = bg.make_batch(fid_set)

        batch_results = gen_pred(model, batch, vocab.nl2index, config['comlen'], len(fid_set))

        for key, val in batch_results.items():
            outf.write("{}\t{}\n".format(key, val))

        end = timer()
        print("{} processed, {} per second this batch".format((c + 1) * batchsize, int(batchsize / (end - st))),
              end='\r')

    outf.close()

